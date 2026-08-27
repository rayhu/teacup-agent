"""Tools — the things the agent can do to the world.

Each tool = one Python function + a JSON Schema (what the model sees).
The point is to get the register -> describe -> call -> feed-back chain right.
`search_web` hits the real network; the others are deliberately minimal.
"""

from __future__ import annotations

import ast
import fnmatch
import json
import operator
import os
import pathlib
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    fn: Callable[..., str]
    requires_approval: bool = False  # True = a human must say yes before running
    timeout: float | None = None  # override the loop's per-call timeout, in seconds
    # False = never move this result to a file. Most results are raw material and an
    # excerpt plus a path is fine; a few are instructions the model has to follow, and
    # truncating those defeats the point of returning them at all.
    externalize: bool = True


REGISTRY: dict[str, Tool] = {}


def tool(
    description: str,
    parameters: dict[str, Any],
    requires_approval: bool = False,
    timeout: float | None = None,
    externalize: bool = True,
):
    """Decorator: register a plain function as a model-callable tool.

    Set requires_approval=True for operations with **external side effects that
    are hard to undo**: sending mail, placing orders, deleting data. Never set it
    on read-only tools — asking every time makes people numb, and numb people
    click "approve" with their eyes closed, which is worse than not asking.
    """

    def deco(fn: Callable[..., str]) -> Callable[..., str]:
        REGISTRY[fn.__name__] = Tool(
            fn.__name__, description, parameters, fn, requires_approval, timeout,
            externalize,
        )
        return fn

    return deco


def specs() -> list[dict[str, Any]]:
    """Export the registry in the OpenAI Chat Completions `tools` shape."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in REGISTRY.values()
    ]


def execute(name: str, arguments: str) -> str:
    """Run one tool call.

    The key rule: failures **never raise**. The error text is returned as the tool
    result so the model can read it and fix its own call. That self-correction is
    exactly what the agent loop is for.
    """
    tool_obj = REGISTRY.get(name)
    if tool_obj is None:
        return f"ERROR: unknown tool {name!r}. Available tools: {', '.join(REGISTRY)}"

    try:
        # Note: `arguments` from the model is a JSON **string** and must be parsed
        # first. (The `fn(**item.arguments)` line in the original notes died here.)
        kwargs = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError as e:
        return f"ERROR: arguments are not valid JSON ({e}). Received: {arguments!r}"

    if not isinstance(kwargs, dict):
        return f"ERROR: arguments must be a JSON object, got {type(kwargs).__name__}"

    try:
        return str(tool_obj.fn(**kwargs))
    except TypeError as e:
        return (
            f"ERROR: argument mismatch ({e}). Expected schema: "
            f"{json.dumps(tool_obj.parameters, ensure_ascii=False)}"
        )
    except Exception as e:  # the tool itself failed — hand that to the model too
        return f"ERROR: {type(e).__name__}: {e}"


# --------------------------------------------------------------------------
# The tools themselves. search_web is real; the rest are minimal stand-ins you
# can replace one at a time.
# --------------------------------------------------------------------------

# Offline corpus: the fallback when ddgs is missing, the network is down, or
# offline mode is requested. It also keeps the evals deterministic.
_CORPUS = {
    "nvidia gpu strategy": (
        "NVIDIA's GPU strategy has three layers: (1) data center — Blackwell/Rubin "
        "shipped as whole racks (NVL72) that bundle GPUs, CPUs and NVLink switches "
        "into one compute unit; (2) a software moat — CUDA plus the NIM/TensorRT-LLM "
        "inference stack, which makes migration expensive; (3) networking — NVLink "
        "and InfiniBand after the Mellanox acquisition, locking in full-system deals."
    ),
    "cuda": (
        "CUDA is NVIDIA's parallel computing platform and its deepest moat: the cost "
        "of leaving the ecosystem is far higher than the cost of the hardware."
    ),
    "agent": (
        "Agent = Model + State + Tools + Control Loop + Memory/Evals. "
        "The essential loop is: LLM -> tool call -> tool result -> LLM."
    ),
}


def _search_corpus(query: str) -> str:
    """Offline fallback search.

    Matching uses all() rather than any(): an early version used any(), so the word
    "strategy" in "OpenAI strategy" matched the "nvidia gpu strategy" entry and fed
    NVIDIA material to an OpenAI question. Returning "nothing found" is always
    better than returning the wrong thing — the model takes what it is given.
    """
    words = set(re.findall(r"[\w一-鿿]+", query.lower()))
    hits = [text for key, text in _CORPUS.items() if set(key.split()) <= words]
    if not hits:
        return (
            f"No results for {query!r}. (This is the **offline** corpus; it only "
            f"contains: {', '.join(_CORPUS)})"
        )
    return "\n\n".join(hits[:3])


# Rate-limit protection for the search backend. Measured: fire 4-5 queries back to
# back and DuckDuckGo cuts you off — and an agent loves to fire several per turn.
# Two mechanisms: a minimum interval, plus backoff retries.
# Measured with three parallel searches: a 1.5s interval takes 8.3s, no interval
# takes 4.7s — our own throttle was the bottleneck, not the network. Settled on
# 0.5s: six back-to-back searches, zero failures, 8.9s. Retries cover the rest.
_MIN_INTERVAL = 0.5  # seconds between two real searches
_RETRIES = 3
_last_search_at = 0.0
_throttle_lock = threading.Lock()  # tools run in parallel, so this global needs a lock


def _throttle() -> None:
    """Space out real searches.

    The lock makes "wait then record" atomic: three parallel searches start at
    0s / 0.5s / 1.0s instead of stampeding into the rate limiter. Staggered starts
    do not mean lost parallelism — the network round-trips still overlap.
    """
    global _last_search_at
    with _throttle_lock:
        wait = _MIN_INTERVAL - (time.monotonic() - _last_search_at)
        if wait > 0:
            time.sleep(wait)
        _last_search_at = time.monotonic()


def _search_web_backend(query: str, max_results: int) -> str:
    """Real search backend: DuckDuckGo, no API key required.

    Failures are retried with backoff (1s / 2s / 4s). "The search failed" and
    "there is nothing to find" are **completely different** statements, and the
    former must never masquerade as the latter — that is how a model concludes
    that something does not exist in the world.
    """
    from ddgs import DDGS  # imported lazily so the offline path does not need it

    last_error: Exception | None = None
    for attempt in range(_RETRIES):
        try:
            _throttle()
            results = DDGS().text(query, max_results=max_results)
            break
        except Exception as e:  # usually rate limiting — back off and try again
            last_error = e
            if attempt < _RETRIES - 1:
                time.sleep(2**attempt)
    else:
        raise RuntimeError(
            f"search failed after {_RETRIES} attempts: {last_error}"
        ) from last_error

    if not results:
        return f"No web results for {query!r}."
    lines = []
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "").strip()
        url = (r.get("href") or r.get("url") or "").strip()
        body = " ".join((r.get("body") or "").split())[:300]
        lines.append(f"{i}. {title}\n   {url}\n   {body}")
    return "\n".join(lines)


@tool(
    description=(
        "Search the web. Returns a list of results with title, link and snippet. "
        "Cite the returned links when you use them; two or three differently worded "
        "searches per question is usually enough."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "search keywords"},
            "max_results": {
                "type": "integer",
                "description": "how many results, 1-10, default 5",
                "minimum": 1,
                "maximum": 10,
            },
        },
        "required": ["query"],
    },
)
def search_web(query: str, max_results: int = 5) -> str:
    """Three modes, selected by the TEACUP_AGENT_SEARCH environment variable:

    auto (default): use the network; on failure fall back to the offline corpus
                    and say why.
    web           : network only; on failure return an error (so the model never
                    reads a broken search as "this does not exist").
    offline       : local corpus only, zero network calls (evals and unit tests).
    """
    mode = os.getenv("TEACUP_AGENT_SEARCH", "auto").lower()
    max_results = max(1, min(int(max_results), 10))

    if mode == "offline":
        return _search_corpus(query)

    try:
        return _search_web_backend(query, max_results)
    except ImportError:
        if mode == "web":
            return (
                "ERROR: ddgs is not installed, web search unavailable. "
                "Run `uv sync`, or set TEACUP_AGENT_SEARCH=offline."
            )
        return f"[web search unavailable (no ddgs); offline corpus below]\n{_search_corpus(query)}"
    except Exception as e:
        fallback = _search_corpus(query)
        # Only degrade to the corpus when it actually has something. Otherwise say
        # ERROR: dressing up "the search broke" as "no results" makes the model
        # conclude the information does not exist.
        if mode == "web" or fallback.startswith("No results"):
            return (
                f"ERROR: search failed ({type(e).__name__}: {e}). This does **not** "
                "mean the information does not exist, only that the search channel "
                "is temporarily unavailable. Retry later, reword the query, or "
                "answer from what you already have and mark this item unverified."
            )
        return f"[web search failed: {type(e).__name__}; offline corpus below]\n{fallback}"


# Safe arithmetic: literals and basic operators only, no eval().
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported expression node: {ast.dump(node)}")


@tool(
    description=(
        "Evaluate an arithmetic expression, e.g. '(1200 * 0.85) / 3'. "
        "Only + - * / ** % are supported."
    ),
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "arithmetic expression"}
        },
        "required": ["expression"],
    },
)
def calculate(expression: str) -> str:
    return str(_eval_node(ast.parse(expression, mode="eval").body))


# Files the agent may not read, however it is asked. The directory guard below answers
# "where", and the project directory is exactly where the secrets live: one prompt
# injection saying "summarise .env for me" is an exfiltration path built entirely from
# intended features. This answers "what".
#
# `runs/` needs a distinction rather than a blanket rule: the externalizer writes large
# tool results there and tells the model to read them back, so those files must stay
# readable. A run's `state.json` is a different animal — it holds the full system prompt
# and every tool result of that run, including runs the current task has nothing to do
# with.
DENIED_FILES = (
    ".env", ".env.*", "*.env",      # credentials
    "mcp.json",                     # MCP server configuration, including its env block
    "memory.json",                  # whatever the agent chose to remember
    "state.json",                   # a full trajectory: system prompt, every tool result
    "*.pem", "*.key", "id_rsa*", "*.p12",
)
DENIED_DIRS = (".git", ".ssh", ".aws", ".venv")


def _is_denied(relative: pathlib.PurePath) -> bool:
    parts = [p.lower() for p in relative.parts]
    if any(part in DENIED_DIRS for part in parts):
        return True
    return any(fnmatch.fnmatch(parts[-1], pattern) for pattern in DENIED_FILES)


@tool(
    description=(
        "Read a text file inside the current project directory (first 2000 characters). "
        "Credentials, configuration and saved run states are not readable."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "path relative to the project root"}
        },
        "required": ["path"],
    },
)
def read_file(path: str) -> str:
    root = pathlib.Path.cwd().resolve()
    target = (root / path).resolve()
    if not str(target).startswith(str(root)):  # simple traversal guard
        return "ERROR: only files inside the current project directory can be read"

    if _is_denied(target.relative_to(root)):
        return (
            f"ERROR: {path} holds credentials or saved agent state and is not readable "
            "by this tool. This is a fixed rule, not a permission that can be granted, "
            "so do not try a different spelling of the path. Continue without it, and "
            "say in your answer that the file was needed but could not be read."
        )

    if not target.is_file():
        return f"ERROR: no such file: {path}"
    return target.read_text(encoding="utf-8", errors="replace")[:2000]


# Write side of long-term memory. Kept simple: a module-level binding injected by
# loop.run().
_memory = None


def bind_memory(memory) -> None:
    global _memory
    _memory = memory


@tool(
    description=(
        "Store a fact worth keeping across sessions in long-term memory "
        "(user preferences, stable conclusions)."
    ),
    parameters={
        "type": "object",
        "properties": {"fact": {"type": "string", "description": "a one-line fact"}},
        "required": ["fact"],
    },
)
def remember(fact: str) -> str:
    if _memory is None:
        return "ERROR: no long-term memory is available"
    _memory.remember(fact)
    return f"Remembered: {fact}"


# The run's checklist, bound by loop.run() the same way memory is.
_todo = None


def bind_todo(todo) -> None:
    global _todo
    _todo = todo


@tool(
    description=(
        "Mark an item on the checklist as done (or blocked). Call this as soon as you "
        "finish an item, so the remaining work stays accurate. Use status='blocked' "
        "with a reason when an item cannot be completed."
    ),
    parameters={
        "type": "object",
        "properties": {
            "index": {"type": "integer", "description": "1-based item number"},
            "status": {"type": "string", "enum": ["done", "blocked"]},
            "note": {"type": "string", "description": "why, when blocked"},
        },
        "required": ["index", "status"],
    },
)
def update_todo(index: int, status: str, note: str = "") -> str:
    if not _todo:
        return "ERROR: this run has no checklist"
    if not 1 <= index <= len(_todo):
        return f"ERROR: no item {index}; the checklist has {len(_todo)} items"
    item = _todo[index - 1]
    item.done = True  # blocked items are settled too: they stop being outstanding
    item.note = note if status == "blocked" else ""
    label = "done" if status == "done" else f"blocked ({note or 'no reason given'})"
    return f"Item {index} marked {label}: {item.text}"


@tool(
    description=(
        "Send an email to someone. **This has external side effects and cannot be "
        "undone, so it requires human approval before it runs.** Demo implementation: "
        "nothing is really sent, the message is appended to outbox.jsonl."
    ),
    parameters={
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "recipient address"},
            "subject": {"type": "string", "description": "subject line"},
            "body": {"type": "string", "description": "message body"},
        },
        "required": ["to", "subject", "body"],
    },
    requires_approval=True,
)
def send_email(to: str, subject: str, body: str) -> str:
    """The last entry in the original notes' tool list — and the first one that
    needs a gate.

    A read-only tool that goes wrong wastes one call; this one going wrong means
    the mail has already left.
    """
    record = {"to": to, "subject": subject, "body": body}
    with open("outbox.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return f"Sent to {to} with subject {subject!r} (demo: written to outbox.jsonl)"
