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

    Failures are retried: 3 attempts, sleeping 1s then 2s. "The search failed" and
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



# The hosted backend, and why it is opt-in rather than the default. ddgs scrapes a
# search page: free, key-less, no account, and average at both quality and stability
# — which is exactly right for `auto`, because `auto` is what runs when nobody has
# configured anything. A hosted search is better on both counts and costs real money
# per call, so it is selected explicitly and never fallen back *into*: a mode that
# silently starts spending is a worse surprise than a mediocre result.
_HOSTED_MODEL_ENV = "TEACUP_AGENT_SEARCH_MODEL"
_HOSTED_DEFAULT_MODEL = "gpt-5-mini"
# Published as $10 per 1000 calls; goes stale like everything else in a price constant.
# Charged per call on top of the tokens, so the budget brake sees this tool at all.
_HOSTED_CALL_FEE = 0.01


class _SearchNotConfigured(RuntimeError):
    """The hosted backend cannot run until a human changes something — as opposed to
    the network being unhappy, which is worth retrying. The two must not read alike."""

# The loop's per-tool default is 30s and it cannot cancel a thread already inside an
# HTTP call: on overrun the request still completes and still bills, while the model
# gets an error and retries. A shorter client timeout is what actually stops that.
_HOSTED_TIMEOUT = 20.0

# What the hosted backend has spent since the loop last collected it. A tool function
# has no access to `state`, and threading one in would put run state into every tool
# signature for the sake of a single tool — so the spend is accumulated here and the
# loop drains it after each step (loop.py). Without this the only tool in the repo that
# costs money is invisible to `remaining_budget`, and a run can spend many times its
# stated ceiling while `state.snapshot()` reports the ceiling untouched.
_hosted_spend = 0.0


def take_hosted_spend() -> float:
    """Hand the accumulated hosted-search spend to the caller and reset it."""
    global _hosted_spend
    spent, _hosted_spend = _hosted_spend, 0.0
    return spent


def reset_hosted_spend() -> None:
    """Drop anything not yet collected. Called at the start of a run.

    The accumulator is module state and a process runs many agents — bench.py's whole
    matrix, an A2A server answering task after task. A search whose thread finished
    after its own run ended (execute_calls abandons a timed-out tool rather than
    killing it) would otherwise be charged to whichever run happened to drain next,
    which is both a wrong number and a wrong run.
    """
    global _hosted_spend
    _hosted_spend = 0.0


def _search_hosted_backend(query: str, max_results: int) -> str:
    """OpenAI's hosted web search, via the Responses API.

    OpenAI's specifically, not "whatever provider this run is using": it builds its own
    client and reads OPENAI_API_KEY, so a run whose model profile points at Anthropic or
    a local endpoint still searches through OpenAI — or fails here for want of a key it
    was never told it needed.

    Same tool, same arguments, same numbered list of sources — but not byte-identical
    output: this backend has no per-source snippet, and carries a summary the scraper
    has no equivalent for. Sources come from the response's
    `url_citation` annotations rather than from parsing the prose: a citation the API
    attached is a link it actually used, where a URL scraped out of the text is a
    string the model may have written from memory.

    No throttling here, unlike the scraper: this is a metered API being called
    normally, not a public page being polled faster than it likes.
    """
    from openai import OpenAI  # lazy, same as the ddgs import above

    if not os.getenv("OPENAI_API_KEY"):
        raise _SearchNotConfigured(
            "hosted search needs OPENAI_API_KEY. Set it, or use "
            "TEACUP_AGENT_SEARCH=auto for the key-less backend"
        )
    global _hosted_spend
    model = os.getenv(_HOSTED_MODEL_ENV, _HOSTED_DEFAULT_MODEL)
    resp = OpenAI(timeout=_HOSTED_TIMEOUT).responses.create(
        model=model,
        tools=[{"type": "web_search"}],
        # Forced, not offered. "Like any other tool, the model can choose to search the
        # web or not" — and a model that chooses not to has answered from memory, which
        # is precisely the thing this tool exists to replace. Left optional, the caller
        # cannot tell a searched answer from a recalled one.
        tool_choice="required",
        input=(
            f"Search the web for: {query}\n\n"
            f"Summarise what you find in a few sentences, citing your sources. "
            f"Prefer the {max_results} most relevant and most recent results."
        ),
    )
    done, broken = _search_actions(resp)
    # Billed per search action, not per API call: a reasoning model routinely issues
    # several in one response, and a response that searched none should cost none. The
    # fee is charged after counting, so the "it never searched" branch below no longer
    # bills for a search that did not happen.
    _hosted_spend += _HOSTED_CALL_FEE * done + _hosted_token_cost(resp, model)

    if broken:
        # The API says the search itself failed or was cut short. This is the
        # distinction that has caused real wrong answers here, and the first version of
        # this backend reintroduced it one layer down by only asking *whether* a search
        # item existed and never what it said.
        return (
            f"ERROR: the hosted search did not complete ({broken}). This does **not** "
            "mean the information does not exist, only that the search channel is "
            "unhealthy. Retry later, or reword the query."
        )

    if not done:
        # A successful API call in which no search happened at all — the model answered
        # from memory. Saying "no results" here would tell it the information does not
        # exist, which is the same failure in a different coat.
        return (
            "ERROR: the hosted search did not run a query (the model answered without "
            "searching). This does **not** mean the information does not exist. Retry, "
            "or reword the query."
        )

    sources = _url_citations(resp)[:max_results]
    if not sources:
        # A search ran, completed, and produced nothing citable. Deliberately *not*
        # returning the summary: without a citation there is no way to tell text the
        # search grounded from text the model wrote from memory, and an uncited
        # paragraph presented as a search result is the failure this backend removes.
        return (
            f"The hosted search ran and returned no citable sources for {query!r}. "
            "(The search itself worked; treat this as 'nothing found', not as an error.)"
        )

    summary = (getattr(resp, "output_text", "") or "").strip()
    lines = [f"{i}. {title}\n   {url}" for i, (title, url) in enumerate(sources, 1)]
    parts = ["\n".join(lines)]
    if summary:
        parts.append(f"Summary:\n{summary}")
    return "\n\n".join(parts)


def _search_actions(resp: Any) -> tuple[int, str]:
    """(completed searches, why-it-is-broken) from the response's web_search_call items.

    The status field is the point. It is documented as one of in_progress, searching,
    completed, failed or incomplete — so "a web_search_call item exists" and "a search
    happened" are different claims, and treating the first as the second turns a failed
    search into "there is nothing to find".
    """
    done = 0
    bad: list[str] = []
    for item in getattr(resp, "output", None) or []:
        if getattr(item, "type", None) != "web_search_call":
            continue
        status = getattr(item, "status", None) or "completed"
        if status == "completed":
            done += 1
        elif status in ("failed", "incomplete"):
            bad.append(status)
    return done, ", ".join(sorted(set(bad)))


def _hosted_token_cost(resp: Any, model: str) -> float:
    from teacup_agent.model import estimate_cost

    usage = getattr(resp, "usage", None)
    if usage is None:
        return 0.0
    details = getattr(usage, "input_tokens_details", None)
    return estimate_cost(
        model,
        getattr(usage, "input_tokens", 0) or 0,
        getattr(usage, "output_tokens", 0) or 0,
        getattr(details, "cached_tokens", 0) or 0 if details else 0,
    )


def _url_citations(resp: Any) -> list[tuple[str, str]]:
    """(title, url) pairs from the response's annotations, de-duplicated in order."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for item in getattr(resp, "output", None) or []:
        for block in getattr(item, "content", None) or []:
            for ann in getattr(block, "annotations", None) or []:
                if getattr(ann, "type", None) != "url_citation":
                    continue
                url = (getattr(ann, "url", "") or "").strip()
                if not url or url in seen:
                    continue
                seen.add(url)
                out.append(((getattr(ann, "title", "") or url).strip(), url))
    return out


@tool(
    description=(
        "Search the web. Returns a numbered list of sources — title and link, "
        "with a snippet on the key-less backend and a summary on the hosted one. "
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
    """Four modes, selected by the TEACUP_AGENT_SEARCH environment variable:

    auto (default): the key-less scraper; on failure fall back to the offline
                    corpus and say why.
    web           : scraper only; on failure return an error (so the model never
                    reads a broken search as "this does not exist").
    hosted        : OpenAI's hosted web search, whatever provider the model
                    profile names (better results, costs
                    money per call, needs OPENAI_API_KEY). Errors are reported,
                    never degraded into the corpus — a paid backend quietly
                    answering from a local corpus is worse than saying it failed.
    offline       : local corpus only, zero network calls (evals and unit tests).

    `auto` deliberately does not reach for `hosted` even when a key is present:
    picking the backend that costs money should be a decision someone made, not
    one an unset environment variable made for them.
    """
    mode = os.getenv("TEACUP_AGENT_SEARCH", "auto").lower()
    max_results = max(1, min(int(max_results), 10))

    if mode == "offline":
        return _search_corpus(query)

    if mode == "hosted":
        try:
            return _search_hosted_backend(query, max_results)
        except _SearchNotConfigured as e:
            # A missing key is permanent. Telling the model to "retry later" would send
            # it back to a mode that cannot work until a human changes something, and
            # it would keep going until the step ceiling.
            return (
                f"ERROR: hosted search is not configured ({e}). This is a setup "
                "problem, not a temporary one — retrying will not help. Answer from "
                "what you already have and mark anything unverified as unverified."
            )
        except Exception as e:
            return (
                f"ERROR: hosted search failed ({type(e).__name__}: {e}). This does "
                "**not** mean the information does not exist, only that the search "
                "channel is temporarily unavailable. Retry later, reword the query, "
                "or answer from what you already have and mark this item unverified."
            )

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
    if not parts:
        # relative_to() returns a zero-part path ('.') when the target *is* the
        # project root itself (e.g. list_files(".") in coding_tools.py) — that is
        # never denied, but parts[-1] below would otherwise raise IndexError.
        return False
    return any(fnmatch.fnmatch(parts[-1], pattern) for pattern in DENIED_FILES)


# The project root read_file's boundary is drawn against. Defaulting to None (falling
# back to Path.cwd() when unset) keeps every existing test's behaviour unchanged, since
# none of them call set_project_root(). What changes is that "project root" is now a
# fact `cli.py` can state once and pass in, rather than something read_file recomputes
# from the shell's cwd on every call — the same distinction #14 draws: without a way for
# the two to diverge, "outside the project root but inside the launch directory" was
# never a real, testable case.
_project_root: pathlib.Path | None = None


def set_project_root(root: pathlib.Path | None) -> None:
    global _project_root
    _project_root = root


def _get_project_root() -> pathlib.Path:
    return _project_root or pathlib.Path.cwd().resolve()


def _resolve_project_path(path: str) -> pathlib.Path | str:
    """Resolve `path` against the project root and confirm it stays inside it.
    Returns the resolved absolute Path, or an ERROR string if it escapes the root.

    The one traversal guard read_file and coding_tools.py's list_files/edit_file/
    write_file all call, rather than each re-deriving its own copy: a naive
    `str(target).startswith(str(root))` check (what this used to be) is fooled by a
    sibling directory that shares the root as a string prefix — root=/a/repo,
    target=/a/repo-secrets/f.txt passes that check, because relative_to() is what
    actually rejects it, one line later, by raising. is_relative_to() does the real,
    component-wise check up front instead of relying on that accident of ordering."""
    root = _get_project_root()
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        return "ERROR: only paths inside the current project directory are allowed"
    return target


@tool(
    description=(
        "Read a text file inside the current project directory. "
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
    """Returns the full file, unlike the fixed 2000-char slice this used to take:
    a coding-agent tool that reads its own truncated view before editing is a data-
    loss hazard (see coding_tools.py's edit_file/write_file split), and a second,
    earlier cap here pre-empted the loop's own EXTERNALIZE_OVER/excerpt mechanism
    (loop.py) that already exists to move a long result to disk with a path back —
    read_file's own result could never even reach that threshold before this."""
    target = _resolve_project_path(path)
    if isinstance(target, str):
        return target
    root = _get_project_root()

    if _is_denied(target.relative_to(root)):
        return (
            f"ERROR: {path} holds credentials or saved agent state and is not readable "
            "by this tool. This is a fixed rule, not a permission that can be granted, "
            "so do not try a different spelling of the path. Continue without it, and "
            "say in your answer that the file was needed but could not be read."
        )

    if not target.is_file():
        return f"ERROR: no such file: {path}"
    return target.read_text(encoding="utf-8", errors="replace")


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
    # The schema declares an enum, but nothing enforces it on the way in. Accepting
    # an unrecognised status would settle the item anyway — the exact "silently
    # half-done" failure the checklist exists to prevent, reachable through one
    # malformed argument. Refuse it as a tool result so the model can re-send.
    if status not in ("done", "blocked"):
        return (
            f"ERROR: unknown status {status!r}; nothing was changed. Use 'done' when the "
            "item is finished, or 'blocked' with a note saying why it cannot be."
        )
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
