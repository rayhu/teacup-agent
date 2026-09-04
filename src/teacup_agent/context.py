"""Context engineering — the layer that lets the agent run long tasks.

`state.messages` only ever grows. After 20-30 turns three things can go wrong:
the context window overflows, every turn is resent at full price, and the model's
attention is diluted by stale tool output.

Two complementary mechanisms live here, and **externalizing beats compacting, so it
comes first**:

1. **Externalize**: large tool results are written into the run directory; the
   context keeps an excerpt plus the file path. The model reads the file back with
   read_file when it needs detail. The tokens saved are real and nothing is lost.
2. **Compact**: if the context is still over the limit, an earlier slice is handed
   to the model for summarization and replaced by a single summary message. This is
   lossy, which is why it is the second resort.

Compaction has **two constraints you must not break**, and both end in the same 400:

1. never separate a tool call from its result — a function_call without its output
   fails the next request;
2. in the Responses shape, never separate a `function_call` from the `reasoning` item
   it came with. That one was found the hard way, by the first live bench run: the
   cut landed immediately after a reasoning item, the call it belonged to survived
   into the kept tail, and the API refused the next request with "Item 'fc_...' of
   type 'function_call' was provided without its required 'reasoning' item".

The second constraint is really "do not cut inside a turn". A Responses turn arrives as
a *group* — reasoning item(s), an optional message, then that turn's function_call(s) —
and `state.messages.extend(reply.items)` lays the whole group down at once. Anywhere
inside the group is a cut that orphans part of it.
"""

from __future__ import annotations

import pathlib
import re
from typing import Any

from teacup_agent.model import Model

# --- token estimation --------------------------------------------------------

_CJK = re.compile(r"[㐀-鿿豈-﫿]")


def estimate_tokens(text: str) -> int:
    """Rough token count: ~4 characters per token for Latin text, ~1.5 for CJK.

    Only used to decide whether to compact — never for billing. Billing uses the
    real usage numbers the model returns.
    """
    cjk = len(_CJK.findall(text))
    return int(cjk / 1.5 + (len(text) - cjk) / 4) + 1


def messages_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(estimate_tokens(str(m)) for m in messages)


# --- externalize: big results to disk, excerpt + path in the context ----------

EXCERPT = 600  # characters kept inline in the context


def externalize(result: str, run_dir: pathlib.Path, step: int, index: int, name: str) -> str:
    """Write an oversized tool result to disk and return an excerpt plus its path.

    The model reads the file back with read_file when it wants the detail — more
    faithful than any summarization algorithm, and the extra tool call is paid for
    **only when it is actually needed**.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"step{step:02d}_{index}_{name}.txt"
    path.write_text(result, encoding="utf-8")
    try:
        rel = path.relative_to(pathlib.Path.cwd())
    except ValueError:
        rel = path
    return (
        f"{result[:EXCERPT]}\n\n"
        f"[Result was long ({len(result)} characters) and has been saved to {rel}. "
        f"The text above is only the beginning — read that path with read_file for "
        f"the full content.]"
    )


# --- compact: find a safe cut point, replace early context with one summary ---


def safe_cut_points(messages: list[dict[str, Any]]) -> list[int]:
    """Every position where no tool call is left dangling — the only places where
    cutting cannot break the message protocol.

    Same scan as the ordering invariant in evals: an announced id goes into the set,
    a filled id comes out, and any moment the set is empty is a safe point. Both API
    shapes are recognised.

    Plus one rule for the Responses shape: **only cut on a turn boundary.** A turn
    arrives as a group (reasoning, then an optional message, then that turn's
    function_calls), so a cut directly after a `reasoning` or `message` item can leave a
    later call in the same group without the reasoning item the API demands with it.

    An earlier version of this only checked whether the kept tail *began* with a
    `function_call`, which assumed the reasoning item and its call were adjacent. They
    are not when the model narrates first — `reasoning -> message -> function_call` is
    an ordinary turn, and it walked straight through that check (roadmap Field patch G,
    second attempt). Chat-shaped entries carry no `type`, so nothing changes for them.
    """
    open_ids: set[str] = set()
    points = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                open_ids.add(tc["id"])
        elif msg.get("role") == "tool":
            open_ids.discard(msg.get("tool_call_id"))
        elif msg.get("type") == "function_call":
            open_ids.add(msg["call_id"])
        elif msg.get("type") == "function_call_output":
            open_ids.discard(msg.get("call_id"))
        # Inside a Responses turn: a function_call later in the same group may need
        # this entry's reasoning item, so this is not a boundary.
        mid_turn = msg.get("type") in ("reasoning", "message")
        if not open_ids and not mid_turn:
            points.append(i + 1)  # cutting after this entry is safe
    return points


SUMMARIZER = """You are compacting an AI agent's working context.

Condense the history below into a handover note the agent will use to continue its
own work. You must keep:
- facts and conclusions already verified, together with their source links
- what was tried and what failed (so it does not repeat the same dead ends)
- questions that are still open

Drop: pleasantries, repeated intermediate steps, and claims that later findings
have superseded.

Output the note itself, with no preamble like "Sure, here is the summary"."""


def render(messages: list[dict[str, Any]]) -> str:
    """Flatten a slice of messages into plain text for the summarizer."""
    lines = []
    for m in messages:
        kind = m.get("type") or m.get("role")
        body = m.get("content") or m.get("output") or m.get("arguments") or ""
        if isinstance(body, list):  # rich content from the Responses API
            body = " ".join(str(x.get("text", "")) for x in body if isinstance(x, dict))
        lines.append(f"[{kind}] {str(body)[:1500]}")
    return "\n".join(lines)


def compact(
    state,
    model: Model,
    limit: int,
    keep_recent: int = 8,
    profile: str = "",  # which model profile ran the summary, for the spend breakdown
) -> int:
    """Compact the early context into one summary once the limit is exceeded.

    Returns the estimated tokens saved (0 means nothing was compacted). Kept intact:
    the system message (touching the prefix would void the prompt cache), the
    original goal, and the last `keep_recent` entries.
    """
    messages = state.messages
    if messages_tokens(messages) <= limit:
        return 0

    head = 2  # system + original goal, always kept
    candidates = [p for p in safe_cut_points(messages) if head < p <= len(messages) - keep_recent]
    if not candidates:
        return 0  # no safe cut: better not to compact than to split a call from its result
    cut = candidates[-1]

    before = messages_tokens(messages)
    summary = model.complete(
        [
            {"role": "system", "content": SUMMARIZER},
            {"role": "user", "content": render(messages[head:cut])},
        ],
        [],
    )
    state.charge(summary.cost, profile)
    if not summary.text:
        return 0

    state.messages = messages[:head] + [
        {
            "role": "system",
            "content": (
                f"[context summary] The earlier {cut - head} entries were compacted "
                f"into this:\n{summary.text}"
            ),
        }
    ] + messages[cut:]
    state.compactions += 1
    return before - messages_tokens(state.messages)
