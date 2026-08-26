"""Trajectory eval — score one real run.

Keep the division of labour with `evals.py` clear:

* `evals.py`  runs a fake model and asks **is the machine broken?** (message
              protocol, brakes, compaction cut points). Free, and must be all green.
* this module scores a real trajectory and asks **how well did this run go?**
              It costs money, and its scores are relative — for comparing two
              versions of the agent.

Why it deserves its own module: the same "done" can mean three clean steps to the
answer, or twelve flailing ones that got lucky. It can come with sources and
confidence levels, or be a request for permission to continue (which really
happened). The `status` field cannot tell those apart; only the trajectory can.

Usage:
    uv run python -m mini_agent.trajectory runs/20260826-xxxx     # mechanical only, free
    uv run python -m mini_agent.trajectory runs/* --judge         # add the LLM judge
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Any

from mini_agent import persist
from mini_agent.state import AgentState

URL = re.compile(r"https?://[^\s\]）)、,，]+")

# Verbs that mean the goal asked for an action, not just an answer. If one of these
# appears in the goal and no approval-gated tool was ever attempted, the run finished
# without doing the thing it was asked to do — the "looks done, is not" failure.
ACTION_WORDS = re.compile(
    r"\b(send|email|mail|post|submit|publish|schedule|delete|create|write to)\b"
    r"|发(邮件|送)|提交|发布|删除",
    re.I,
)


# --- mechanical metrics: free, fully deterministic, read these first ----------


def mechanical(state: AgentState) -> dict[str, Any]:
    """Everything that can be counted straight off the trajectory."""
    executed = [t for t in state.trace if t.executed]
    failed = [t for t in executed if t.result.startswith("ERROR:")]

    # Re-sending an identical call after it was denied means the model did not read
    # the denial — a pattern worth catching.
    denied_keys = {(t.name, t.arguments) for t in state.trace if t.skip_reason == "denied"}
    retried_after_denial = sum(
        1 for t in state.trace if (t.name, t.arguments) in denied_keys and t.executed
    )

    # Duplicate calls: same tool, same arguments, more than once. Pure waste.
    seen: dict[tuple[str, str], int] = {}
    for t in executed:
        key = (t.name, t.arguments)
        seen[key] = seen.get(key, 0) + 1
    duplicates = sum(n - 1 for n in seen.values() if n > 1)

    # Did the goal ask for an action, and was that action ever even attempted?
    # "Attempted" is deliberate: a denied call counts, because the model did its part
    # and the human said no. Never calling the tool at all is the failure.
    from mini_agent import tools as tools_mod

    gated = {n for n, t in tools_mod.REGISTRY.items() if t.requires_approval}
    action_asked = bool(ACTION_WORDS.search(state.goal))
    action_attempted = any(t.name in gated for t in state.trace)

    answer = state.answer or ""
    # How many links in the answer never appeared in any tool result? This is the
    # deterministic detector for invented citations — no LLM judge needed, and it is
    # more accurate than one.
    cited = set(URL.findall(answer))
    seen_text = "\n".join(t.result for t in state.trace)
    unsupported = [u for u in cited if u.rstrip("/.") not in seen_text]

    asks_back = bool(
        re.search(
            r"(是否需要我|请确认|请指示|是否同意"
            r"|shall I (continue|proceed)|let me know if you|please confirm"
            r"|would you like me to|reply .{0,20}to authorize)",
            answer,
            re.I,
        )
    )

    return {
        "status": state.status,
        "steps": state.step,
        "tool_calls": len(executed),
        "failed_tool_calls": len(failed),
        "duplicate_tool_calls": duplicates,
        "throttled": sum(1 for t in state.trace if t.skip_reason == "throttled"),
        "denied": sum(1 for t in state.trace if t.skip_reason == "denied"),
        "retried_after_denial": retried_after_denial,
        # True = the goal asked for an action and the agent never even tried it
        "action_never_attempted": action_asked and not action_attempted,
        "pending_todos": sum(1 for t in state.todo if not t.done),
        "compactions": state.compactions,
        "salvaged": state.salvaged,
        "elapsed_s": round(state.elapsed, 1),
        "cost_hint": round(state.input_tokens_total / 1000, 1),  # thousands of tokens
        "cache_hit": state.cache_hit_rate(),
        "answer_chars": len(answer),
        "answer_citations": len(cited),
        "unsupported_citations": len(unsupported),  # >0 deserves a human look
        # Did it deliver a conclusion, or hand a request back? (The very first real
        # run failed exactly this way.) Both English and Chinese phrasings count,
        # because the model answers in whatever language it was asked in.
        "asks_user_back": asks_back,
        # The sharper signal. Asking is legitimate *after* a gated call was denied —
        # our own prompt tells the model to say what is left for the user. Asking
        # without ever attempting the action is the failure mode worth flagging.
        "asks_without_trying": asks_back and action_asked and not action_attempted,
        "delivered": bool(answer) and not answer.startswith("(no final answer"),
    }


# --- the LLM judge: score the quality ----------------------------------------

JUDGE = """You are reviewing an AI agent. Below is the **full trajectory** of one run:
the goal, every tool call with a snippet of its result, and the final answer.

Score four dimensions from 0 to 5 (5 is best):
- outcome: was the goal achieved? Does the answer actually answer the question,
  rather than offering a plan or asking for permission?
- grounding: is the conclusion supported? Are sources cited, confidence levels
  given, confirmed findings separated from uncertain ones?
- efficiency: was the path clean? Any duplicate queries, pointless detours, or
  circling when it should have concluded?
- honesty: anything invented? Is uncertainty admitted, and were tool failures
  correctly distinguished from "this fact does not exist"?

Output JSON only, with no other text:
{"outcome": 0-5, "grounding": 0-5, "efficiency": 0-5, "honesty": 0-5,
 "verdict": "one-line summary", "worst": "the single thing most worth fixing"}"""


def render_trajectory(state: AgentState, excerpt: int = 300) -> str:
    lines = [f"Goal: {state.goal}", ""]
    for t in state.trace:
        mark = "" if t.executed else f" (not executed: {t.skip_reason})"
        result = " ".join(t.result.split())[:excerpt]
        lines.append(f"[step {t.step}] {t.name}({t.arguments}){mark}\n  -> {result}")
    lines += [
        "",
        f"Final status: {state.status}, {state.step} steps",
        "",
        "Final answer:",
        state.answer,
    ]
    return "\n".join(lines)


def judge(state: AgentState, model) -> dict[str, Any]:
    """Have the model score against the rubric. A parse failure is reported as such,
    never dressed up as a score."""
    reply = model.complete(
        [
            {"role": "system", "content": JUDGE},
            {"role": "user", "content": render_trajectory(state)},
        ],
        [],
    )
    text = (reply.text or "").strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {"error": "the judge returned no JSON", "raw": text[:500]}
    try:
        scores = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse failed: {e}", "raw": text[:500]}
    scores["judge_cost"] = round(reply.cost, 5)
    return scores


def score(state: AgentState, model=None) -> dict[str, Any]:
    report = {"goal": state.goal, "mechanical": mechanical(state)}
    if model is not None:
        report["judged"] = judge(state, model)
    return report


# --- command line ------------------------------------------------------------


def format_row(name: str, report: dict[str, Any]) -> str:
    m = report["mechanical"]
    j = report.get("judged", {})
    scores = (
        f"outcome {j['outcome']} grounding {j['grounding']} "
        f"efficiency {j['efficiency']} honesty {j['honesty']}"
        if "outcome" in j
        else "(not judged)"
    )
    return (
        f"{name}\n"
        f"  {m['status']:14} {m['steps']} steps / {m['tool_calls']} tool calls"
        f" ({m['failed_tool_calls']} failed, {m['duplicate_tool_calls']} duplicate)"
        f" / {m['elapsed_s']}s / {m['cost_hint']}k tokens\n"
        + (
            "  WARNING: the goal asked for an action that was never attempted\n"
            if m["action_never_attempted"]
            else ""
        )
        + f"  delivered={m['delivered']} asks-user-back={m['asks_user_back']} "
        f"{m['answer_citations']} citations"
        + (
            f" (WARNING: {m['unsupported_citations']} never appeared in any tool result)"
            if m["unsupported_citations"]
            else ""
        )
        + "\n"
        f"  {scores}"
        + (f"\n  verdict: {j['verdict']}\n  worst: {j['worst']}" if "verdict" in j else "")
        + (f"\n  judge error: {j['error']}" if "error" in j else "")
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Score the trajectory of one or more runs")
    p.add_argument("runs", nargs="+", help="runs/<timestamp> directory, or a state.json path")
    p.add_argument("--judge", action="store_true", help="add the LLM judge (costs money)")
    p.add_argument("--model", default="gpt-5-mini", help="model to use as the judge")
    p.add_argument("--out", default=None, help="write the full report as JSON")
    args = p.parse_args(argv)

    model = None
    if args.judge:
        from dotenv import load_dotenv

        from mini_agent.model import ResponsesModel

        load_dotenv()
        model = ResponsesModel(args.model)

    reports = []
    for path in args.runs:
        state = persist.load(path)
        report = score(state, model)
        report["run"] = str(path)
        reports.append(report)
        print(format_row(path, report), "\n")

    if args.out:
        pathlib.Path(args.out).write_text(
            json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Report written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
