"""Evals — a health check on the control loop using a scripted model. No API key,
no cost.

These do not measure how smart the model is; they measure whether the **loop** is
correct: does a bad argument heal itself, does every tool call get its result back,
do the ceilings actually stop the car.

Run with: uv run python -m teacup_agent.evals   (or uv run pytest)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable

from teacup_agent import loop
from teacup_agent import routing
from teacup_agent.memory import NullMemory
from teacup_agent.model import ScriptedModel, assistant_calls, assistant_says
from teacup_agent.state import AgentState


# ScriptedModel already answers the framework's side calls (planning, compaction),
# so this name is kept only because tests and examples refer to it.
ScriptedWithSummarizer = ScriptedModel


@dataclass
class Case:
    name: str
    script: list
    check: Callable[[AgentState], bool]
    max_steps: int = 8
    budget: float = 0.05
    max_tool_calls: int = 3
    time_budget: float | None = None
    context_limit: int = 30_000
    subagents: bool = False
    coding_tools: bool = False
    skills: str | None = None
    plan: bool = False
    plan_items: list[str] | None = None
    # role -> profile, when the case splits the run across two scripted models. Every
    # profile other than "main" gets its own empty-script model, so a role that resolves
    # elsewhere provably ran somewhere else.
    roles: dict[str, str] | None = None
    clock_values: list[float] | None = None  # fake clock, makes the time brake repeatable


def tool_results_follow_their_call(state: AgentState) -> bool:
    """Every tool result's id must appear in an entry **before** it that announced
    the call, and every announced id must be filled exactly once. Both API shapes
    are recognised."""
    announced: set[str] = set()
    for msg in state.messages:
        # Chat Completions shape
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                announced.add(tc["id"])
        elif msg.get("role") == "tool":
            if msg.get("tool_call_id") not in announced:
                return False
            announced.discard(msg["tool_call_id"])  # an id may only be filled once
        # Responses shape
        elif msg.get("type") == "function_call":
            announced.add(msg["call_id"])
        elif msg.get("type") == "function_call_output":
            if msg.get("call_id") not in announced:
                return False
            announced.discard(msg["call_id"])
    return not announced  # every announced call must have a result


CASES: list[Case] = [
    Case(
        name="direct answer: the loop ends as soon as the model calls no tools",
        script=[assistant_says("42")],
        check=lambda s: s.status == "done" and s.answer == "42" and s.step == 1,
    ),
    Case(
        name="refill: every tool call in a turn must get a result message",
        script=[
            assistant_calls(
                [
                    ("search_web", {"query": "cuda"}),
                    ("calculate", {"expression": "2+3"}),
                ]
            ),
            assistant_says("done"),
        ],
        check=lambda s: (
            len(s.trace) == 2
            and sum(m["role"] == "tool" for m in s.messages) == 2
            and tool_results_follow_their_call(s)  # trap 1: order cannot be reversed
            and "5" in s.trace[1].result
        ),
    ),
    Case(
        name="self-healing: malformed JSON comes back as an ERROR result, loop continues",
        script=[
            assistant_calls([("calculate", "{broken json")]),
            assistant_calls([("calculate", {"expression": "1+1"})]),
            assistant_says("fixed, it is 2"),
        ],
        check=lambda s: (
            s.status == "done"
            and s.trace[0].result.startswith("ERROR:")
            and s.trace[1].result == "2"
        ),
    ),
    Case(
        name="unknown tool: no crash, tell the model which tools exist",
        script=[assistant_calls([("teleport", {"to": "Mars"})]), assistant_says("another way")],
        check=lambda s: "unknown tool" in s.trace[0].result and s.status == "done",
    ),
    Case(
        name="step ceiling: a model that keeps calling tools must be stopped",
        script=[assistant_calls([("calculate", {"expression": "1+1"})]) for _ in range(10)],
        check=lambda s: s.status == "max_steps" and s.step == 3,
        max_steps=3,
    ),
    Case(
        name="budget ceiling: overspending must stop the run",
        script=[assistant_calls([("calculate", {"expression": "1+1"})], cost=0.02) for _ in range(10)],
        check=lambda s: s.status == "out_of_budget" and s.remaining_budget <= 0,
        budget=0.03,
    ),
    Case(
        name="long-term memory: the remember tool writes the fact into memory",
        script=[
            assistant_calls([("remember", {"fact": "user prefers concise answers"})]),
            assistant_says("noted"),
        ],
        check=lambda s: "Remembered" in s.trace[0].result,
    ),
    Case(
        name="per-turn cap: excess calls are refused but still get a tool message each",
        script=[
            assistant_calls([("calculate", {"expression": f"{i}+1"}) for i in range(5)]),
            assistant_says("enough"),
        ],
        max_tool_calls=2,
        check=lambda s: (
            sum(t.executed for t in s.trace) == 2  # only two actually ran
            and sum(not t.executed for t in s.trace) == 3  # three were held back
            and sum(m["role"] == "tool" for m in s.messages) == 5  # all five ids filled (trap 2)
            and tool_results_follow_their_call(s)
            and s.trace[4].result.startswith("ERROR:")
            and s.status == "done"
        ),
    ),
    Case(
        name="never empty-handed: running out of steps forces a wrap-up conclusion",
        # Two turns of tool calls burn the steps; the wrap-up turn has no tools left
        # and can only conclude.
        script=[
            assistant_calls([("calculate", {"expression": "1+1"})]),
            assistant_calls([("calculate", {"expression": "1+1"})]),
            assistant_says("Best available conclusion: 1+1=2 (high confidence); the rest is unverified."),
        ],
        max_steps=2,
        check=lambda s: (
            s.status == "max_steps"
            and s.salvaged  # the forced wrap-up turn ran
            and s.answer
            and not s.answer.startswith("(no final answer")
        ),
    ),
    Case(
        name="a burnt budget also wraps up instead of printing a stop reason",
        script=[
            assistant_calls([("calculate", {"expression": "1+1"})], cost=0.02),
            assistant_calls([("calculate", {"expression": "1+1"})], cost=0.02),
            assistant_says("Conclusion before the budget ran out: 1+1=2."),
        ],
        budget=0.03,
        check=lambda s: s.status == "out_of_budget" and s.salvaged and s.answer,
    ),
    Case(
        name="time brake: running out of wall-clock time also stops and wraps up",
        script=[
            assistant_calls([("calculate", {"expression": "1+1"})]),
            assistant_calls([("calculate", {"expression": "1+1"})]),
            assistant_says("Out of time, so: 1+1=2."),
        ],
        time_budget=10.0,
        # Fake clock: two turns run, the third check is already over the limit
        # (run() consumes these readings in order).
        clock_values=[0.0, 1.0, 2.0, 12.0, 12.0, 12.0],
        check=lambda s: (
            s.status == "out_of_time"
            and s.step == 2  # money and steps both had slack; time stopped it
            and s.remaining_budget > 0
            and s.salvaged
            and s.answer
        ),
    ),
    Case(
        name="compaction: the message protocol must survive it",
        script=[assistant_calls([("calculate", {"expression": "1+1"})]) for _ in range(5)]
        + [assistant_says("still able to wrap up after compaction")],
        context_limit=200,  # deliberately tiny, to force compaction
        max_steps=7,
        check=lambda s: (
            s.compactions >= 1
            and tool_results_follow_their_call(s)  # no call was split from its result
            and any("[context summary]" in str(m.get("content", "")) for m in s.messages)
            and s.messages[1]["content"] == "compaction: the message protocol must survive it"
        ),
    ),
    Case(
        name="tool calls in the wrap-up turn still need result messages (or resume 400s)",
        script=[
            assistant_calls([("calculate", {"expression": "1+1"})]),
            assistant_calls([("calculate", {"expression": "9+9"})]),  # wrap-up turn tries a tool
        ],
        max_steps=1,
        check=lambda s: (
            s.status == "max_steps"
            and tool_results_follow_their_call(s)  # no dangling tool_call_id
            and any("forced wrap-up" in str(m.get("content", "")) for m in s.messages)
        ),
    ),
    Case(
        name="approval gate: unattended runs deny side effects and never execute them",
        script=[
            assistant_calls([("send_email", {"to": "a@b.c", "subject": "s", "body": "b"})]),
            assistant_says("The email was not sent; you will have to send it yourself."),
        ],
        check=lambda s: (
            s.trace[0].skip_reason == "denied"
            and not s.trace[0].executed
            and "was NOT executed" in s.trace[0].result
            and tool_results_follow_their_call(s)  # a denied call still gets a result
            and s.status == "done"
        ),
    ),
    Case(
        name="approval gate: read-only tools are never gated",
        script=[
            assistant_calls([("calculate", {"expression": "1+1"})]),
            assistant_says("2"),
        ],
        check=lambda s: s.trace[0].executed and s.trace[0].result == "2",
    ),
    Case(
        name="approval gate: a denial is followed by a different tool, not a stall",
        # Pins the SYSTEM_PROMPT/DENIED rule added after a live dogfood run where the
        # model tried run_command to read a file, got denied (unattended default), and
        # gave up rather than using the already-ungated read_file for the same goal.
        # This case scripts the *well-behaved* response the prompt now asks for, and
        # checks the loop actually lets it through end to end — it does not (and
        # cannot, with a scripted model) verify a real model chooses to behave this
        # way; that is a prompt-effectiveness question only a live run can answer.
        coding_tools=True,
        script=[
            assistant_calls([("run_command", {"command": "cat notes.txt"})]),
            assistant_calls([("read_file", {"path": "notes.txt"})]),
            assistant_says("Read via read_file after run_command was denied."),
        ],
        check=lambda s: (
            s.trace[0].name == "run_command"
            and s.trace[0].skip_reason == "denied"
            and not s.trace[0].executed
            and s.trace[1].name == "read_file"
            and s.trace[1].executed  # the fallback tool actually ran, unlike the denied one
            and tool_results_follow_their_call(s)
            and s.status == "done"
        ),
    ),
    Case(
        name="checklist: finishing with an untouched action item gets pushed back once",
        # Turn 1 answers without doing the second item; the completion check fires and
        # the model then sends the email.
        script=[
            assistant_says("Here is the research. I have not sent the email."),
            assistant_calls([("send_email", {"to": "a@b.c", "subject": "s", "body": "b"})]),
            assistant_calls([("update_todo", {"index": 2, "status": "blocked", "note": "denied"})]),
            assistant_says("Research done; the email needs you to send it."),
        ],
        plan=True,
        plan_items=["research the topic", "email the result to a@b.c"],
        check=lambda s: (
            s.completion_checked
            and any("[completion check]" in str(m.get("content", "")) for m in s.messages)
            # the push-back made it actually attempt the gated call
            and any(t.name == "send_email" for t in s.trace)
            and s.status == "done"
        ),
    ),
    Case(
        name="checklist: the push-back happens at most once, never a loop",
        script=[assistant_says("done") for _ in range(6)],
        plan=True,
        plan_items=["research the topic", "email the result"],
        check=lambda s: (
            s.completion_checked
            and s.status == "done"
            and s.step == 2  # one push-back, then the answer stands
            and sum("[completion check]" in str(m.get("content", "")) for m in s.messages) == 1
        ),
    ),
    Case(
        name="checklist: update_todo ticks an item off and it stops being outstanding",
        script=[
            assistant_calls([("update_todo", {"index": 1, "status": "done"})]),
            assistant_calls([("update_todo", {"index": 2, "status": "done"})]),
            assistant_says("both items handled"),
        ],
        plan=True,
        plan_items=["research the topic", "email the result"],
        check=lambda s: (
            all(t.done for t in s.todo)
            and not s.completion_checked  # nothing outstanding, so no push-back
            and s.status == "done"
        ),
    ),
    Case(
        name="checklist: an unrecognised update_todo status is refused, not silently done",
        # The failure this pins down: a bogus status used to set done=True, the item
        # left the pending list, and the run could report finished with the work
        # never attempted.
        script=[
            assistant_calls([("update_todo", {"index": 1, "status": "in_progress"})]),
            assistant_says("first item handled"),
            assistant_says("first item handled"),
        ],
        plan=True,
        plan_items=["research the topic"],
        check=lambda s: (
            any(t.name == "update_todo" and t.result.startswith("ERROR:") for t in s.trace)
            and not s.todo[0].done  # the bogus status did not settle the item
            and s.completion_checked  # so the run pushed back instead of finishing quietly
            and s.status == "done"
        ),
    ),
    Case(
        name="delegation: a child's reading never lands in the parent's messages",
        script=[
            assistant_calls([("delegate", {"task": "CHILD: look up cuda"})]),
            assistant_says("answered from the subagent's conclusion"),
        ],
        subagents=True,
        budget=1.0,
        check=lambda s: (
            s.subagent_runs == 1
            and tool_results_follow_their_call(s)
            # the child ran with its own context; the parent holds one tool result
            and sum(m["role"] == "tool" for m in s.messages) == 1
            and s.status == "done"
        ),
    ),
    Case(
        name="routing: a role sent to another profile runs there, and the protocol holds",
        # The whole risk of routing is that a turn produced by one model and a turn
        # produced by another end up in the same message list. Here the checklist comes
        # from the `planner` profile and the conversation from `main`; the spend
        # breakdown proves both actually ran, and the ordering invariant still holds.
        script=[
            assistant_calls([("calculate", {"expression": "6*7"})]),
            assistant_calls([("update_todo", {"index": 1, "status": "done"})]),
            assistant_says("42"),
        ],
        plan=True,
        plan_items=["compute the product"],
        roles={"plan": "planner"},
        check=lambda s: (
            [t.text for t in s.todo] == ["compute the product"]
            and set(s.spend_by_profile) == {"main", "planner"}
            and s.status == "done"
            and tool_results_follow_their_call(s)
        ),
    ),
    Case(
        name="skills: the body arrives as a tool result, never in the system prompt",
        script=[
            assistant_calls([("load_skill", {"name": "web-research"})]),
            assistant_says("followed the procedure"),
        ],
        skills="skills",
        check=lambda s: (
            s.loaded_skills == ["web-research"]
            # the catalog is in the prefix, the procedure is not
            and "web-research:" in s.messages[0]["content"]
            and "Grade every source" not in s.messages[0]["content"]
            and "Grade every source" in s.trace[0].result
            and tool_results_follow_their_call(s)
        ),
    ),
]


def run_case(case: Case) -> tuple[bool, AgentState]:
    # Evals must be deterministic: force offline search, no network calls.
    os.environ.setdefault("TEACUP_AGENT_SEARCH", "offline")
    main_model = ScriptedWithSummarizer(list(case.script), plan_items=case.plan_items)
    model = main_model
    if case.roles:
        # A separate scripted model per other profile: proof by construction that a
        # routed role did not quietly run on the main one.
        others = {
            name: ScriptedWithSummarizer([], plan_items=case.plan_items)
            for name in set(case.roles.values())
            if name != "main"
        }
        model = routing.from_models({"main": main_model, **others}, case.roles, "main")
    state = loop.run(
        goal=case.name,
        model=model,
        memory=NullMemory(),
        max_steps=case.max_steps,
        budget=case.budget,
        max_tool_calls_per_step=case.max_tool_calls,
        time_budget=case.time_budget,
        context_limit=case.context_limit,
        subagents=case.subagents,
        coding_tools=case.coding_tools,
        skills=case.skills,
        plan=case.plan,
        run_dir=None,  # evals never write to disk
        # A fresh iterator per run, so a case can be executed repeatedly.
        **({"clock": iter(case.clock_values).__next__} if case.clock_values else {}),
    )
    return case.check(state), state


def main() -> int:
    failed = 0
    for case in CASES:
        ok, state = run_case(case)
        print(f"{'PASS' if ok else 'FAIL'}  {case.name}")
        if not ok:
            failed += 1
            print(f"      final state: {state.snapshot()}")
            for t in state.trace:
                print(f"      - {t.name}({t.arguments}) -> {t.result[:80]}")
    total = len(CASES)
    print(f"\n{total - failed}/{total} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
