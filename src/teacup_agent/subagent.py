"""Subagents — the one form of context compression that loses nothing.

Compaction (context.py) summarises and therefore discards. Externalizing keeps the
text but still spends the excerpt. A subagent avoids the cost entirely: it reads five
pages in a context of its own and hands back three sentences. The parent never pays
for the pages, because it never sees them.

That is the whole mechanism, and the whole risk. What comes back is a conclusion the
parent cannot audit without opening the child's trace, so:

* the child's messages never touch the parent's context, but its trace and its own
  `state.json` are written under the parent's run directory;
* the child gets a slice of the parent's budget and its own step ceiling, and every
  dollar it spends is charged to the parent;
* the child cannot delegate further. One level, enforced by leaving `delegate` out of
  the tool list it is given, because a recursion here is a recursion that spends money.

One parent step buys a whole child run. That is the trade being made: a step is cheap,
a context window is not.
"""

from __future__ import annotations

import pathlib
from typing import Any

from teacup_agent import tools as tools_mod

NAME = "delegate"

DESCRIPTION = (
    "Hand a self-contained subtask to a fresh agent with its own context, and get back "
    "only its conclusion. Use this when a subtask needs a lot of reading (several pages, "
    "a long file) whose details you do not need afterwards — the bulk stays out of your "
    "context. State the task completely: the subagent cannot see this conversation."
)

PARAMETERS = {
    "type": "object",
    "properties": {
        "task": {
            "type": "string",
            "description": "The complete, self-contained task. Include every detail the "
            "subagent needs; it starts from a blank context.",
        },
        "wanted": {
            "type": "string",
            "description": "What the answer should contain, e.g. 'three bullet points "
            "with source links'.",
        },
    },
    "required": ["task"],
}

_config: dict[str, Any] | None = None


def enable(
    parent_state,
    model,
    *,
    max_steps: int = 4,
    budget_share: float = 0.4,
    approve=None,
    run_dir: pathlib.Path | None = None,
    tool_timeout: float = 30.0,
    on_event=None,
    timeout: float = 300.0,
) -> None:
    """Register the delegate tool for one run.

    budget_share is the fraction of the parent's *remaining* budget a single subagent
    may spend. It is read at call time, not now, so a parent that has already spent
    most of its budget cannot fund an expensive child.
    """
    global _config
    _config = {
        "state": parent_state,
        "model": model,
        "max_steps": max_steps,
        "budget_share": budget_share,
        "approve": approve,
        "run_dir": run_dir,
        "tool_timeout": tool_timeout,
        "on_event": on_event,
        "count": 0,
    }
    tools_mod.REGISTRY[NAME] = tools_mod.Tool(
        name=NAME,
        description=DESCRIPTION,
        parameters=PARAMETERS,
        fn=_delegate,
        requires_approval=False,
        timeout=timeout,  # a child run is not a page fetch; it needs longer
    )


def disable() -> None:
    global _config
    _config = None
    tools_mod.REGISTRY.pop(NAME, None)


def _delegate(task: str, wanted: str = "") -> str:
    from teacup_agent import loop  # imported here: loop imports tools, tools imports us

    if _config is None:
        return "ERROR: delegation is not enabled for this run"

    parent = _config["state"]
    _config["count"] += 1
    index = _config["count"]

    budget = round(parent.remaining_budget * _config["budget_share"], 6)
    if budget <= 0:
        return "ERROR: no budget left to fund a subagent; answer from what you have"

    goal = f"{task}\n\nWhat the answer must contain: {wanted}" if wanted else task
    child_dir = _config["run_dir"] / f"sub{index:02d}" if _config["run_dir"] else None

    # The child gets every tool except this one. One level of delegation only.
    child = loop.run(
        goal=goal,
        model=_config["model"],
        max_steps=_config["max_steps"],
        budget=budget,
        time_budget=parent.time_left(),
        tool_timeout=_config["tool_timeout"],
        run_dir=child_dir,
        approve=_config["approve"] or loop.deny_all,
        exclude_tools=[NAME],
        on_event=_child_events(index, _config["on_event"]),
    )

    # Money and tokens are the parent's, wherever they were spent.
    parent.charge(budget - child.remaining_budget)
    parent.input_tokens_total += child.input_tokens_total
    parent.cached_tokens_total += child.cached_tokens_total
    parent.subagent_runs += 1

    if not child.answer:
        return f"ERROR: the subagent finished without an answer (status: {child.status})"
    note = "" if child.status == "done" else f" [subagent stopped early: {child.status}]"
    return f"{child.answer}{note}"


def _child_events(index: int, parent_on_event):
    """Forward the child's events with a marker, so a watching human can see the run
    happening without the parent's context being touched by it."""
    if parent_on_event is None:
        return None

    def on_event(event: str, data: dict[str, Any]) -> None:
        parent_on_event(event, {**data, "subagent": index})

    return on_event
