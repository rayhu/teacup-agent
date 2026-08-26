"""The checklist: decomposing the goal, showing it, and refusing to finish half a task.

This exists because of a real run: "research X, then email me the result" produced
excellent research, no email, and `status: done`. It stopped at turn 6 of 14 with 97%
of the budget unspent, so this was never a resource problem — nothing was keeping
track of the second half of the request.
"""

import json

from mini_agent import loop, plan, tools
from mini_agent.evals import ScriptedWithSummarizer
from mini_agent.memory import NullMemory
from mini_agent.model import ScriptedModel, assistant_calls, assistant_says
from mini_agent.state import TodoItem


class _Planner(ScriptedModel):
    """A model whose planner reply we control, for parsing tests."""

    def __init__(self, planner_text, script=None):
        super().__init__(list(script or []))
        self.planner_text = planner_text

    def complete(self, messages, tools_):
        if messages and str(messages[0].get("content", "")).startswith("Break the user's"):
            return assistant_says(self.planner_text)
        return super().complete(messages, tools_)


# --- decomposition ------------------------------------------------------------


def test_decompose_reads_a_json_array():
    items = plan.decompose("x", _Planner('["research X", "email the result"]'))
    assert [i.text for i in items] == ["research X", "email the result"]
    assert all(not i.done for i in items)


def test_decompose_tolerates_surrounding_prose():
    items = plan.decompose("x", _Planner('Sure:\n```json\n["a", "b"]\n```'))
    assert [i.text for i in items] == ["a", "b"]


def test_broken_planner_never_stops_the_run():
    """A planner that returns nonsense degrades to the old behaviour, silently."""
    assert plan.decompose("x", _Planner("I could not parse that")) == []


def test_decompose_caps_the_list():
    items = plan.decompose("x", _Planner(json.dumps([f"item {i}" for i in range(20)])))
    assert len(items) == 5


# --- the checklist in the loop ------------------------------------------------


def _run(script, plan_items, **kw):
    model = ScriptedWithSummarizer(list(script), plan_items=plan_items)
    state = loop.run(
        "research X and email it",
        model,
        memory=NullMemory(),
        today="2026-08-26",
        plan=True,
        **kw,
    )
    return model, state


def test_status_line_shows_outstanding_items():
    _, state = _run([assistant_calls([("calculate", {"expression": "1+1"})]), assistant_says("ok")],
                    ["research X", "email the result"])
    note = [m for m in state.messages if str(m.get("content", "")).startswith("[run status]")][0]
    assert "Checklist:" in note["content"]
    assert "[ ] 1. research X" in note["content"]


def test_update_todo_marks_an_item_and_the_status_line_follows():
    _, state = _run(
        [
            assistant_calls([("update_todo", {"index": 1, "status": "done"})]),
            assistant_calls([("calculate", {"expression": "1+1"})]),
            assistant_says("ok"),
        ],
        ["research X", "email the result"],
    )
    assert state.todo[0].done and not state.todo[1].done
    later = [m for m in state.messages if str(m.get("content", "")).startswith("[run status]")][-1]
    assert "[x] 1. research X" in later["content"]


def test_blocked_item_counts_as_settled_with_a_reason():
    """Blocked is a settled state: the item stops being outstanding, but keeps the
    reason so the final answer can say what was left undone and why."""
    _, state = _run(
        [
            assistant_calls([("update_todo", {"index": 2, "status": "blocked", "note": "no address"})]),
            assistant_says("done what I could"),
        ],
        ["research X", "email the result"],
    )
    assert state.todo[1].done and state.todo[1].note == "no address"

    # Item 1 is still open, so the push-back fires — and mentions only item 1.
    check = [m for m in state.messages if "[completion check]" in str(m.get("content", ""))][0]
    assert "research X" in check["content"]
    assert "email the result" not in check["content"]


def test_finishing_with_an_open_item_is_pushed_back_once():
    model, state = _run(
        [assistant_says("all done") for _ in range(5)],
        ["research X", "email the result"],
    )
    checks = [m for m in state.messages if "[completion check]" in str(m.get("content", ""))]
    assert len(checks) == 1  # exactly one push-back, never a loop
    assert "email the result" in checks[0]["content"]
    assert state.completion_checked and state.status == "done"


def test_no_pushback_when_there_is_nothing_outstanding():
    _, state = _run(
        [
            assistant_calls([("update_todo", {"index": 1, "status": "done"})]),
            assistant_calls([("update_todo", {"index": 2, "status": "done"})]),
            assistant_says("both done"),
        ],
        ["research X", "email the result"],
    )
    assert not state.completion_checked


def test_update_todo_rejects_a_bad_index():
    state_todo = [TodoItem("only item")]
    tools.bind_todo(state_todo)
    assert tools.execute("update_todo", '{"index": 9, "status": "done"}').startswith("ERROR:")
    assert not state_todo[0].done


def test_forced_wrapup_names_the_unfinished_items():
    _, state = _run(
        [assistant_calls([("calculate", {"expression": "1+1"})]) for _ in range(5)],
        ["research X", "email the result"],
        max_steps=2,
    )
    wrapup = [m for m in state.messages if "[forced wrap-up]" in str(m.get("content", ""))][0]
    assert "email the result" in wrapup["content"]  # the run admits what it never did
