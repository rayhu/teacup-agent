"""The checklist: decomposing the goal, showing it, and refusing to finish half a task.

This exists because of a real run: "research X, then email me the result" produced
excellent research, no email, and `status: done`. It stopped at turn 6 of 14 with 97%
of the budget unspent, so this was never a resource problem — nothing was keeping
track of the second half of the request.
"""

import json

from teacup_agent import loop, plan, tools
from teacup_agent.evals import ScriptedWithSummarizer
from teacup_agent.memory import NullMemory
from teacup_agent.model import ScriptedModel, assistant_calls, assistant_says
from teacup_agent.state import TodoItem


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


# --- finishing without having written anything -------------------------------


def _coding_run(replies, *, approve=lambda call, spec: True, tmp_path=None):
    """A run with coding tools registered, so edit_file/write_file are in `specs`.

    `approve` defaults to allow: the point of these tests is what the model does with
    a working tool, not the approval gate (denied calls are covered in test_hooks).
    """
    from teacup_agent import coding_tools

    coding_tools.enable()
    try:
        return loop.run(
            "make the change",
            ScriptedModel(replies),
            memory=NullMemory(),
            coding_tools=True,
            plan=False,
            approve=approve,
        )
    finally:
        coding_tools.disable()


def test_finishing_without_changing_a_file_is_pushed_back_once():
    """The checklist branch cannot catch this: a model that never called update_todo
    has an empty todo, so nothing was outstanding. A live coding run stopped at step
    8 of 30 with no edits and "in the next step I'll re-open the three files" as its
    final answer."""
    state = _coding_run([assistant_says("I'll start by re-reading those files now.") for _ in range(5)])
    checks = [m for m in state.messages if "[completion check]" in str(m.get("content", ""))]
    assert len(checks) == 1  # exactly one push-back, never a loop
    assert "without having changed" in checks[0]["content"]
    assert state.completion_checked and state.status == "done"


def test_no_pushback_once_a_file_is_written_and_a_command_has_passed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tools.set_project_root(tmp_path)
    try:
        state = _coding_run(
            [
                assistant_calls([("write_file", {"path": "new.py", "content": "x = 1\n"})]),
                assistant_calls([("run_command", {"command": "true"})]),
                assistant_says("done"),
            ]
        )
    finally:
        tools.set_project_root(None)
    assert (tmp_path / "new.py").exists()  # the write really landed
    assert not state.completion_checked


def test_changing_files_without_running_anything_is_pushed_back(tmp_path, monkeypatch):
    """Writing is not verifying. run_command was available the whole time."""
    monkeypatch.chdir(tmp_path)
    tools.set_project_root(tmp_path)
    try:
        state = _coding_run(
            [assistant_calls([("write_file", {"path": "new.py", "content": "x = 1\n"})])]
            + [assistant_says("all done") for _ in range(4)]
        )
    finally:
        tools.set_project_root(None)
    checks = [m for m in state.messages if "[completion check]" in str(m.get("content", ""))]
    assert len(checks) == 1
    assert "without a single successful command run" in checks[0]["content"]


def test_finishing_while_the_last_command_failed_is_pushed_back(tmp_path, monkeypatch):
    """The defect this exists for: run the suite, watch it go red, report done."""
    monkeypatch.chdir(tmp_path)
    tools.set_project_root(tmp_path)
    try:
        state = _coding_run(
            [
                assistant_calls([("write_file", {"path": "new.py", "content": "x = 1\n"})]),
                assistant_calls([("run_command", {"command": "false"})]),
            ]
            + [assistant_says("task complete") for _ in range(4)]
        )
    finally:
        tools.set_project_root(None)
    checks = [m for m in state.messages if "[completion check]" in str(m.get("content", ""))]
    assert len(checks) == 1
    assert "reported failure" in checks[0]["content"]


def test_a_failure_that_was_fixed_and_rerun_is_not_pushed_back(tmp_path, monkeypatch):
    """Only the most recent command counts — red, fix, green is the workflow we want."""
    monkeypatch.chdir(tmp_path)
    tools.set_project_root(tmp_path)
    try:
        state = _coding_run(
            [
                assistant_calls([("write_file", {"path": "new.py", "content": "x = 1\n"})]),
                assistant_calls([("run_command", {"command": "false"})]),
                assistant_calls([("run_command", {"command": "true"})]),
                assistant_says("fixed and green"),
            ]
        )
    finally:
        tools.set_project_root(None)
    assert not state.completion_checked


def test_a_denied_write_still_counts_as_having_written_nothing(tmp_path, monkeypatch):
    """An attempted edit is not a made edit. If approval denied it, the file on disk
    is unchanged and the push-back is exactly as warranted as if nothing was tried."""
    monkeypatch.chdir(tmp_path)
    tools.set_project_root(tmp_path)
    try:
        state = _coding_run(
            [assistant_calls([("write_file", {"path": "new.py", "content": "x = 1\n"})])]
            + [assistant_says("could not do it") for _ in range(4)],
            approve=lambda call, spec: False,
        )
    finally:
        tools.set_project_root(None)
    assert not (tmp_path / "new.py").exists()
    assert state.completion_checked
