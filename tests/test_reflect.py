"""reflect.py: trigger conditions, the reflection call, and end-to-end via loop.run().

Mirrors test_plan.py's shape: unit tests for the pure logic, then a few runs through
the real loop with a scripted model.
"""

from __future__ import annotations

from teacup_agent import loop, reflect
from teacup_agent.memory import Memory, NullMemory
from teacup_agent.model import ScriptedModel, assistant_calls, assistant_says
from teacup_agent.state import AgentState, TodoItem, ToolTrace

CLEAN_METRICS = {
    "pending_todos": 0,
    "duplicate_tool_calls": 0,
    "action_never_attempted": False,
    "failed_tool_calls": 0,
    "delivered": True,
}


def _state(status="done", salvaged=False):
    return AgentState(goal="x", status=status, answer="the answer", salvaged=salvaged)


# --- should_reflect: pure trigger logic ------------------------------------------


def test_a_clean_finish_earns_an_experience_note():
    assert reflect.should_reflect(_state(), CLEAN_METRICS) == (True, False)


def test_a_pending_todo_blocks_the_experience_note():
    metrics = {**CLEAN_METRICS, "pending_todos": 1}
    assert reflect.should_reflect(_state(), metrics)[0] is False


def test_a_duplicate_call_blocks_the_experience_note():
    metrics = {**CLEAN_METRICS, "duplicate_tool_calls": 1}
    assert reflect.should_reflect(_state(), metrics)[0] is False


def test_an_untried_action_blocks_the_experience_note():
    metrics = {**CLEAN_METRICS, "action_never_attempted": True}
    assert reflect.should_reflect(_state(), metrics)[0] is False


def test_a_salvaged_run_never_earns_an_experience_note():
    """A forced wrap-up is not a clean finish, whatever the mechanical metrics say."""
    assert reflect.should_reflect(_state(salvaged=True), CLEAN_METRICS)[0] is False


def test_a_recovered_error_earns_a_lesson():
    metrics = {**CLEAN_METRICS, "failed_tool_calls": 1}
    assert reflect.should_reflect(_state(), metrics) == (True, True)


def test_an_error_with_no_delivered_answer_is_not_a_lesson_yet():
    """An error that was never worked around is just a failed run, not a lesson."""
    metrics = {**CLEAN_METRICS, "failed_tool_calls": 1, "delivered": False}
    assert reflect.should_reflect(_state(), metrics)[1] is False


# --- maybe_record: the model call and the write ----------------------------------


def _clean_trajectory_state():
    state = _state()
    state.trace = [ToolTrace(step=1, name="search_web", arguments="{}", result="found it")]
    return state


def test_maybe_record_writes_an_experience_note(tmp_path):
    state = _clean_trajectory_state()
    model = ScriptedModel([], reflection={"experience": "one search was enough here"})
    memory = Memory(tmp_path / "memory.json")
    written = reflect.maybe_record(state, model, memory)
    assert written == ["experience"]
    assert memory.notes == [{"kind": "experience", "text": "one search was enough here"}]


def test_maybe_record_writes_a_lesson_after_a_recovered_error(tmp_path):
    state = _clean_trajectory_state()
    state.trace.append(ToolTrace(step=1, name="fetch", arguments="{}", result="ERROR: timeout"))
    model = ScriptedModel([], reflection={"lesson": "retry fetch with a shorter timeout"})
    memory = Memory(tmp_path / "memory.json")
    written = reflect.maybe_record(state, model, memory)
    assert "lesson" in written
    assert {"kind": "lesson", "text": "retry fetch with a shorter timeout"} in memory.notes


def test_maybe_record_writes_nothing_for_a_messy_run(tmp_path):
    state = _clean_trajectory_state()
    state.todo = [TodoItem("unfinished", done=False)]
    model = ScriptedModel([], reflection={"experience": "should never be written"})
    memory = Memory(tmp_path / "memory.json")
    assert reflect.maybe_record(state, model, memory) == []
    assert memory.notes == []


def test_maybe_record_writes_nothing_when_the_model_omits_the_field(tmp_path):
    """The trigger fired, but the model's JSON left the key out — write nothing rather
    than a blank note."""
    state = _clean_trajectory_state()
    model = ScriptedModel([], reflection={})
    memory = Memory(tmp_path / "memory.json")
    assert reflect.maybe_record(state, model, memory) == []


def test_maybe_record_degrades_silently_when_the_model_call_fails(tmp_path):
    class Explodes(ScriptedModel):
        def complete(self, messages, tools):
            raise RuntimeError("network down")

    state = _clean_trajectory_state()
    memory = Memory(tmp_path / "memory.json")
    assert reflect.maybe_record(state, Explodes([]), memory) == []


def test_maybe_record_charges_its_own_cost():
    state = _clean_trajectory_state()
    before = state.remaining_budget
    model = ScriptedModel([], reflection={"experience": "worked"})
    reflect.maybe_record(state, model, NullMemory())
    assert state.remaining_budget < before


# --- end to end, through loop.run() -----------------------------------------------


def test_a_clean_live_run_writes_an_experience_note_via_the_loop():
    model = ScriptedModel(
        script=[assistant_says("42")],
        reflection={"experience": "the calculation needed no research"},
    )
    memory = NullMemory()
    loop.run("compute 6*7", model, memory=memory, reflect=True)
    assert memory.notes and memory.notes[0]["kind"] == "experience"


def test_a_run_that_hits_the_step_ceiling_writes_no_note():
    """status != "done" fails should_reflect's very first check, whatever the
    mechanical metrics say — a run that never really finished earns nothing."""
    model = ScriptedModel(
        script=[assistant_calls([("calculate", {"expression": "1+1"})])] * 3,
        reflection={"experience": "should never appear"},
    )
    memory = NullMemory()
    state = loop.run("x", model, memory=memory, max_steps=2, reflect=True)
    assert state.status != "done"
    assert memory.notes == []


def test_reflect_off_by_default_writes_nothing():
    model = ScriptedModel(script=[assistant_says("42")], reflection={"experience": "x"})
    memory = NullMemory()
    loop.run("compute 6*7", model, memory=memory)  # reflect defaults to False
    assert memory.notes == []
