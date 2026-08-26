"""The system prompt, and the per-turn status injection.

Both are direct fixes for an earlier failed run: the model knew neither what day it
was (so it dismissed real news as "suspicious") nor how much budget it had left (so
it handed a to-do list back to a terminal nobody was reading).
"""

from mini_agent import loop
from mini_agent.memory import NullMemory
from mini_agent.model import ScriptedModel, assistant_calls, assistant_says


def _run(script, **kw):
    model = ScriptedModel(list(script))
    state = loop.run("test goal", model, memory=NullMemory(), today="2026-08-25", **kw)
    return model, state


def test_system_prompt_carries_today_and_autonomy_rules():
    _, state = _run([assistant_says("ok")])
    system = state.messages[0]["content"]
    assert "2026-08-25" in system
    # do not let a stale memory override a search result
    assert "trust the search result" in system
    # unattended mode: never ask for permission
    assert "Nobody will answer your questions" in system


def test_status_note_injected_before_every_model_call():
    _, state = _run(
        [assistant_calls([("calculate", {"expression": "1+1"})]), assistant_says("2")],
        max_steps=5,
        budget=0.5,
    )
    notes = [m for m in state.messages if str(m.get("content", "")).startswith("[run status]")]
    assert len(notes) == 2  # two turns, one note each
    assert "turn 1/5" in notes[0]["content"] and "turn 2/5" in notes[1]["content"]
    assert "$0.5" in notes[0]["content"]  # the budget is visible to the model


def test_status_note_is_the_last_thing_model_sees():
    model, _ = _run([assistant_says("ok")])
    assert str(model.calls[0][-1]["content"]).startswith("[run status]")


def test_status_note_does_not_break_prefix_caching():
    """The status line is appended at the end and never touches the system prefix —
    otherwise prompt caching (#2) is void."""
    model, _ = _run(
        [assistant_calls([("calculate", {"expression": "1+1"})]), assistant_says("2")]
    )
    first, second = model.calls[0], model.calls[1]
    assert first[0] == second[0]  # the system message is byte-identical
    assert second[: len(first)] == first  # turn two is a strict extension of turn one


# --- what happens when resources run out -------------------------------------


def test_last_turn_has_no_tools_and_says_so():
    """Wording can be ignored, so the final turn simply gets an empty tool list."""
    model, state = _run(
        [assistant_calls([("calculate", {"expression": "1+1"})]) for _ in range(5)],
        max_steps=2,
    )
    assert model.tool_specs[0]  # turn one has tools
    assert model.tool_specs[1] == []  # turn two (the last) does not
    last_note = [m for m in state.messages if str(m.get("content", "")).startswith("[run status]")][-1]
    assert "this is the FINAL turn" in last_note["content"]


def test_salvage_turn_runs_after_resources_exhausted():
    model, state = _run(
        [
            assistant_calls([("calculate", {"expression": "1+1"})]),
            assistant_calls([("calculate", {"expression": "1+1"})]),
            assistant_says("Best effort conclusion: 1+1=2."),
        ],
        max_steps=2,
    )
    assert state.status == "max_steps" and state.salvaged
    assert state.answer and not state.answer.startswith("(no final answer")
    assert model.tool_specs[-1] == []  # the wrap-up turn gets no tools either
    assert "[forced wrap-up]" in state.messages[-2]["content"]


# --- the time budget ----------------------------------------------------------


def test_status_note_shows_time_left_and_urges_when_tight():
    model = ScriptedModel([assistant_says("ok")])
    state = loop.run(
        "test goal",
        model,
        memory=NullMemory(),
        today="2026-08-26",
        time_budget=60.0,
        clock=iter([0.0, 55.0, 55.0]).__next__,
    )
    note = [m for m in state.messages if str(m.get("content", "")).startswith("[run status]")][0]
    assert "5s left" in note["content"]
    # steps are plentiful, so it is time that is doing the nagging
    assert "Start wrapping up" in note["content"]


def test_time_brake_stops_run_with_budget_and_steps_left():
    """Money and steps both have slack; time still stops the run."""
    state = loop.run(
        "test goal",
        ScriptedModel(
            [
                assistant_calls([("calculate", {"expression": "1+1"})]),
                assistant_says("Conclusion before the timeout: 1+1=2."),
            ]
        ),
        memory=NullMemory(),
        today="2026-08-26",
        time_budget=10.0,
        clock=iter([0.0, 1.0, 30.0, 30.0]).__next__,
    )
    assert state.status == "out_of_time"
    assert state.remaining_budget > 0 and state.step < state.max_steps
    assert state.salvaged and state.answer  # a timeout is not an excuse to come back empty
    assert state.snapshot()["elapsed_s"] == 30.0
