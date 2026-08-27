"""Persistence and resume.

Saving after every step is not about tidiness: it is so that **the crash that most
needed the data** leaves something behind. Save only at the end and the run that
crashed is exactly the one with nothing to show.
"""

import json

from teacup_agent import loop, persist
from teacup_agent.evals import ScriptedWithSummarizer, tool_results_follow_their_call
from teacup_agent.memory import NullMemory
from teacup_agent.model import ScriptedModel, assistant_calls, assistant_says


def test_state_is_saved_after_every_step(tmp_path):
    loop.run(
        "persistence test",
        ScriptedModel(
            [
                assistant_calls([("calculate", {"expression": "1+1"})]),
                assistant_calls([("calculate", {"expression": "2+2"})]),
                assistant_says("ok"),
            ]
        ),
        memory=NullMemory(),
        run_dir=tmp_path,
    )
    saved = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert saved["status"] == "done"
    assert len(saved["trace"]) == 2
    assert saved["messages"][1]["content"] == "persistence test"


def test_nothing_written_when_run_dir_is_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loop.run("no persistence", ScriptedModel([assistant_says("ok")]), memory=NullMemory(), run_dir=None)
    assert list(tmp_path.iterdir()) == []


def test_resume_continues_without_redoing_work(tmp_path):
    # First leg: only one step allowed, so it must hit the ceiling
    first = loop.run(
        "resume test",
        ScriptedModel([assistant_calls([("calculate", {"expression": "1+1"})])] * 3),
        memory=NullMemory(),
        max_steps=1,
        run_dir=tmp_path,
    )
    assert first.status == "max_steps"
    tool_calls_before = len(first.trace)

    # Second leg: load it back from disk and grant a few more turns
    resumed = persist.load(tmp_path)
    assert resumed.step == 1 and resumed.trace  # the first leg's work is still there
    resumed.max_steps = resumed.step + 3

    second = loop.run(
        "resume test",
        ScriptedModel([assistant_says("finished on the second leg")]),
        memory=NullMemory(),
        resume=resumed,
        run_dir=tmp_path,
    )

    assert second.status == "done" and second.answer == "finished on the second leg"
    assert second.step == 2  # continues from turn 2 rather than starting over
    assert len(second.trace) == tool_calls_before  # completed tool calls not redone
    assert second.messages[1]["content"] == "resume test"  # the original goal survives
    assert tool_results_follow_their_call(second)  # protocol intact after resume


def test_resume_keeps_the_system_prefix_byte_identical(tmp_path):
    first = loop.run(
        "prefix test",
        ScriptedModel([assistant_calls([("calculate", {"expression": "1+1"})])] * 3),
        memory=NullMemory(),
        max_steps=1,
        run_dir=tmp_path,
        today="2026-08-26",
    )
    resumed = persist.load(tmp_path)
    resumed.max_steps += 2
    model = ScriptedModel([assistant_says("ok")])
    loop.run("prefix test", model, memory=NullMemory(), resume=resumed, run_dir=tmp_path, today="2099-01-01")

    # Resume must not rebuild the system message: that changes the prefix and voids
    # every prompt-cache entry earned so far.
    assert model.calls[0][0]["content"] == first.messages[0]["content"]
    assert "2099" not in model.calls[0][0]["content"]


def test_todo_survives_the_roundtrip_as_objects(tmp_path):
    """asdict() flattens nested dataclasses on the way out; load() has to rebuild
    them, or everything downstream sees dicts and breaks on attribute access."""
    state = loop.run(
        "research X and email it",
        ScriptedWithSummarizer(
            [assistant_calls([("update_todo", {"index": 1, "status": "done"})]), assistant_says("ok")],
            plan_items=["research X", "email the result"],
        ),
        memory=NullMemory(),
        plan=True,
        run_dir=tmp_path,
    )
    assert [t.done for t in state.todo] == [True, False]

    back = persist.load(tmp_path)
    assert [t.text for t in back.todo] == ["research X", "email the result"]
    assert back.todo[0].done is True and back.todo[1].done is False
