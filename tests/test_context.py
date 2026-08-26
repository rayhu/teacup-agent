"""Context management: externalizing big results, and compacting over the limit.

The dangerous part of compaction is not summary quality, it is **splitting a tool
call from its result** — that makes the next request fail with a 400. So every case
here also re-checks the message-protocol invariant.
"""

import pytest

from mini_agent import context as ctx
from mini_agent import loop, tools
from mini_agent.evals import ScriptedWithSummarizer, tool_results_follow_their_call
from mini_agent.memory import NullMemory
from mini_agent.model import ScriptedModel, assistant_calls, assistant_says


def test_estimate_tokens_weighs_cjk_heavier():
    assert ctx.estimate_tokens("中文" * 100) > ctx.estimate_tokens("ab" * 100)  # CJK is denser


def test_safe_cut_points_never_split_a_call_from_its_result():
    messages = [
        {"role": "system"},
        {"role": "user"},
        {"role": "assistant", "tool_calls": [{"id": "a"}, {"id": "b"}]},
        {"role": "tool", "tool_call_id": "a"},
        {"role": "tool", "tool_call_id": "b"},
        {"role": "assistant", "content": "done"},
    ]
    points = ctx.safe_cut_points(messages)
    assert 3 not in points and 4 not in points  # cutting here splits call from result
    assert 5 in points and 6 in points


def test_safe_cut_points_handles_responses_shape():
    messages = [
        {"role": "system"},
        {"role": "user"},
        {"type": "reasoning"},
        {"type": "function_call", "call_id": "fc_1"},
        {"type": "function_call_output", "call_id": "fc_1"},
    ]
    assert ctx.safe_cut_points(messages) == [1, 2, 3, 5]


# --- externalization ---------------------------------------------------------


@pytest.fixture
def big_tool(monkeypatch):
    registry = dict(tools.REGISTRY)
    registry["dump"] = tools.Tool(
        "dump", "returns a wall of text", {"type": "object", "properties": {}}, lambda **_: "X" * 5000
    )
    monkeypatch.setattr(tools, "REGISTRY", registry)


def test_big_tool_result_is_written_to_disk_and_shrunk(big_tool, tmp_path):
    state = loop.run(
        "externalization test",
        ScriptedModel([assistant_calls([("dump", {})]), assistant_says("ok")]),
        memory=NullMemory(),
        run_dir=tmp_path,
    )
    kept = [m for m in state.messages if m.get("role") == "tool"][0]["content"]
    assert len(kept) < 1200  # only an excerpt is left in the context
    assert "read_file" in kept  # and it tells the model how to get the rest

    files = list(tmp_path.glob("*.txt"))
    assert len(files) == 1 and len(files[0].read_text()) == 5000  # full text intact
    # The trace records what the model actually saw (the trimmed version); the full
    # text lives on disk. Trajectory eval cares about the former; read the file for
    # the details.
    assert state.trace[0].result == kept


# --- compaction --------------------------------------------------------------


def test_compaction_replaces_history_and_keeps_protocol_valid():
    # A few turns of tool calls to inflate the context
    script = [assistant_calls([("calculate", {"expression": "1+1"})]) for _ in range(6)]
    script.append(assistant_says("finished"))
    model = ScriptedWithSummarizer(script)

    state = loop.run(
        "compaction test",
        model,
        memory=NullMemory(),
        max_steps=8,
        context_limit=200,  # deliberately tiny, to force compaction
        run_dir=None,
    )

    assert model.summaries >= 1 and state.compactions >= 1
    assert any("[context summary]" in str(m.get("content", "")) for m in state.messages)
    assert tool_results_follow_their_call(state)  # protocol survives compaction
    assert state.messages[0]["role"] == "system"  # prefix untouched, cache intact
    assert state.messages[1]["content"] == "compaction test"  # the original goal always stays


def test_no_compaction_below_the_limit():
    model = ScriptedWithSummarizer([assistant_says("short task")])
    loop.run("small task", model, memory=NullMemory(), context_limit=100_000)
    assert model.summaries == 0
