"""The Anthropic Messages backend: shape translation verified with a fake client.

No network, no SDK, no cost. What matters here are the five shape differences from
the OpenAI paths — the system prompt is a parameter, tools use `input_schema`,
output is a block list, a tool result is a user message, and consecutive same-role
turns have to be merged.
"""

import json
from types import SimpleNamespace

import pytest

from teacup_agent.model import AnthropicModel, ToolCall


def _fake_client(blocks, usage=None):
    sent = {}

    class Messages:
        def create(self, **kwargs):
            sent.update(kwargs)
            return SimpleNamespace(content=blocks, usage=usage)

    return SimpleNamespace(messages=Messages()), sent


CHAT_TOOL_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "arithmetic",
            "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}},
        },
    }
]


def test_tools_are_flattened_and_use_input_schema():
    client, sent = _fake_client([])
    AnthropicModel("claude-sonnet-5", client=client).complete([], CHAT_TOOL_SPEC)
    assert sent["tools"] == [
        {
            "name": "calculate",
            "description": "arithmetic",
            "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}},
        }
    ]
    assert "function" not in json.dumps(sent["tools"])  # no nested OpenAI layer


def test_the_first_system_message_becomes_the_system_parameter():
    client, sent = _fake_client([])
    AnthropicModel(client=client).complete(
        [{"role": "system", "content": "you are an agent"}, {"role": "user", "content": "hi"}], []
    )
    assert sent["system"] == [{"type": "text", "text": "you are an agent"}]
    assert sent["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert all(m["role"] != "system" for m in sent["messages"])


def test_a_mid_run_system_note_stays_where_it_is_as_a_user_turn():
    """Run-status notes and completion push-backs are feedback about the turn that
    just happened. Hoisting them into the system prompt would move them away from it."""
    client, sent = _fake_client([])
    AnthropicModel(client=client).complete(
        [
            {"role": "system", "content": "persona"},
            {"role": "user", "content": "goal"},
            {"role": "assistant", "content": [{"type": "text", "text": "working"}]},
            {"role": "system", "content": "[run status] 3 steps left"},
        ],
        [],
    )
    assert sent["system"] == [{"type": "text", "text": "persona"}]
    assert sent["messages"][-1] == {
        "role": "user",
        "content": [{"type": "text", "text": "[run status] 3 steps left"}],
    }


def test_consecutive_same_role_turns_are_merged():
    """A tool result followed by a run-status note is two user turns in this repo's
    model and one here; sent separately the API rejects them."""
    client, sent = _fake_client([])
    m = AnthropicModel(client=client)
    m.complete(
        [
            {"role": "system", "content": "persona"},
            {"role": "user", "content": "goal"},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            m.tool_result_item(ToolCall(id="tu_1", name="calculate", arguments="{}"), "42"),
            {"role": "system", "content": "[run status] 2 steps left"},
        ],
        [],
    )
    roles = [msg["role"] for msg in sent["messages"]]
    assert roles == ["user", "assistant", "user"]  # not user, assistant, user, user
    assert sent["messages"][-1]["content"] == [
        {"type": "tool_result", "tool_use_id": "tu_1", "content": "42"},
        {"type": "text", "text": "[run status] 2 steps left"},
    ]


def test_tool_result_item_is_a_user_message_keyed_by_tool_use_id():
    """Pinned deliberately: this is the shape the whole round-trip depends on, and it
    is a `user` message rather than a role of its own the way both OpenAI paths use."""
    item = AnthropicModel(client=_fake_client([])[0]).tool_result_item(
        ToolCall(id="toolu_abc", name="calculate", arguments='{"expression": "1+1"}'), "2"
    )
    assert item == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_abc", "content": "2"}],
    }


def test_a_tool_call_round_trips_through_the_messages_shape():
    """The definition of done: a tool_use block out, a tool_result back in, and the
    assistant turn carried verbatim in between."""
    blocks = [
        {"type": "text", "text": "let me compute that"},
        {"type": "tool_use", "id": "toolu_1", "name": "calculate", "input": {"expression": "2+2"}},
    ]
    client, sent = _fake_client(blocks, usage=SimpleNamespace(input_tokens=100, output_tokens=20))
    m = AnthropicModel("claude-sonnet-5", client=client)

    reply = m.complete([{"role": "system", "content": "p"}, {"role": "user", "content": "2+2?"}], CHAT_TOOL_SPEC)
    assert reply.text == "let me compute that"
    assert len(reply.tool_calls) == 1
    call = reply.tool_calls[0]
    assert (call.id, call.name) == ("toolu_1", "calculate")
    # arguments is a JSON *string*, because that is what the loop hands to json.loads
    assert json.loads(call.arguments) == {"expression": "2+2"}
    assert reply.items == [{"role": "assistant", "content": blocks}]

    # feed the result back the way the loop does, and the next request is well-formed
    m.complete(
        [{"role": "system", "content": "p"}, {"role": "user", "content": "2+2?"}]
        + reply.items
        + [m.tool_result_item(call, "4")],
        CHAT_TOOL_SPEC,
    )
    assert [msg["role"] for msg in sent["messages"]] == ["user", "assistant", "user"]
    assert sent["messages"][1]["content"][1]["id"] == "toolu_1"
    assert sent["messages"][2]["content"][0]["tool_use_id"] == "toolu_1"


def test_cached_input_is_added_to_the_total_and_named_inside_it():
    """This API reports fresh and cached input separately; estimate_cost wants the
    total with the cached part named within it, or the cached tokens are billed twice."""
    usage = SimpleNamespace(input_tokens=100, output_tokens=10, cache_read_input_tokens=900)
    client, _ = _fake_client([{"type": "text", "text": "hi"}], usage=usage)
    reply = AnthropicModel(client=client, prices=(1.0, 0.1, 5.0)).complete([], [])
    assert reply.input_tokens == 1000  # 100 fresh + 900 cached
    assert reply.cached_tokens == 900
    expected = (100 * 1.0 + 900 * 0.1 + 10 * 5.0) / 1_000_000
    assert reply.cost == pytest.approx(expected)


def test_a_cache_key_marks_the_system_prompt_for_caching():
    """Anthropic's cache is explicit — marked on a block, not inferred from a stable
    prefix — and the system prompt is the part identical on every turn of a run."""
    client, sent = _fake_client([])
    m = AnthropicModel(client=client)
    m.complete([{"role": "system", "content": "persona"}], [])
    assert "cache_control" not in sent["system"][0]

    m.set_cache_key("run-1")
    m.complete([{"role": "system", "content": "persona"}], [])
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_a_run_with_no_tools_sends_no_tools_key():
    """An empty list is not the same as absent for this API."""
    client, sent = _fake_client([])
    AnthropicModel(client=client).complete([{"role": "user", "content": "hi"}], [])
    assert "tools" not in sent


# --- through the real loop ----------------------------------------------------
#
# The tests above call complete() directly, which structurally cannot see a bug that
# only appears in how the loop *uses* the model — the wrap-up turn's empty tool list,
# the protocol invariant across a whole run, what compaction does to this shape. Those
# are the ones that cost real money to find.


def _scripted_client(responses):
    """A client that returns each response in turn, recording every request sent."""
    sent = []

    class Messages:
        def create(self, **kwargs):
            sent.append(kwargs)
            return responses[min(len(sent) - 1, len(responses) - 1)]

    return SimpleNamespace(messages=Messages()), sent


def _blocks(*bs):
    return SimpleNamespace(content=list(bs), usage=SimpleNamespace(input_tokens=10, output_tokens=5))


def test_a_whole_run_keeps_the_tool_call_protocol_intact():
    from teacup_agent import loop
    from teacup_agent.evals import tool_results_follow_their_call
    from teacup_agent.memory import NullMemory

    call = {"type": "tool_use", "id": "tu_1", "name": "calculate", "input": {"expression": "2+2"}}
    client, sent = _scripted_client([_blocks(call), _blocks({"type": "text", "text": "it is 4"})])

    state = loop.run(
        "compute 2+2",
        AnthropicModel(client=client),
        memory=NullMemory(),
        max_steps=4,
        run_dir=None,
        plan=False,
    )
    assert state.status == "done"
    assert tool_results_follow_their_call(state)  # would be vacuously True before the guard learned this shape
    assert state.trace[0].name == "calculate"


def test_the_forced_wrap_up_still_sends_the_tool_definitions():
    """The loop passes an empty tool list on the final turn to stop the model starting
    more work. Dropping the `tools` array outright is a 400 here — "requests which
    include tool_use or tool_result blocks must define tools" — so a run that called a
    tool and then hit its step ceiling would die on the turn meant to rescue its answer.
    """
    from teacup_agent import loop
    from teacup_agent.memory import NullMemory

    call = {"type": "tool_use", "id": "tu_1", "name": "calculate", "input": {"expression": "2+2"}}
    # never stops calling tools, so the run is guaranteed to hit the ceiling
    client, sent = _scripted_client([_blocks(call)])

    loop.run(
        "compute 2+2",
        AnthropicModel(client=client),
        memory=NullMemory(),
        max_steps=2,
        run_dir=None,
        plan=False,
    )
    wrap_up = sent[-1]
    assert "tools" in wrap_up, "the wrap-up turn dropped the tool definitions -> 400"
    assert wrap_up["tool_choice"] == {"type": "none"}  # defined, but may not be used


def test_compaction_never_cuts_a_tool_use_away_from_its_result():
    from teacup_agent import context as ctx

    history = [
        {"role": "user", "content": "goal"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "tu_1", "name": "c", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "2"}]},
    ]
    # index 2 sits between the call and its result and must not be offered
    assert 2 not in ctx.safe_cut_points(history)


def test_the_summarizer_can_see_tool_calls_and_their_results():
    """render() read only a "text" key, so a Messages-API history flattened to nothing
    and compaction summarized a blank document while discarding the real entries."""
    from teacup_agent import context as ctx

    text = ctx.render(
        [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "search_web", "input": {"query": "nvidia"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "NVDA up 3%"}]},
        ]
    )
    assert "search_web" in text and "nvidia" in text
    assert "NVDA up 3%" in text


def test_cache_writes_are_counted_not_dropped():
    """Turn 1 of every cached run uploads the whole system prompt as a cache *write*.
    input_tokens excludes it, so leaving it out made that turn cost almost nothing and
    report a context far smaller than the one actually sent."""
    usage = SimpleNamespace(
        input_tokens=10, output_tokens=2, cache_read_input_tokens=0, cache_creation_input_tokens=5000
    )
    client, _ = _scripted_client([SimpleNamespace(content=[{"type": "text", "text": "hi"}], usage=usage)])
    reply = AnthropicModel(client=client, prices=(3.0, 0.3, 15.0)).complete([], [])
    assert reply.input_tokens == 5010
    assert reply.cost == pytest.approx((5010 * 3.0 + 2 * 15.0) / 1_000_000)


def test_the_guard_catches_a_tool_use_with_no_result():
    """The discriminating input. A well-formed history returns True under both the old
    and the new scanner, so a test built only on one proves nothing about the fix — an
    earlier version of this file made exactly that mistake. A *dangling* tool_use is
    the input where the old scanner said True and the new one says False."""
    from teacup_agent.evals import tool_results_follow_their_call
    from teacup_agent.state import AgentState

    s = AgentState(goal="g")
    s.messages = [
        {"role": "user", "content": "goal"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "tu_1", "name": "c", "input": {}}]},
    ]
    assert not tool_results_follow_their_call(s)


def test_the_guard_catches_a_result_for_a_call_that_was_never_announced():
    """The other half of the new branch: an orphan tool_result. Also old-True/new-False,
    and also untested until now."""
    from teacup_agent.evals import tool_results_follow_their_call
    from teacup_agent.state import AgentState

    s = AgentState(goal="g")
    s.messages = [
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "never_announced", "content": "2"}]},
    ]
    assert not tool_results_follow_their_call(s)


def test_the_wrap_up_works_on_a_fresh_model_as_a_resume_builds_one():
    """_last_tools is per-instance memory guarding a per-history API requirement. A
    resumed run constructs a *new* model over a history full of tool blocks, so the
    memory is empty — and that is exactly the path a run that hit its ceiling takes
    next. The definitions have to come from the history, not from what this object
    happens to remember."""
    hist = [
        {"role": "user", "content": "goal"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "tu_1", "name": "calculate", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "4"}]},
    ]
    client, sent = _scripted_client([_blocks({"type": "text", "text": "4"})])
    AnthropicModel(client=client).complete(hist, [])  # fresh instance, never saw tools

    assert "tools" in sent[0], "a resumed wrap-up dropped the tool definitions -> 400"
    assert [t["name"] for t in sent[0]["tools"]] == ["calculate"]
    assert sent[0]["tool_choice"] == {"type": "none"}
