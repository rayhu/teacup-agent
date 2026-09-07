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
