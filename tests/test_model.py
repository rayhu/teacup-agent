"""Verify OpenAIModel's parsing with a fake client — no network, no cost.

It covers the three things that are easy to get wrong: how tool_calls are read, how
the assistant message is carried back verbatim, and how cost is computed.
"""

import pytest

from types import SimpleNamespace

from teacup_agent.model import OpenAIModel


class _Msg(SimpleNamespace):
    def model_dump(self, exclude_none=False):
        data = {"role": "assistant", "content": self.content, "tool_calls": self._raw}
        return {k: v for k, v in data.items() if not (exclude_none and v is None)}


def _fake_client(content, tool_calls, usage):
    raw = [
        SimpleNamespace(
            id=tc["id"],
            type="function",
            function=SimpleNamespace(name=tc["name"], arguments=tc["arguments"]),
        )
        for tc in tool_calls
    ]
    msg = _Msg(content=content, tool_calls=raw or None, _raw=raw or None)
    resp = SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)
    calls = {}

    class Completions:
        def create(self, **kwargs):
            calls.update(kwargs)
            return resp

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    return client, calls


def test_parses_tool_calls_and_cost():
    client, sent = _fake_client(
        None,
        [{"id": "call_0", "name": "calculate", "arguments": '{"expression":"1+1"}'}],
        SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=0),
    )
    reply = OpenAIModel("gpt-5", client=client).complete([{"role": "user", "content": "hi"}], [])

    assert [(c.name, c.arguments) for c in reply.tool_calls] == [
        ("calculate", '{"expression":"1+1"}')
    ]
    assert reply.items[0]["role"] == "assistant"
    assert reply.cost == 1.25  # 1M input tokens x $1.25/M
    assert sent["model"] == "gpt-5"


def test_plain_answer_has_no_tool_calls():
    client, _ = _fake_client("42", [], SimpleNamespace(prompt_tokens=0, completion_tokens=0))
    reply = OpenAIModel("gpt-5", client=client).complete([], [])
    assert reply.tool_calls == [] and reply.text == "42"


# --- prompt caching and pricing ----------------------------------------------


def test_cached_input_tokens_are_billed_at_a_tenth():
    """Cached input bills at a tenth — the direct payoff for a stable prefix."""
    from types import SimpleNamespace as NS

    from teacup_agent.model import estimate_cost

    full = estimate_cost("gpt-5", input_tokens=1_000_000, output_tokens=0)
    half_cached = estimate_cost("gpt-5", 1_000_000, 0, cached_tokens=500_000)
    assert full == 1.25
    assert half_cached == pytest.approx(0.5 * 1.25 + 0.5 * 0.125)

    client, _ = _fake_client(
        "hi",
        [],
        NS(prompt_tokens=1000, completion_tokens=0, prompt_tokens_details=NS(cached_tokens=800)),
    )
    reply = OpenAIModel("gpt-5", client=client).complete([], [])
    assert reply.cached_tokens == 800
    assert reply.cost == pytest.approx((200 * 1.25 + 800 * 0.125) / 1_000_000)


def test_missing_cache_details_are_treated_as_zero():
    from types import SimpleNamespace as NS

    client, _ = _fake_client("hi", [], NS(prompt_tokens=100, completion_tokens=0))
    assert OpenAIModel("gpt-5", client=client).complete([], []).cached_tokens == 0
