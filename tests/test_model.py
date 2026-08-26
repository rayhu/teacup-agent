"""用假 client 验证 OpenAIModel 的解析逻辑（不联网、不花钱）。

它检查的是三件容易写错的事：tool_calls 的取法、assistant 消息的原样回填、费用换算。
"""

from types import SimpleNamespace

from mini_agent.model import OpenAIModel


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
    assert reply.cost == 1.25  # 1M 输入 token × $1.25/M
    assert sent["model"] == "gpt-5"


def test_plain_answer_has_no_tool_calls():
    client, _ = _fake_client("42", [], SimpleNamespace(prompt_tokens=0, completion_tokens=0))
    reply = OpenAIModel("gpt-5", client=client).complete([], [])
    assert reply.tool_calls == [] and reply.text == "42"
