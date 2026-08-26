"""Responses API 后端：用假 client 验证解析，再用假模型验证整条循环。

不联网、不花钱。要点是这个后端和 Chat Completions 的四处形状差异：
扁平的工具定义 / output 列表 / call_id / function_call_output。
"""

from types import SimpleNamespace

from mini_agent import loop
from mini_agent.memory import NullMemory
from mini_agent.model import Reply, ResponsesModel, ToolCall
from mini_agent.evals import tool_results_follow_their_call


class _Item(SimpleNamespace):
    def model_dump(self, exclude_none=False):
        return {k: v for k, v in self.__dict__.items() if not (exclude_none and v is None)}


def _fake_client(output, usage=None, text=""):
    sent = {}

    class Responses:
        def create(self, **kwargs):
            sent.update(kwargs)
            return SimpleNamespace(output=output, usage=usage, output_text=text)

    return SimpleNamespace(responses=Responses()), sent


CHAT_TOOL_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "算数",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def test_tool_specs_are_flattened():
    """Responses 的工具定义没有嵌套的 "function" 层。"""
    client, sent = _fake_client([])
    ResponsesModel("gpt-5", client=client).complete([], CHAT_TOOL_SPEC)
    assert sent["tools"] == [
        {
            "type": "function",
            "name": "calculate",
            "description": "算数",
            "parameters": {"type": "object", "properties": {}},
        }
    ]


def test_reasoning_items_are_carried_back():
    """收益来源：reasoning 项必须原样进入 items，下一轮才能带回去。"""
    output = [
        _Item(type="reasoning", id="rs_1", summary=[]),
        _Item(type="function_call", call_id="fc_1", name="calculate", arguments='{"expression":"1+1"}'),
    ]
    client, _ = _fake_client(output, SimpleNamespace(input_tokens=1_000_000, output_tokens=0))
    reply = ResponsesModel("gpt-5", client=client).complete([], [])

    assert [i["type"] for i in reply.items] == ["reasoning", "function_call"]
    assert [(c.id, c.name) for c in reply.tool_calls] == [("fc_1", "calculate")]  # 用的是 call_id
    assert reply.cost == 1.25


def test_tool_result_shape_differs_from_chat():
    client, _ = _fake_client([])
    item = ResponsesModel("gpt-5", client=client).tool_result_item(
        ToolCall(id="fc_1", name="calculate", arguments="{}"), "2"
    )
    assert item == {"type": "function_call_output", "call_id": "fc_1", "output": "2"}


# --- 整条循环跑 Responses 形状 ------------------------------------------------


class _ScriptedResponses:
    """按剧本返回 Responses 形状的输出，用来验证 loop.py 不用改也能跑。"""

    def __init__(self, script):
        self.script = list(script)

    def complete(self, messages, tools):
        return self.script.pop(0)

    def tool_result_item(self, call, result):
        return {"type": "function_call_output", "call_id": call.id, "output": result}


def test_loop_handles_responses_shape_end_to_end():
    script = [
        Reply(
            items=[
                {"type": "reasoning", "id": "rs_1"},
                {"type": "function_call", "call_id": "fc_1", "name": "calculate", "arguments": '{"expression":"1+1"}'},
                {"type": "function_call", "call_id": "fc_2", "name": "calculate", "arguments": '{"expression":"2+2"}'},
            ],
            tool_calls=[
                ToolCall(id="fc_1", name="calculate", arguments='{"expression":"1+1"}'),
                ToolCall(id="fc_2", name="calculate", arguments='{"expression":"2+2"}'),
            ],
        ),
        Reply(items=[{"type": "message", "role": "assistant", "content": "2 和 4"}], text="2 和 4"),
    ]
    state = loop.run("算 1+1 和 2+2", _ScriptedResponses(script), memory=NullMemory())

    assert state.status == "done" and state.answer == "2 和 4"
    outputs = [m for m in state.messages if m.get("type") == "function_call_output"]
    assert [o["output"] for o in outputs] == ["2", "4"]
    assert tool_results_follow_their_call(state)  # 顺序不变量对两种形状都成立
    assert any(m.get("type") == "reasoning" for m in state.messages)  # 推理项被带回上下文
