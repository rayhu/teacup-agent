"""Model —— 唯一会「思考」的部件，其余部件都是管道。

这里把模型抽象成一个接口 `complete(messages, tools) -> Reply`，于是可以有两种实现：

* OpenAIModel   真实调用（需要 OPENAI_API_KEY）
* ScriptedModel 按剧本返回，用于离线演示和 evals（零依赖、零费用）

这层抽象不是过度设计：没有它，你就没法在不花钱的情况下测试控制循环。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # 注意是 JSON **字符串**，不是 dict


@dataclass
class Reply:
    """模型一轮输出的归一化表示。

    items 是「要原样追加回上下文的条目」，而不是「一条 assistant 消息」——
    Chat Completions 一轮只产出 1 条；Responses 一轮可能产出多条
    （reasoning 项 + 若干 function_call 项），把它们原样带回去正是保住推理状态的关键。
    """

    items: list[dict[str, Any]]
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    cost: float = 0.0  # 这一轮花掉的美元
    input_tokens: int = 0  # 这一轮送进去的上下文有多大（压缩决策的依据）


class Model(Protocol):
    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Reply: ...

    def tool_result_item(self, call: ToolCall, result: str) -> dict[str, Any]:
        """工具结果要以什么形状塞回上下文 —— 两个 API 这里的形状完全不同。"""
        ...


def chat_tool_result(call: ToolCall, result: str) -> dict[str, Any]:
    """Chat Completions 的工具结果形状。"""
    return {
        "role": "tool",
        "tool_call_id": call.id,
        "name": call.name,
        "content": result,
    }


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = PRICES.get(model, _DEFAULT_PRICE)
    return (input_tokens * pin + output_tokens * pout) / 1_000_000


# --------------------------------------------------------------------------
# 真实模型
# --------------------------------------------------------------------------

# 每百万 token 的价格（美元），仅用于预算演示，可能过时 —— 以官方价目表为准。
PRICES: dict[str, tuple[float, float]] = {
    "gpt-5": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-4.1-mini": (0.40, 1.60),
}
_DEFAULT_PRICE = (1.25, 10.00)


class OpenAIModel:
    """用 Chat Completions 接口。

    为什么不用 Responses API（你笔记里那个）？Chat Completions 的 `messages`
    列表就是 Agent 的状态本身，"把工具结果 append 回去" 这件事一眼可见，
    更适合学习。换成 Responses API 只需要改这个类，循环不用动。
    """

    def __init__(self, model: str = "gpt-5", client: Any = None):
        self.model = model
        if client is None:
            from openai import OpenAI  # 延迟导入：离线运行不需要装 openai

            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError(
                    "缺少 OPENAI_API_KEY。请在 .env 里配置，或改用离线模式（去掉 --live）。"
                )
            client = OpenAI()
        self.client = client

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Reply:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
        )
        msg = resp.choices[0].message
        calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
            for tc in (msg.tool_calls or [])
        ]
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        return Reply(
            items=[msg.model_dump(exclude_none=True)],
            text=msg.content or "",
            tool_calls=calls,
            cost=estimate_cost(
                self.model,
                prompt_tokens,
                getattr(usage, "completion_tokens", 0) if usage else 0,
            ),
            input_tokens=prompt_tokens,
        )

    def tool_result_item(self, call: ToolCall, result: str) -> dict[str, Any]:
        return chat_tool_result(call, result)


class ResponsesModel:
    """Responses API —— 面向推理模型（gpt-5 系列）的推荐路径。

    和 Chat Completions 的四处形状差异，全部封在这个类里，loop.py 一行都不用改：

    1. 工具定义是**扁平**的：{"type": "function", "name": ...}，没有嵌套的 "function" 层；
    2. 输出是 resp.output 列表（可能含 reasoning 项 + 多个 function_call 项），
       而不是单条 message；
    3. 工具调用的 id 字段叫 call_id（不是 id）；
    4. 工具结果回传形状是 {"type": "function_call_output", "call_id", "output"}，
       而不是 role="tool" 的消息。

    为什么值得换：把 output 里的 reasoning 项**原样带回下一轮 input**，
    模型跨工具调用的推理状态就不会丢。Chat Completions 每轮都会把它丢掉。
    OpenAI 的迁移文档称同 prompt 下 SWE-bench 高约 3%。

    状态管理这里用「无状态重发」：每轮把完整上下文重新发一遍（含上一轮的 reasoning 项）。
    另一条路是 previous_response_id + 只发增量，更省钱，等做 #2 prompt caching 时再说。
    """

    def __init__(
        self,
        model: str = "gpt-5",
        client: Any = None,
        reasoning_effort: str | None = None,
    ):
        self.model = model
        self.reasoning_effort = reasoning_effort
        if client is None:
            from openai import OpenAI  # 延迟导入：离线运行不需要装 openai

            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError(
                    "缺少 OPENAI_API_KEY。请在 .env 里配置，或改用离线模式（去掉 --live）。"
                )
            client = OpenAI()
        self.client = client

    @staticmethod
    def _flatten_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """把 Chat 形状的工具定义摊平成 Responses 形状。"""
        flat = []
        for t in tools:
            fn = t.get("function", t)
            flat.append(
                {
                    "type": "function",
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "parameters": fn["parameters"],
                }
            )
        return flat

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Reply:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": messages,
            "tools": self._flatten_tools(tools),
        }
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}

        resp = self.client.responses.create(**kwargs)

        items: list[dict[str, Any]] = []
        calls: list[ToolCall] = []
        for item in resp.output:
            data = item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else dict(item)
            items.append(data)  # reasoning 项也要原样带回去 —— 这就是收益来源
            if data.get("type") == "function_call":
                calls.append(
                    ToolCall(
                        id=data["call_id"],
                        name=data["name"],
                        arguments=data.get("arguments", "{}"),
                    )
                )

        usage = getattr(resp, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        return Reply(
            items=items,
            text=getattr(resp, "output_text", "") or "",
            tool_calls=calls,
            cost=estimate_cost(
                self.model,
                input_tokens,
                getattr(usage, "output_tokens", 0) if usage else 0,
            ),
            input_tokens=input_tokens,
        )

    def tool_result_item(self, call: ToolCall, result: str) -> dict[str, Any]:
        return {"type": "function_call_output", "call_id": call.id, "output": result}


# --------------------------------------------------------------------------
# 脚本模型（离线演示 / 评测）
# --------------------------------------------------------------------------


def assistant_says(text: str, cost: float = 0.001) -> Reply:
    """构造一条「直接回答、不调工具」的模型输出。"""
    return Reply(items=[{"role": "assistant", "content": text}], text=text, cost=cost)


def assistant_calls(calls: list[tuple[str, Any]], cost: float = 0.001) -> Reply:
    """构造一条「请求调用工具」的模型输出。

    calls: [(工具名, 参数)]，参数可以是 dict，也可以直接给字符串（用来模拟坏 JSON）。
    """
    tool_calls = []
    for i, (name, args) in enumerate(calls):
        raw = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
        tool_calls.append(ToolCall(id=f"call_{i}", name=name, arguments=raw))
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in tool_calls
        ],
    }
    return Reply(items=[message], tool_calls=tool_calls, cost=cost)


class ScriptedModel:
    """按剧本逐条返回，剧本用完后无限返回收尾答案。"""

    def __init__(self, script: list[Reply], fallback: str = "（剧本已结束）"):
        self.script = list(script)
        self.fallback = fallback
        self.calls: list[list[dict[str, Any]]] = []  # 记录每次收到的 messages，便于断言
        self.tool_specs: list[list[dict[str, Any]]] = []  # 每次收到的工具清单

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Reply:
        self.calls.append(list(messages))
        self.tool_specs.append(list(tools))
        if self.script:
            return self.script.pop(0)
        return assistant_says(self.fallback)

    def tool_result_item(self, call: ToolCall, result: str) -> dict[str, Any]:
        return chat_tool_result(call, result)
