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
    """模型一轮输出的归一化表示。"""

    message: dict[str, Any]  # 原样塞回 messages 的 assistant 消息
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    cost: float = 0.0  # 这一轮花掉的美元


class Model(Protocol):
    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Reply: ...


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
        return Reply(
            message=msg.model_dump(exclude_none=True),
            text=msg.content or "",
            tool_calls=calls,
            cost=self._cost(getattr(resp, "usage", None)),
        )

    def _cost(self, usage: Any) -> float:
        if usage is None:
            return 0.0
        pin, pout = PRICES.get(self.model, _DEFAULT_PRICE)
        return (
            getattr(usage, "prompt_tokens", 0) * pin
            + getattr(usage, "completion_tokens", 0) * pout
        ) / 1_000_000


# --------------------------------------------------------------------------
# 脚本模型（离线演示 / 评测）
# --------------------------------------------------------------------------


def assistant_says(text: str, cost: float = 0.001) -> Reply:
    """构造一条「直接回答、不调工具」的模型输出。"""
    return Reply(message={"role": "assistant", "content": text}, text=text, cost=cost)


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
    return Reply(message=message, tool_calls=tool_calls, cost=cost)


class ScriptedModel:
    """按剧本逐条返回，剧本用完后无限返回收尾答案。"""

    def __init__(self, script: list[Reply], fallback: str = "（剧本已结束）"):
        self.script = list(script)
        self.fallback = fallback
        self.calls: list[list[dict[str, Any]]] = []  # 记录每次收到的 messages，便于断言

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Reply:
        self.calls.append(list(messages))
        if self.script:
            return self.script.pop(0)
        return assistant_says(self.fallback)
