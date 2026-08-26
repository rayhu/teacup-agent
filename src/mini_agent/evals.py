"""Evals —— 用脚本模型给控制循环做体检，零 API key、零费用。

评测的不是「模型聪不聪明」，而是「循环对不对」：
坏参数能不能自愈、多工具是否全部回填、上限是否真的刹得住车。

跑法：uv run python -m mini_agent.evals   （或 uv run pytest）
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from mini_agent import loop
from mini_agent.memory import NullMemory
from mini_agent.model import ScriptedModel, assistant_calls, assistant_says
from mini_agent.state import AgentState


class ScriptedWithSummarizer(ScriptedModel):
    """脚本模型 + 会应答压缩请求（压缩会额外调一次模型，剧本里不该为它留位置）。"""

    def __init__(self, script, fallback: str = "（剧本已结束）"):
        super().__init__(script, fallback)
        self.summaries = 0

    def complete(self, messages, tools):
        if messages and str(messages[0].get("content", "")).startswith("你在压缩"):
            self.summaries += 1
            return assistant_says("【摘要】已查证 A、B 两条事实；C 尝试失败。")
        return super().complete(messages, tools)


@dataclass
class Case:
    name: str
    script: list
    check: Callable[[AgentState], bool]
    max_steps: int = 8
    budget: float = 0.05
    max_tool_calls: int = 3
    time_budget: float | None = None
    context_limit: int = 30_000
    clock_values: list[float] | None = None  # 假时钟读数，让时间刹车可复现


def tool_results_follow_their_call(state: AgentState) -> bool:
    """每条工具结果的 id，必须在它**之前**的某条「宣告了该调用」的条目里出现过，
    且每个宣告过的 id 都必须恰好被回填一次。两种 API 形状都认。"""
    announced: set[str] = set()
    for msg in state.messages:
        # Chat Completions 形状
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                announced.add(tc["id"])
        elif msg.get("role") == "tool":
            if msg.get("tool_call_id") not in announced:
                return False
            announced.discard(msg["tool_call_id"])  # 一个 id 只应被回填一次
        # Responses 形状
        elif msg.get("type") == "function_call":
            announced.add(msg["call_id"])
        elif msg.get("type") == "function_call_output":
            if msg.get("call_id") not in announced:
                return False
            announced.discard(msg["call_id"])
    return not announced  # 所有宣告过的调用都必须有结果


CASES: list[Case] = [
    Case(
        name="直接回答：模型不调工具时循环应立即结束",
        script=[assistant_says("42")],
        check=lambda s: s.status == "done" and s.answer == "42" and s.step == 1,
    ),
    Case(
        name="工具回填：一轮内的多个工具调用必须全部有结果消息",
        script=[
            assistant_calls(
                [
                    ("search_web", {"query": "cuda"}),
                    ("calculate", {"expression": "2+3"}),
                ]
            ),
            assistant_says("好了"),
        ],
        check=lambda s: (
            len(s.trace) == 2
            and sum(m["role"] == "tool" for m in s.messages) == 2
            and tool_results_follow_their_call(s)  # 坑 1：顺序不能反
            and "5" in s.trace[1].result
        ),
    ),
    Case(
        name="错误自愈：坏 JSON 参数应作为 ERROR 结果回传，循环继续",
        script=[
            assistant_calls([("calculate", "{坏掉的 json")]),
            assistant_calls([("calculate", {"expression": "1+1"})]),
            assistant_says("修好了，等于 2"),
        ],
        check=lambda s: (
            s.status == "done"
            and s.trace[0].result.startswith("ERROR:")
            and s.trace[1].result == "2"
        ),
    ),
    Case(
        name="未知工具：不应崩溃，而是告诉模型可用工具",
        script=[assistant_calls([("send_email", {"to": "x"})]), assistant_says("换个方式")],
        check=lambda s: "未知工具" in s.trace[0].result and s.status == "done",
    ),
    Case(
        name="步数上限：模型一直调工具时必须刹车",
        script=[assistant_calls([("calculate", {"expression": "1+1"})]) for _ in range(10)],
        check=lambda s: s.status == "max_steps" and s.step == 3,
        max_steps=3,
    ),
    Case(
        name="预算上限：花超了必须停",
        script=[assistant_calls([("calculate", {"expression": "1+1"})], cost=0.02) for _ in range(10)],
        check=lambda s: s.status == "out_of_budget" and s.remaining_budget <= 0,
        budget=0.03,
    ),
    Case(
        name="长期记忆：remember 工具应把事实写进 memory",
        script=[
            assistant_calls([("remember", {"fact": "用户偏好中文回答"})]),
            assistant_says("记住了"),
        ],
        check=lambda s: "已记住" in s.trace[0].result,
    ),
    Case(
        name="每轮上限：超额的调用必须被拒绝执行，但仍要各回一条 tool 消息",
        script=[
            assistant_calls([("calculate", {"expression": f"{i}+1"}) for i in range(5)]),
            assistant_says("够了"),
        ],
        max_tool_calls=2,
        check=lambda s: (
            sum(t.executed for t in s.trace) == 2  # 只真跑了 2 个
            and sum(not t.executed for t in s.trace) == 3  # 3 个被拦
            and sum(m["role"] == "tool" for m in s.messages) == 5  # 但 5 个 id 都回填了（坑 2）
            and tool_results_follow_their_call(s)
            and s.trace[4].result.startswith("ERROR:")
            and s.status == "done"
        ),
    ),
    Case(
        name="绝不空手而归：步数用光后必须强制收尾给出结论",
        # 前两轮一直调工具把步数耗光，收尾轮没有工具可用，只能给结论
        script=[
            assistant_calls([("calculate", {"expression": "1+1"})]),
            assistant_calls([("calculate", {"expression": "1+1"})]),
            assistant_says("目前能给出的结论：1+1=2（高置信）；其余项未能核实。"),
        ],
        max_steps=2,
        check=lambda s: (
            s.status == "max_steps"
            and s.salvaged  # 走了强制收尾轮
            and s.answer
            and not s.answer.startswith("（未得出最终答案")
        ),
    ),
    Case(
        name="预算烧光同样要收尾，而不是丢下一句「停止原因」",
        script=[
            assistant_calls([("calculate", {"expression": "1+1"})], cost=0.02),
            assistant_calls([("calculate", {"expression": "1+1"})], cost=0.02),
            assistant_says("预算耗尽前的结论：1+1=2。"),
        ],
        budget=0.03,
        check=lambda s: s.status == "out_of_budget" and s.salvaged and s.answer,
    ),
    Case(
        name="时间刹车：墙上时间用光同样要停并收尾",
        script=[
            assistant_calls([("calculate", {"expression": "1+1"})]),
            assistant_calls([("calculate", {"expression": "1+1"})]),
            assistant_says("时间到，先给结论：1+1=2。"),
        ],
        time_budget=10.0,
        # 假时钟：跑完两轮后第三次检查已超时（读数依次被 run() 取用）
        clock_values=[0.0, 1.0, 2.0, 12.0, 12.0, 12.0],
        check=lambda s: (
            s.status == "out_of_time"
            and s.step == 2  # 钱和步数都还有富余，是时间把它拦下的
            and s.remaining_budget > 0
            and s.salvaged
            and s.answer
        ),
    ),
    Case(
        name="上下文压缩：压完消息协议必须依然完整",
        script=[assistant_calls([("calculate", {"expression": "1+1"})]) for _ in range(5)]
        + [assistant_says("压缩后仍能收尾")],
        context_limit=200,  # 故意设小，逼它压缩
        max_steps=7,
        check=lambda s: (
            s.compactions >= 1
            and tool_results_follow_their_call(s)  # 没把调用和结果拆散
            and any("[上下文摘要]" in str(m.get("content", "")) for m in s.messages)
            and s.messages[1]["content"] == "上下文压缩：压完消息协议必须依然完整"  # 目标还在
        ),
    ),
    Case(
        name="收尾轮若硬发工具调用，也必须补齐结果消息（否则恢复时会 400）",
        script=[
            assistant_calls([("calculate", {"expression": "1+1"})]),
            assistant_calls([("calculate", {"expression": "9+9"})]),  # 收尾轮还想调工具
        ],
        max_steps=1,
        check=lambda s: (
            s.status == "max_steps"
            and tool_results_follow_their_call(s)  # 没有悬空的 tool_call_id
            and any("强制收尾阶段" in str(m.get("content", "")) for m in s.messages)
        ),
    ),
]


def run_case(case: Case) -> tuple[bool, AgentState]:
    # 评测必须确定性：强制离线检索，不发网络请求
    os.environ.setdefault("MINI_AGENT_SEARCH", "offline")
    state = loop.run(
        goal=case.name,
        model=ScriptedWithSummarizer(list(case.script)),
        memory=NullMemory(),
        max_steps=case.max_steps,
        budget=case.budget,
        max_tool_calls_per_step=case.max_tool_calls,
        time_budget=case.time_budget,
        context_limit=case.context_limit,
        run_dir=None,  # 评测不写盘
        # 每次跑都新建一个迭代器，用例可以重复执行
        **({"clock": iter(case.clock_values).__next__} if case.clock_values else {}),
    )
    return case.check(state), state


def main() -> int:
    failed = 0
    for case in CASES:
        ok, state = run_case(case)
        print(f"{'PASS' if ok else 'FAIL'}  {case.name}")
        if not ok:
            failed += 1
            print(f"      终态: {state.snapshot()}")
            for t in state.trace:
                print(f"      · {t.name}({t.arguments}) -> {t.result[:80]}")
    total = len(CASES)
    print(f"\n{total - failed}/{total} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
