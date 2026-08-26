"""State —— Agent 在一次任务里携带的全部可变数据。

对应你笔记里的那个 dict，只是换成 dataclass，字段都是真的会被用到的：
step / remaining_budget 会在每轮被更新，并且是循环的终止条件之一。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Status = Literal[
    "idle", "running", "done", "max_steps", "out_of_budget", "out_of_time", "error"
]


@dataclass
class ToolTrace:
    """一次工具调用的留痕，方便事后调试/评测。"""

    step: int
    name: str
    arguments: str
    result: str
    executed: bool = True  # False = 被每轮上限拦下，没有真正执行


@dataclass
class AgentState:
    goal: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    step: int = 0
    max_steps: int = 8
    max_tool_calls_per_step: int = 3  # 每轮最多真正执行几个工具调用，0 = 不限
    remaining_budget: float = 0.05  # 单位：美元，每轮按 token 用量扣减
    time_budget: float | None = None  # 墙上时间上限（秒），None = 不限
    elapsed: float = 0.0  # 已经跑了多久（秒）
    status: Status = "idle"
    answer: str = ""
    salvaged: bool = False  # True = 资源耗尽后靠强制收尾轮抢回来的答案
    trace: list[ToolTrace] = field(default_factory=list)

    # ---- 循环的守卫条件 -------------------------------------------------
    def can_continue(self) -> bool:
        return self.stop_reason() == "running"

    def stop_reason(self) -> Status:
        if self.step >= self.max_steps:
            return "max_steps"
        if self.remaining_budget <= 0:
            return "out_of_budget"
        if self.time_left() is not None and self.time_left() <= 0:
            return "out_of_time"
        return "running"

    def time_left(self) -> float | None:
        """还剩多少秒；没设时间预算则返回 None。"""
        if self.time_budget is None:
            return None
        return round(self.time_budget - self.elapsed, 3)

    def charge(self, cost: float) -> None:
        self.remaining_budget = round(self.remaining_budget - cost, 6)

    def snapshot(self) -> dict[str, Any]:
        """给人看的状态摘要（不含完整 messages）。"""
        return {
            "goal": self.goal,
            "step": self.step,
            "max_steps": self.max_steps,
            "remaining_budget": self.remaining_budget,
            "elapsed_s": round(self.elapsed, 1),
            "status": self.status,
            "salvaged": self.salvaged,
            "messages": len(self.messages),
            "tool_calls": sum(t.executed for t in self.trace),
            "throttled": sum(not t.executed for t in self.trace),
        }
