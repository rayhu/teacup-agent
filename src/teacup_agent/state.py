"""State — everything mutable the agent carries through one task.

Every field is actually used: `step` and `remaining_budget` are updated each turn and are two of
the loop's termination conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Status = Literal[
    "idle", "running", "done", "max_steps", "out_of_budget", "out_of_time", "error"
]


@dataclass
class ToolTrace:
    """One tool call, recorded for debugging and evaluation."""

    step: int
    name: str
    arguments: str
    result: str
    executed: bool = True  # False = never actually ran
    skip_reason: str = ""  # "throttled" | "denied" | "vetoed"


@dataclass
class TodoItem:
    """One action the goal asks for.

    The model can hold a plan in its head, but it cannot be reminded of one it has
    forgotten. Keeping the list in the state means the loop can put unfinished items
    back in front of it every turn — and can refuse to finish while one is open.
    """

    text: str
    done: bool = False
    note: str = ""  # why it was skipped, when it is not done


@dataclass
class AgentState:
    goal: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    step: int = 0
    max_steps: int = 8
    max_tool_calls_per_step: int = 3  # tool calls actually run per turn, 0 = unlimited
    remaining_budget: float = 0.05  # USD, charged per turn from token usage
    time_budget: float | None = None  # wall-clock limit in seconds, None = unlimited
    elapsed: float = 0.0  # seconds spent so far
    context_tokens: int = 0  # size of the context sent on the last turn
    compactions: int = 0  # how many times the context was compacted
    input_tokens_total: int = 0  # cumulative input tokens
    cached_tokens_total: int = 0  # of which served from the prompt cache
    status: Status = "idle"
    answer: str = ""
    salvaged: bool = False  # True = answer rescued by the forced wrap-up turn
    todo: list[TodoItem] = field(default_factory=list)
    completion_checked: bool = False  # the "anything left undone?" push-back fired
    subagent_runs: int = 0  # how many subagents this run delegated to
    loaded_skills: list[str] = field(default_factory=list)  # skills pulled into context
    trace: list[ToolTrace] = field(default_factory=list)
    # profile name -> dollars, when a run routes roles to different models (routing.py).
    # A **diagnostic** for routing decisions, not a second ledger: `remaining_budget` is
    # the ledger, and a subagent charges its parent one rounded delta rather than a
    # per-turn breakdown, so the two can disagree in the last decimal.
    spend_by_profile: dict[str, float] = field(default_factory=dict)

    # ---- the loop's guards ----------------------------------------------
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
        """Seconds remaining, or None when no time budget was set."""
        if self.time_budget is None:
            return None
        return round(self.time_budget - self.elapsed, 3)

    def charge(self, cost: float, profile: str = "") -> None:
        self.remaining_budget = round(self.remaining_budget - cost, 6)
        if profile:  # unnamed charges (a bare model, a subagent's rounded delta) stay
            self.spend_by_profile[profile] = round(  # out of the breakdown entirely
                self.spend_by_profile.get(profile, 0.0) + cost, 6
            )

    def cache_hit_rate(self) -> str:
        if not self.input_tokens_total:
            return "n/a"
        return f"{self.cached_tokens_total / self.input_tokens_total:.0%}"

    def snapshot(self) -> dict[str, Any]:
        """Human-readable summary (without the full message list)."""
        return {
            "goal": self.goal,
            "step": self.step,
            "max_steps": self.max_steps,
            "remaining_budget": self.remaining_budget,
            "elapsed_s": round(self.elapsed, 1),
            "context_tokens": self.context_tokens,
            "compactions": self.compactions,
            "cache_hit": self.cache_hit_rate(),
            "status": self.status,
            "salvaged": self.salvaged,
            "subagents": self.subagent_runs,
            "skills_loaded": len(self.loaded_skills),
            "todo_done": f"{sum(t.done for t in self.todo)}/{len(self.todo)}"
            if self.todo
            else "n/a",
            "messages": len(self.messages),
            "tool_calls": sum(t.executed for t in self.trace),
            "throttled": sum(not t.executed for t in self.trace),
            "spend": dict(self.spend_by_profile),
        }
