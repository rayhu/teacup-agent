"""Memory — two layers, both deliberately tiny.

Short term: `AgentState.messages` itself, gone when the task ends.
Long term:  a JSON file that survives across sessions. It is injected into the
            system prompt at startup, and the model can write to it with the
            `remember` tool.

A real project swaps this layer for a vector store or a database; keep the
remember() + recall() interface and nothing else has to change.
"""

from __future__ import annotations

import json
import pathlib


class Memory:
    def __init__(self, path: str | pathlib.Path = "memory.json", limit: int = 20):
        self.path = pathlib.Path(path)
        self.limit = limit
        self.facts: list[str] = []
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.facts = [str(x) for x in data.get("facts", [])]
            except (json.JSONDecodeError, OSError):
                self.facts = []  # corrupt memory must not take the agent down

    def save(self) -> None:
        self.path.write_text(
            json.dumps({"facts": self.facts}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def remember(self, fact: str) -> None:
        fact = fact.strip()
        if fact and fact not in self.facts:
            self.facts.append(fact)
            self.facts = self.facts[-self.limit :]  # simplest eviction: keep last N
            self.save()

    def recall(self) -> str:
        """The memory block to splice into the system prompt."""
        if not self.facts:
            return ""
        lines = "\n".join(f"- {f}" for f in self.facts)
        return f"You remember these facts from earlier sessions:\n{lines}"


class NullMemory(Memory):
    """Memory that never touches disk — for evals and unit tests."""

    def __init__(self) -> None:
        self.path = pathlib.Path("/dev/null")
        self.limit = 20
        self.facts = []

    def save(self) -> None:  # noqa: D102
        pass

    def load(self) -> None:  # noqa: D102
        pass
