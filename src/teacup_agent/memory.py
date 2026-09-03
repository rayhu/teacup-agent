"""Memory — two layers, both deliberately tiny.

Short term: `AgentState.messages` itself, gone when the task ends.
Long term:  a JSON file that survives across sessions. It is injected into the
            system prompt at startup, and the model can write to it with the
            `remember` tool.

A real project swaps this layer for a vector store or a database; keep the
remember() + recall() interface and nothing else has to change.

`notes` is a second, deliberately lower-trust tier: `reflect.py` writes an
after-the-fact self-assessment here (an "experience" or a "lesson"), unreviewed by
construction, never mixed into `facts`. `recall()` labels the two differently so a
model — or a human reading a transcript — can weigh a fact the model chose to
remember mid-task differently from a note the harness generated about a run after
it already ended.
"""

from __future__ import annotations

import json
import pathlib


class Memory:
    def __init__(
        self,
        path: str | pathlib.Path = "memory.json",
        limit: int = 20,
        note_limit: int = 10,
    ):
        self.path = pathlib.Path(path)
        self.limit = limit
        self.note_limit = note_limit
        self.facts: list[str] = []
        self.notes: list[dict[str, str]] = []  # [{"kind": "experience"|"lesson", "text": ...}]
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.facts = [str(x) for x in data.get("facts", [])]
                self.notes = [
                    {"kind": str(n.get("kind", "")), "text": str(n.get("text", ""))}
                    for n in data.get("notes", [])
                    if str(n.get("text", "")).strip()
                ]
            except (json.JSONDecodeError, OSError):
                self.facts, self.notes = [], []  # corrupt memory must not take the agent down

    def save(self) -> None:
        self.path.write_text(
            json.dumps(
                {"facts": self.facts, "notes": self.notes}, ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )

    def remember(self, fact: str) -> None:
        fact = fact.strip()
        if fact and fact not in self.facts:
            self.facts.append(fact)
            self.facts = self.facts[-self.limit :]  # simplest eviction: keep last N
            self.save()

    def note(self, kind: str, text: str) -> None:
        """Record an unreviewed, harness-generated note (see `reflect.py`) — never
        called by the model itself, unlike `remember`."""
        text = text.strip()
        if not text:
            return
        entry = {"kind": kind, "text": text}
        if entry in self.notes:
            return
        self.notes.append(entry)
        self.notes = self.notes[-self.note_limit :]  # same eviction as facts
        self.save()

    def recall(self) -> str:
        """The memory block to splice into the system prompt."""
        blocks = []
        if self.facts:
            lines = "\n".join(f"- {f}" for f in self.facts)
            blocks.append(f"You remember these facts from earlier sessions:\n{lines}")
        if self.notes:
            lines = "\n".join(f"- [{n['kind']}] {n['text']}" for n in self.notes)
            blocks.append(
                "Unreviewed notes a past run wrote about itself (auto-generated, "
                f"not human-verified — weigh accordingly):\n{lines}"
            )
        return "\n\n".join(blocks)


class NullMemory(Memory):
    """Memory that never touches disk — for evals and unit tests."""

    def __init__(self) -> None:
        self.path = pathlib.Path("/dev/null")
        self.limit = 20
        self.note_limit = 10
        self.facts = []
        self.notes = []

    def save(self) -> None:  # noqa: D102
        pass

    def load(self) -> None:  # noqa: D102
        pass
