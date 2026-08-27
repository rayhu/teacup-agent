"""Planning — turn the goal into a checklist the loop can hold the model to.

Why this exists: a real run was asked to "research X, then email me the result". The
agent researched well and never sent the email — not because it ran out of resources
(it stopped at turn 6 of 14 with 97% of the budget unspent) but because **nothing was
keeping track of the second half of the request**. It finished, declared done, and the
status field agreed.

That is the same failure as "the model does not know its own situation", one level up:
it did not know it had missed something, because nobody was remembering.

So: decompose the goal once at the start, show what is still open every turn, and
refuse to finish while an item is untouched (see the completion check in loop.py).
"""

from __future__ import annotations

import json
import re
from typing import Any

from teacup_agent.state import TodoItem

PLANNER = """Break the user's request into the concrete actions it asks for.

Rules:
- One item per action the user actually asked for. Research, analysis, and any
  side-effecting step (sending an email, writing a file) are separate items.
- 1 to 5 items. Do not invent work the user did not ask for.
- Each item is a short imperative phrase, under 12 words.

Output a JSON array of strings and nothing else, e.g.
["research X's investment record", "email the findings to a@b.com"]"""


def decompose(goal: str, model: Any) -> list[TodoItem]:
    """One model call, no tools, to split the goal into action items.

    Returns an empty list if anything goes wrong — a broken planner must never stop
    the run, it just means the loop works the way it did before this feature.
    """
    try:
        reply = model.complete(
            [
                {"role": "system", "content": PLANNER},
                {"role": "user", "content": goal},
            ],
            [],
        )
    except Exception:
        return []

    match = re.search(r"\[.*\]", reply.text or "", re.S)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [TodoItem(text=str(i).strip()) for i in items if str(i).strip()][:5]


def render(todo: list[TodoItem]) -> str:
    """The checklist as the model sees it in the per-turn status line."""
    if not todo:
        return ""
    parts = []
    for i, item in enumerate(todo, 1):
        mark = "x" if item.done else " "
        parts.append(f"[{mark}] {i}. {item.text}")
    return "Checklist: " + "; ".join(parts)


def pending(todo: list[TodoItem]) -> list[TodoItem]:
    return [t for t in todo if not t.done]
