"""Reflect — after a run ends, write down what worked or what went wrong, so a
future run starts with the benefit of this one's experience.

Shaped exactly like plan.py: one extra model call, no tools, and any failure means
"write nothing" rather than sinking an already-finished run.

Two triggers, computed for free from trajectory.mechanical() (no model call unless
one fires):
- experience (success): the run finished cleanly — done, not salvaged, nothing left
  open, nothing duplicated, the action asked for was actually attempted.
- lesson (recovered error): a tool call failed and the run still delivered a real
  answer afterward — proof the failure was actually worked around, not merely present.

Storage is a second, deliberately lower-trust tier on Memory (see memory.py's
`notes`), not folded into the model's own remember() facts: these are the harness's
own after-the-fact self-assessment, unreviewed by construction. The intended path for
a good one is a human promoting it into docs/roadmap.md's "Field patches" — this is
the candidate feed for that, not a replacement.
"""

from __future__ import annotations

import json
import re
from typing import Any

from teacup_agent import trajectory
from teacup_agent.memory import Memory
from teacup_agent.state import AgentState

REFLECT = """Below is the full trajectory of one completed agent run. Write down what a
future run of a similar task should learn from it.

Rules:
- Generalize beyond this specific question — a rule that helps on a different task of
  the same kind, not a restatement of this one's answer.
- Ground every claim in what the trajectory actually shows. If you are not sure why
  something worked, say so rather than inventing a mechanism.
- One sentence per field.

{fields}

Output JSON only, with exactly the keys requested and no others."""

_EXPERIENCE_FIELD = '"experience": what approach or tool sequence worked well here, generalized.'
_LESSON_FIELD = (
    '"lesson": name the tool/step that failed, why, and what fixed it, generalized.'
)


def should_reflect(state: AgentState, metrics: dict[str, Any]) -> tuple[bool, bool]:
    """Returns (write_experience, write_lesson). Deliberately strict: a messy run that
    happened to finish should not be written up as a model to imitate."""
    experience = (
        state.status == "done"
        and not state.salvaged
        and metrics["pending_todos"] == 0
        and metrics["duplicate_tool_calls"] == 0
        and not metrics["action_never_attempted"]
    )
    lesson = metrics["failed_tool_calls"] > 0 and metrics["delivered"]
    return experience, lesson


def maybe_record(state: AgentState, model: Any, memory: Memory) -> list[str]:
    """Write an experience and/or lesson note if this run's trajectory earns one.

    Returns the kinds actually written, so the caller can emit an event; empty if
    nothing qualified or the reflection call itself failed.
    """
    metrics = trajectory.mechanical(state)
    want_experience, want_lesson = should_reflect(state, metrics)
    if not want_experience and not want_lesson:
        return []

    fields = [f for f, want in ((_EXPERIENCE_FIELD, want_experience), (_LESSON_FIELD, want_lesson)) if want]

    try:
        reply = model.complete(
            [
                {"role": "system", "content": REFLECT.format(fields="\n".join(fields))},
                {"role": "user", "content": trajectory.render_trajectory(state)},
            ],
            [],
        )
    except Exception:
        return []
    state.charge(reply.cost)  # honest accounting, even though status is already final

    match = re.search(r"\{.*\}", reply.text or "", re.S)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []

    written = []
    if want_experience and (text := str(data.get("experience", "")).strip()):
        memory.note("experience", text)
        written.append("experience")
    if want_lesson and (text := str(data.get("lesson", "")).strip()):
        memory.note("lesson", text)
        written.append("lesson")
    return written
