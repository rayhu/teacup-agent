"""Persistence and resume — so a run leaves something behind when it ends.

Before this, context, trace and cost all died with the process: when something
looked wrong the only evidence was the 120 characters the terminal had printed.
(That happened for real — the model cited a few links and we could not tell
whether they came from search results or were invented.)

Once a run is on disk you can do three things:
1. Review it: full messages and trace, any step you like;
2. Resume it: after a crash, a Ctrl-C or a timeout, continue from the last step
   without redoing completed tool calls;
3. Evaluate it: trajectory eval (roadmap #7) takes exactly this file as input.

Saving happens **after every step** rather than at the end on purpose — save only
at the end and the one crash that needed the data is the one that leaves nothing.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

from mini_agent.state import AgentState, TodoItem, ToolTrace

FILENAME = "state.json"


def save(state: AgentState, run_dir: str | pathlib.Path) -> pathlib.Path:
    """Write the whole state to run_dir/state.json.

    Writes a temp file and renames it, so a crash mid-write cannot leave a
    half-written state behind.
    """
    directory = pathlib.Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / FILENAME
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(dataclasses.asdict(state), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


def load(path: str | pathlib.Path) -> AgentState:
    """Read a state back from state.json (or from the directory holding it)."""
    p = pathlib.Path(path)
    if p.is_dir():
        p = p / FILENAME
    data: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    # Nested dataclasses have to be rebuilt by hand — asdict() flattened them to dicts
    # on the way out, and everything downstream expects the objects back.
    trace = [ToolTrace(**t) for t in data.pop("trace", [])]
    todo = [TodoItem(**t) for t in data.pop("todo", [])]
    return AgentState(**data, trace=trace, todo=todo)
