"""Skills — procedural knowledge the agent pays for only when it needs it.

The system prompt is static context: every token in it is in every request, whether
the task calls for it or not. That is fine for rules the agent must never forget and
expensive for knowledge it needs once an hour. A skill is the other side of that
boundary: a folder with a `SKILL.md` whose *metadata* is always present and whose
*body* is loaded on demand.

Progressive disclosure, three levels:

1. **Startup**: name and one-line description only, spliced into the system prompt.
   About 25 tokens per skill, so a dozen skills cost less than one tool schema.
2. **On match**: the model calls `load_skill`, and the full procedure arrives as a tool
   result. Tool results are dynamic context, which is exactly where it belongs.
3. **Deep reference**: a skill can point at files beside it, which the model reads with
   `read_file` only if it actually needs them. No new mechanism required.

The result is an agent that carries many specialities and pays for the one it is using.

A skill is knowledge, not code: it is text the model reads. That makes the skills
directory a trust boundary in the same way tool descriptions are, which is why they are
loaded from the project rather than fetched from anywhere.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass
from typing import Any

from mini_agent import tools as tools_mod

NAME = "load_skill"
DEFAULT_DIR = "skills"


@dataclass
class Skill:
    name: str
    description: str
    body: str
    path: pathlib.Path

    @property
    def folder(self) -> str:
        try:
            return str(self.path.parent.relative_to(pathlib.Path.cwd()))
        except ValueError:
            return str(self.path.parent)


def _frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse the `---` header of a SKILL.md.

    Deliberately not a YAML parser: skills use `key: value` lines, and adding a YAML
    dependency to read two fields would cost more than it explains.
    """
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.S)
    if not match:
        return {}, text
    meta = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip("\"'")
    return meta, match.group(2).strip()


def discover(root: str | pathlib.Path = DEFAULT_DIR) -> list[Skill]:
    """Find every `<root>/*/SKILL.md`. A malformed skill is skipped, not fatal."""
    base = pathlib.Path(root)
    found = []
    for path in sorted(base.glob("*/SKILL.md")):
        try:
            meta, body = _frontmatter(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        name = meta.get("name") or path.parent.name
        description = meta.get("description", "").strip()
        if not description or not body:
            continue  # without a description the model cannot know when to load it
        found.append(Skill(name=name, description=description, body=body, path=path))
    return found


def catalog(skills: list[Skill]) -> str:
    """The always-loaded part: what exists and when to reach for it, nothing more."""
    if not skills:
        return ""
    lines = [f"- {s.name}: {s.description}" for s in skills]
    return (
        "Skills available. Each is a procedure written for one kind of task.\n"
        "**If the task in front of you matches one of these descriptions, call "
        "load_skill(name) first and follow what it says.** The one-line description is "
        "all you have until you do; the procedure itself holds the steps, the checks and "
        "the failure modes that make the difference between a good answer and a plausible "
        "one. Loading costs one tool call.\n" + "\n".join(lines)
    )


_loaded: dict[str, Skill] | None = None
_state: Any = None


def enable(skills: list[Skill], state: Any = None) -> None:
    """Register load_skill for one run."""
    global _loaded, _state
    if not skills:
        return
    _loaded = {s.name: s for s in skills}
    _state = state
    names = ", ".join(sorted(_loaded))
    tools_mod.REGISTRY[NAME] = tools_mod.Tool(
        name=NAME,
        description=(
            "Load the full text of a skill when the task matches its description. "
            f"Available: {names}."
        ),
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string", "description": "the skill's name"}},
            "required": ["name"],
        },
        fn=_load_skill,
        # A procedure is an instruction, not raw material: externalizing it would hand
        # the model the first 600 characters of the steps it is supposed to follow.
        externalize=False,
    )


def disable() -> None:
    global _loaded, _state
    _loaded = None
    _state = None
    tools_mod.REGISTRY.pop(NAME, None)


def _load_skill(name: str) -> str:
    if not _loaded:
        return "ERROR: no skills are available in this run"
    skill = _loaded.get(name)
    if skill is None:
        return f"ERROR: no skill named {name!r}. Available: {', '.join(sorted(_loaded))}"

    already = _state is not None and name in getattr(_state, "loaded_skills", [])
    if already:
        # Sending a 600-token procedure twice would undo the saving it exists for.
        return f"The skill {name!r} is already loaded earlier in this conversation; scroll up rather than reloading it."
    if _state is not None:
        _state.loaded_skills.append(name)

    return f"# Skill: {skill.name}\n\n{skill.body}\n\n(Reference files, if any, are in {skill.folder}/ and can be read with read_file.)"
