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

**This is the open Agent Skills format** (a folder + `SKILL.md`, YAML frontmatter,
progressive disclosure), not a one-off invented here — the same shape OpenAI Codex CLI,
Microsoft Agent Framework, Cursor and GitHub Copilot all read. A skill written for any
of those loads here unmodified, and one written here loads there. `discover()` validates
`name`/`description` against the spec's own constraints (openly, at
https://agentskills.io/specification) rather than accepting whatever a hand-rolled
parser happened to tolerate, and exposes the spec's optional fields (`license`,
`compatibility`, `metadata`, `allowed-tools`) instead of silently dropping them.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass, field
from typing import Any

import yaml

from teacup_agent import tools as tools_mod

NAME = "load_skill"
DEFAULT_DIR = "skills"

# Agent Skills spec: name is lowercase letters, digits and hyphens, max 64 chars, and
# must equal the folder name — the folder is how discover() finds it, so a name that
# disagreed would make the skill answer to two different identities depending on who's
# asking. description has no charset constraint but is capped at 1024 chars, meant to
# read as one catalog line, not a paragraph.
_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_MAX_NAME_LEN = 64
_MAX_DESCRIPTION_LEN = 1024


@dataclass
class Skill:
    name: str
    description: str
    body: str
    path: pathlib.Path
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    allowed_tools: tuple[str, ...] = ()

    @property
    def folder(self) -> str:
        try:
            return str(self.path.parent.relative_to(pathlib.Path.cwd()))
        except ValueError:
            return str(self.path.parent)


def _frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse the `---`-delimited YAML header of a SKILL.md.

    Real YAML, not a hand-rolled `key: value` scanner — `metadata` in the spec is
    arbitrary key/value data, which a line-based parser cannot represent, and the repo
    already carries `pyyaml` for `agent.yaml` (agent_config.py), so this adds no new
    dependency to read the format properly.
    """
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.S)
    if not match:
        return {}, text
    parsed = yaml.safe_load(match.group(1))
    meta = parsed if isinstance(parsed, dict) else {}
    return meta, match.group(2).strip()


def discover(root: str | pathlib.Path = DEFAULT_DIR) -> list[Skill]:
    """Find every `<root>/*/SKILL.md`. A malformed skill is skipped, not fatal."""
    base = pathlib.Path(root)
    found = []
    for path in sorted(base.glob("*/SKILL.md")):
        try:
            meta, body = _frontmatter(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        folder_name = path.parent.name
        name = str(meta.get("name") or folder_name)
        description = str(meta.get("description", "")).strip()
        if not description or not body:
            continue  # without a description the model cannot know when to load it
        # A name that isn't spec-shaped, or disagrees with the folder discover() found
        # it under, is exactly the kind of thing that silently works today and quietly
        # breaks the moment this same folder is read by a different, spec-strict tool.
        if name != folder_name or not _NAME_RE.match(name) or len(name) > _MAX_NAME_LEN:
            continue
        if len(description) > _MAX_DESCRIPTION_LEN:
            description = description[:_MAX_DESCRIPTION_LEN]
        allowed_tools_raw = meta.get("allowed-tools", "")
        allowed_tools = tuple(str(allowed_tools_raw).split()) if allowed_tools_raw else ()
        metadata = meta.get("metadata") or {}
        found.append(Skill(
            name=name,
            description=description,
            body=body,
            path=path,
            license=meta.get("license"),
            compatibility=meta.get("compatibility"),
            metadata=metadata if isinstance(metadata, dict) else {},
            allowed_tools=allowed_tools,
        ))
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
