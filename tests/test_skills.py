"""Skills: metadata is static, the body is dynamic.

The mechanism is worth exactly what that split is worth, so most of these tests are
about the boundary rather than about the file format.
"""

import pytest

from mini_agent import loop, skills, tools
from mini_agent.context import estimate_tokens
from mini_agent.memory import NullMemory
from mini_agent.model import ScriptedModel, assistant_calls, assistant_says

SKILL = """---
name: demo-skill
description: Do the demo thing. Load when demoing.
---

## Procedure

Step one. Step two. """ + "Filler sentence. " * 60


@pytest.fixture
def skill_dir(tmp_path):
    d = tmp_path / "skills" / "demo-skill"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(SKILL, encoding="utf-8")
    (d / "reference.md").write_text("Deep reference material.", encoding="utf-8")
    yield tmp_path / "skills"
    skills.disable()


# --- discovery ----------------------------------------------------------------


def test_frontmatter_gives_name_and_description(skill_dir):
    found = skills.discover(skill_dir)
    assert [s.name for s in found] == ["demo-skill"]
    assert found[0].description.startswith("Do the demo thing")
    assert "Step one" in found[0].body and "---" not in found[0].body


def test_a_skill_without_a_description_is_skipped(tmp_path):
    """Without a description the model cannot know when to load it, so it is noise."""
    d = tmp_path / "skills" / "broken"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("---\nname: broken\n---\n\nBody.", encoding="utf-8")
    assert skills.discover(tmp_path / "skills") == []


def test_a_missing_directory_is_not_fatal(tmp_path):
    assert skills.discover(tmp_path / "nope") == []


# --- the static/dynamic split, which is the whole point ------------------------


def test_each_skill_costs_a_fraction_of_its_body(skill_dir):
    """What matters is the marginal cost of carrying one more skill, not the fixed
    preamble that explains the mechanism once."""
    found = skills.discover(skill_dir)
    per_skill = estimate_tokens(f"- {found[0].name}: {found[0].description}")
    body = estimate_tokens(found[0].body)
    assert per_skill * 5 < body


def test_only_metadata_reaches_the_system_prompt(skill_dir):
    state = loop.run(
        "unrelated task", ScriptedModel([assistant_says("done")]),
        memory=NullMemory(), skills=skill_dir,
    )
    system = state.messages[0]["content"]
    assert "demo-skill: Do the demo thing" in system  # the model knows it exists
    assert "Step one" not in system  # but not what it says
    assert state.loaded_skills == []


def test_loading_puts_the_body_in_dynamic_context(skill_dir):
    state = loop.run(
        "demo task",
        ScriptedModel([
            assistant_calls([("load_skill", {"name": "demo-skill"})]),
            assistant_says("followed it"),
        ]),
        memory=NullMemory(), skills=skill_dir,
    )
    assert state.loaded_skills == ["demo-skill"]
    assert "Step one" in state.trace[0].result  # arrived as a tool result
    assert "Step one" not in state.messages[0]["content"]  # still not in the prefix


def test_loading_twice_does_not_pay_twice(skill_dir):
    """Resending a long procedure would undo the saving the mechanism exists for."""
    state = loop.run(
        "demo task",
        ScriptedModel([
            assistant_calls([("load_skill", {"name": "demo-skill"})]),
            assistant_calls([("load_skill", {"name": "demo-skill"})]),
            assistant_says("done"),
        ]),
        memory=NullMemory(), skills=skill_dir,
    )
    assert "already loaded" in state.trace[1].result
    assert "Step one" not in state.trace[1].result


def test_a_skill_body_is_never_externalized(tmp_path, skill_dir):
    """The externalizer would hand back the first 600 characters of the procedure the
    model is supposed to follow. Instructions stay whole; raw material does not have to."""
    state = loop.run(
        "demo task",
        ScriptedModel([
            assistant_calls([("load_skill", {"name": "demo-skill"})]),
            assistant_says("ok"),
        ]),
        memory=NullMemory(), skills=skill_dir, run_dir=tmp_path / "run",
    )
    assert len(state.trace[0].result) > 1000  # whole, not an excerpt
    assert "read_file for the full content" not in state.trace[0].result


def test_reference_files_are_pointed_at_not_inlined(skill_dir):
    state = loop.run(
        "demo task",
        ScriptedModel([
            assistant_calls([("load_skill", {"name": "demo-skill"})]),
            assistant_says("ok"),
        ]),
        memory=NullMemory(), skills=skill_dir,
    )
    result = state.trace[0].result
    assert "read_file" in result  # level 3 of progressive disclosure
    assert "Deep reference material" not in result


# --- guards -------------------------------------------------------------------


def test_an_unknown_name_lists_what_exists(skill_dir):
    state = loop.run(
        "demo task",
        ScriptedModel([
            assistant_calls([("load_skill", {"name": "nope"})]),
            assistant_says("ok"),
        ]),
        memory=NullMemory(), skills=skill_dir,
    )
    assert state.trace[0].result.startswith("ERROR:") and "demo-skill" in state.trace[0].result


def test_no_tool_and_no_prompt_text_when_there_are_no_skills():
    state = loop.run("task", ScriptedModel([assistant_says("ok")]), memory=NullMemory())
    assert "load_skill" not in tools.REGISTRY
    assert "Skills available" not in state.messages[0]["content"]


def test_the_tool_is_removed_after_the_run(skill_dir):
    loop.run("task", ScriptedModel([assistant_says("ok")]), memory=NullMemory(), skills=skill_dir)
    assert "load_skill" not in tools.REGISTRY


# --- the shipped skills --------------------------------------------------------


def test_the_repos_own_skills_parse():
    found = skills.discover("skills")
    assert {s.name for s in found} == {"web-research", "long-document"}
    for s in found:
        assert len(s.description) < 200  # a catalog line, not a paragraph
        assert len(s.body) > 400         # and a procedure worth loading
