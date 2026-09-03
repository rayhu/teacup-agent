"""Build this agent's own Agent Card, for teacup-agent-serve to publish.

Uses **skills**, not raw tool names, as the card's `skills` list: `skills.py`'s catalog
already carries exactly the shape an `AgentSkill` wants (a name plus a one-line
description meant to tell a reader "when does this apply"), while the tool registry is
the wrong grain — enumerating `search_web`, `calculate`, `read_file` and every MCP tool
a server happens to expose would be noisy and would leak implementation detail an Agent
Card is not meant to carry. An agent with no `skills/` directory gets one generic
fallback skill instead of an empty list, since an Agent Card with no skills at all is a
poor advertisement for what it does.
"""

from __future__ import annotations

from typing import Any

from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill

from teacup_agent.skills import Skill

_GENERIC_SKILL = AgentSkill(
    id="general",
    name="general",
    description="General-purpose task execution: research, arithmetic, and whatever "
    "tools this agent has configured.",
)


def build_agent_card(card_cfg: dict[str, Any], skills: list[Skill], url: str) -> AgentCard:
    agent_skills = [
        AgentSkill(id=s.name, name=s.name, description=s.description) for s in skills
    ] or [_GENERIC_SKILL]
    return AgentCard(
        name=card_cfg.get("name", "teacup-agent"),
        description=card_cfg.get("description", "A minimal AI agent."),
        version=card_cfg.get("version", "0.1.0"),
        supported_interfaces=[AgentInterface(url=url, protocol_binding="JSONRPC")],
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=agent_skills,
    )
