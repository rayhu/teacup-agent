"""Agent config — describe an agent in a file instead of a wall of flags.

`cli.py`'s flags are fine for one run. They stop being fine once "the agent" means
"which models it can reach, which MCP servers, which tools and skills, budget and step
ceilings" all at once — that is a *description of an agent*, not a one-off invocation,
and it deserves a file you can diff, copy between projects, and read top to bottom.

`agent.yaml` is that file, and it is a **parallel track**, not a merge: pass `--config
agent.yaml` and it builds everything (model, MCP servers, tools, skills, runtime knobs);
every other behavior flag on `cli.py` is then ignored rather than partially overridden,
because "which of these ten flags silently wins" is worse than "there are two clear
modes." `goal`, `--quiet` and `--resume` stay CLI-only either way: a goal and a resume
path name one specific invocation, not a property of the agent.

Secrets never live in this file. A model profile or an MCP server names an *environment
variable* (`api_key_env`), and any other string value may embed `${VAR}` for the same
reason — resolved from `os.environ` (which `.env` already populates via `load_dotenv()`).
A file with no unresolved `${...}` in it is safe to read over someone's shoulder.

No schema-validation dependency: this is one small typed loader in the same spirit as
`mcp_tools.load_config()`, not a second `pydantic`.
"""

from __future__ import annotations

import os
import pathlib
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from teacup_agent import model as model_mod

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(value: Any) -> Any:
    """Substitute every ${VAR} in every string, recursively.

    Missing a referenced variable is a startup error, not an empty string silently
    spliced in — the same "fail loudly" instinct as everywhere else in this repo.
    """
    if isinstance(value, str):

        def sub(m: re.Match[str]) -> str:
            name = m.group(1)
            if name not in os.environ:
                raise ValueError(
                    f"agent.yaml references ${{{name}}}, but {name} is not set "
                    "(check .env)"
                )
            return os.environ[name]

        return _VAR.sub(sub, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


@dataclass
class ModelProfile:
    provider: str = "openai"  # "openai" | "openai-compatible"
    api: str = "responses"  # "responses" | "chat"
    model: str = "gpt-5"
    api_key_env: str | None = None
    base_url: str | None = None  # any OpenAI-compatible endpoint: vLLM, Ollama, OpenRouter...
    reasoning_effort: str | None = None


@dataclass
class ToolsConfig:
    exclude: list[str] = field(default_factory=list)  # hidden from the model this run
    subagents: bool = False
    subagent_max_steps: int = 4


@dataclass
class RuntimeConfig:
    max_steps: int = 8
    max_tool_calls_per_step: int = 3
    budget: float = 0.05
    deadline: float = 600.0
    tool_timeout: float = 30.0
    context_limit: int = 30_000
    approve: str = "auto"
    plan: str = "auto"
    search: str = "auto"
    memory: str = "memory.json"
    run_dir: str | None = "runs"  # "off" disables persistence + externalization


@dataclass
class AgentConfig:
    models: dict[str, ModelProfile]
    default_model: str
    mcp: dict[str, dict[str, Any]]
    tools: ToolsConfig
    skills_dir: str | None
    runtime: RuntimeConfig
    # Agent2Agent config (peer agents to delegate to, this agent's own Agent Card).
    # Parsed and validated as a plain dict, but not wired into any tool or server yet —
    # that lands in a follow-up. Keeping the field now means the schema does not have
    # to change shape again when it does.
    a2a: dict[str, Any] | None = None


def _model_profile(name: str, spec: dict[str, Any]) -> ModelProfile:
    provider = spec.get("provider", "openai")
    if provider not in ("openai", "openai-compatible"):
        raise ValueError(
            f"models.profiles.{name}.provider is {provider!r}; only 'openai' and "
            "'openai-compatible' are supported today"
        )
    api = spec.get("api", "responses")
    if api not in ("responses", "chat"):
        raise ValueError(
            f"models.profiles.{name}.api must be 'responses' or 'chat', got {api!r}"
        )
    if "model" not in spec:
        raise ValueError(f"models.profiles.{name} is missing 'model'")
    return ModelProfile(
        provider=provider,
        api=api,
        model=spec["model"],
        api_key_env=spec.get("api_key_env"),
        base_url=spec.get("base_url"),
        reasoning_effort=spec.get("reasoning_effort"),
    )


def load(path: str | pathlib.Path) -> AgentConfig:
    text = pathlib.Path(path).read_text(encoding="utf-8")
    raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    raw = _expand_env(raw)

    models_raw = raw.get("models") or {}
    profiles_raw = models_raw.get("profiles") or {}
    if not profiles_raw:
        raise ValueError(f"{path} needs at least one entry under models.profiles")
    profiles = {name: _model_profile(name, spec) for name, spec in profiles_raw.items()}
    default_model = models_raw.get("default") or next(iter(profiles))
    if default_model not in profiles:
        raise ValueError(
            f"models.default {default_model!r} is not one of models.profiles: "
            f"{sorted(profiles)}"
        )

    # Same shape as mcp.json's "servers" map, just nested one level deeper; _-prefixed
    # keys are comments, same convention as mcp_tools.load_config().
    mcp = {
        name: {k: v for k, v in spec.items() if not k.startswith("_")}
        for name, spec in (raw.get("mcp") or {}).items()
    }

    tools_raw = raw.get("tools") or {}
    subagents_raw = tools_raw.get("subagents") or {}
    tools_cfg = ToolsConfig(
        exclude=list(tools_raw.get("exclude") or []),
        subagents=bool(subagents_raw.get("enabled", False)),
        subagent_max_steps=int(subagents_raw.get("max_steps", 4)),
    )

    skills_raw = raw.get("skills") or {}
    skills_dir = _normalize_off_on(skills_raw.get("dir", "skills"))
    if skills_dir == "off":
        skills_dir = None

    runtime_raw = raw.get("runtime") or {}
    runtime = RuntimeConfig(
        max_steps=int(runtime_raw.get("max_steps", 8)),
        max_tool_calls_per_step=int(runtime_raw.get("max_tool_calls_per_step", 3)),
        budget=float(runtime_raw.get("budget", 0.05)),
        deadline=float(runtime_raw.get("deadline", 600.0)),
        tool_timeout=float(runtime_raw.get("tool_timeout", 30.0)),
        context_limit=int(runtime_raw.get("context_limit", 30_000)),
        approve=runtime_raw.get("approve", "auto"),
        plan=_normalize_off_on(runtime_raw.get("plan", "auto")),
        search=runtime_raw.get("search", "auto"),
        memory=runtime_raw.get("memory", "memory.json"),
        run_dir=_normalize_off_on(runtime_raw.get("run_dir", "runs")),
    )

    return AgentConfig(
        models=profiles,
        default_model=default_model,
        mcp=mcp,
        tools=tools_cfg,
        skills_dir=skills_dir,
        runtime=runtime,
        a2a=raw.get("a2a"),
    )


def _normalize_off_on(value: Any) -> Any:
    """PyYAML parses bare on/off/yes/no as booleans (the "Norway problem" — YAML 1.1's
    boolean list). Several fields here use "on"/"off"/"auto" as string sentinels, so
    `plan: off` must mean the same thing whether or not it is quoted, rather than
    silently becoming the Python bool `False` and failing an `== "off"` check later."""
    if value is True:
        return "on"
    if value is False:
        return "off"
    return value


def build_model(profile: ModelProfile) -> "model_mod.Model":
    """Construct the right `Model` for one profile.

    The seam this leans on already exists: `OpenAIModel`/`ResponsesModel` accept a
    pre-built `client`, so reaching a non-default endpoint (a local model, an
    OpenAI-compatible gateway, or just a differently-named API key) needs no change to
    either class — only a client built with `base_url`/`api_key` set here instead of
    left to their own `OPENAI_API_KEY`-from-env default.
    """
    from teacup_agent import model as model_mod

    client = None
    if profile.base_url is not None or profile.api_key_env is not None:
        from openai import OpenAI

        api_key = os.getenv(profile.api_key_env) if profile.api_key_env else None
        if profile.api_key_env and not api_key:
            raise RuntimeError(
                f"model profile references api_key_env {profile.api_key_env!r}, but "
                "that variable is not set (check .env)"
            )
        # A local/self-hosted endpoint may need no key at all; the SDK still requires
        # a non-empty string.
        client = OpenAI(base_url=profile.base_url, api_key=api_key or "not-needed")

    if profile.api == "responses":
        return model_mod.ResponsesModel(
            profile.model, client=client, reasoning_effort=profile.reasoning_effort
        )
    return model_mod.OpenAIModel(profile.model, client=client)


def resolve_run_dir(run_dir: str | None) -> pathlib.Path | None:
    """Same default as cli.py's own: a fresh `runs/<timestamp>` unless told otherwise."""
    if run_dir is None or run_dir == "off":
        return None
    if run_dir == "runs":
        return pathlib.Path("runs") / time.strftime("%Y%m%d-%H%M%S")
    return pathlib.Path(run_dir)
