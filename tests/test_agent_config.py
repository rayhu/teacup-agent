"""agent_config.py: parsing, env expansion, and the --config wiring in cli.py."""

from __future__ import annotations

import pathlib
import types

import pytest

from teacup_agent import agent_config, model as model_mod

MINIMAL = """
models:
  default: main
  profiles:
    main:
      provider: openai
      api: responses
      model: gpt-5
      api_key_env: FAKE_KEY
runtime:
  plan: off
  run_dir: off
"""


def _write(tmp_path, text):
    path = tmp_path / "agent.yaml"
    path.write_text(text, encoding="utf-8")
    return path


# --- loading and validation ----------------------------------------------------


def test_loads_the_minimal_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "sk-test")
    cfg = agent_config.load(_write(tmp_path, MINIMAL))
    assert cfg.default_model == "main"
    assert cfg.models["main"].model == "gpt-5"
    assert cfg.runtime.plan == "off"
    assert cfg.runtime.reflect == "auto"  # not set in MINIMAL, so the default
    assert cfg.runtime.run_dir == "off"
    assert cfg.mcp == {}
    assert cfg.tools.exclude == []
    assert cfg.skills_dir == "skills"  # the built-in default, same as --skills


def test_rejects_an_unknown_default_model(tmp_path):
    bad = MINIMAL.replace("default: main", "default: nope")
    with pytest.raises(ValueError, match="models.default"):
        agent_config.load(_write(tmp_path, bad))


def test_rejects_a_profile_with_no_model_field(tmp_path):
    bad = MINIMAL.replace("      model: gpt-5\n", "")
    with pytest.raises(ValueError, match="missing 'model'"):
        agent_config.load(_write(tmp_path, bad))


def test_rejects_an_unsupported_provider(tmp_path):
    bad = MINIMAL.replace("provider: openai", "provider: azure")
    with pytest.raises(ValueError, match="provider"):
        agent_config.load(_write(tmp_path, bad))


def test_requires_at_least_one_model_profile(tmp_path):
    with pytest.raises(ValueError, match="models.profiles"):
        agent_config.load(_write(tmp_path, "runtime:\n  plan: off\n"))


# --- ${VAR} expansion ------------------------------------------------------------


def _with_base_url(var_expr: str) -> str:
    return MINIMAL.replace(
        "api_key_env: FAKE_KEY", f"api_key_env: FAKE_KEY\n      base_url: {var_expr}"
    )


def test_expands_dollar_brace_vars_anywhere_in_the_document(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "sk-test")
    monkeypatch.setenv("MY_URL", "http://localhost:11434/v1")
    cfg = agent_config.load(_write(tmp_path, _with_base_url("${MY_URL}")))
    assert cfg.models["main"].base_url == "http://localhost:11434/v1"


def test_a_missing_var_fails_loudly_instead_of_substituting_blank(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "sk-test")
    monkeypatch.delenv("NOPE", raising=False)
    with pytest.raises(ValueError, match=r"\$\{NOPE\}"):
        agent_config.load(_write(tmp_path, _with_base_url("${NOPE}")))


# --- the YAML "Norway problem" (bare off/on parse as booleans) ---------------------


def test_bare_off_is_not_silently_swallowed_as_a_boolean(tmp_path, monkeypatch):
    """plan: off (unquoted) parses through PyYAML as the bool False, not the string
    "off" — must still behave like "off", not silently fall through to the default."""
    monkeypatch.setenv("FAKE_KEY", "sk-test")
    text = MINIMAL.replace("plan: off", "plan: off\n  reflect: off") + "\nskills:\n  dir: off\n"
    cfg = agent_config.load(_write(tmp_path, text))
    assert cfg.runtime.plan == "off"
    assert cfg.runtime.reflect == "off"
    assert cfg.runtime.run_dir == "off"
    assert cfg.skills_dir is None


# --- the tools/skills/mcp blocks --------------------------------------------------


def test_tools_and_skills_and_mcp_blocks(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "sk-test")
    text = MINIMAL + """
mcp:
  fetch:
    command: uvx
    args: ["mcp-server-fetch"]
    _comment: dropped like mcp.json's
tools:
  exclude: ["send_email"]
  subagents:
    enabled: true
    max_steps: 6
skills:
  dir: off
"""
    cfg = agent_config.load(_write(tmp_path, text))
    assert cfg.mcp == {"fetch": {"command": "uvx", "args": ["mcp-server-fetch"]}}
    assert cfg.tools.exclude == ["send_email"]
    assert cfg.tools.subagents is True
    assert cfg.tools.subagent_max_steps == 6
    assert cfg.skills_dir is None


def test_a2a_card_is_parsed_as_a_plain_dict(tmp_path, monkeypatch):
    """The card is #18's concern (building an AgentCard); this module just carries it.
    Peers are #17's concern — both landed, so both get real shape here."""
    monkeypatch.setenv("FAKE_KEY", "sk-test")
    text = MINIMAL + "\na2a:\n  card:\n    name: my-agent\n"
    cfg = agent_config.load(_write(tmp_path, text))
    assert cfg.a2a.card == {"name": "my-agent"}
    assert cfg.a2a.peers == {}


def test_a2a_peers_get_real_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "sk-test")
    text = MINIMAL + (
        "\na2a:\n  peers:\n    finance:\n      url: https://agents.example.com/finance\n"
        "      api_key_env: FINANCE_TOKEN\n"
    )
    cfg = agent_config.load(_write(tmp_path, text))
    peer = cfg.a2a.peers["finance"]
    assert peer.url == "https://agents.example.com/finance"
    assert peer.api_key_env == "FINANCE_TOKEN"


def test_a2a_peer_missing_url_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "sk-test")
    text = MINIMAL + "\na2a:\n  peers:\n    finance:\n      api_key_env: FINANCE_TOKEN\n"
    with pytest.raises(ValueError, match="a2a.peers.finance"):
        agent_config.load(_write(tmp_path, text))


def test_a2a_defaults_to_no_peers_and_no_card(tmp_path, monkeypatch):
    """MINIMAL has no a2a: block at all — the default must not crash anything reading
    cfg.a2a.peers/cfg.a2a.card downstream."""
    monkeypatch.setenv("FAKE_KEY", "sk-test")
    cfg = agent_config.load(_write(tmp_path, MINIMAL))
    assert cfg.a2a.peers == {}
    assert cfg.a2a.card == {}


def test_the_shipped_example_config_parses(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = agent_config.load("agent.example.yaml")
    assert cfg.default_model == "gpt5-responses"
    assert cfg.models["gpt5-responses"].api == "responses"
    assert cfg.a2a.card == {
        "name": "my-agent",
        "description": "What this agent does.",
        "version": "0.1.0",
    }
    assert cfg.a2a.peers == {}  # peers stay commented out in the template


# --- build_model ------------------------------------------------------------------


def test_default_profile_builds_without_a_custom_client(monkeypatch):
    """No base_url and no api_key_env: behave exactly like today's --live path, which
    reads OPENAI_API_KEY itself and needs no client injected here."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    m = agent_config.build_model(agent_config.ModelProfile(api="responses", model="gpt-5"))
    assert isinstance(m, model_mod.ResponsesModel)
    assert m.client.api_key == "sk-test"


def test_base_url_reaches_an_openai_compatible_endpoint(monkeypatch):
    monkeypatch.setenv("LOCAL_KEY", "unused")
    profile = agent_config.ModelProfile(
        provider="openai-compatible",
        api="chat",
        model="llama-3.1-70b",
        base_url="http://localhost:11434/v1",
        api_key_env="LOCAL_KEY",
    )
    m = agent_config.build_model(profile)
    assert isinstance(m, model_mod.OpenAIModel)
    assert str(m.client.base_url).startswith("http://localhost:11434/v1")


def test_missing_api_key_env_fails_at_build_time(monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    profile = agent_config.ModelProfile(api_key_env="MISSING_KEY")
    with pytest.raises(RuntimeError, match="MISSING_KEY"):
        agent_config.build_model(profile)


# --- resolve_run_dir ---------------------------------------------------------------


def test_resolve_run_dir_off_and_none_disable_persistence():
    assert agent_config.resolve_run_dir("off") is None
    assert agent_config.resolve_run_dir(None) is None


def test_resolve_run_dir_default_is_a_fresh_timestamped_dir():
    path = agent_config.resolve_run_dir("runs")
    assert path.parent == pathlib.Path("runs")


def test_resolve_run_dir_explicit_path_is_used_verbatim():
    assert agent_config.resolve_run_dir("scratch/here") == pathlib.Path("scratch/here")


# --- end-to-end through cli._main_config, offline ----------------------------------


def test_main_config_runs_a_full_loop_offline(tmp_path, monkeypatch):
    """The --config path wires model + runtime knobs into loop.run() correctly, with
    no network: build_model is swapped for a ScriptedModel, same trick real API calls
    would otherwise require an API key for."""
    from teacup_agent import cli

    monkeypatch.setenv("FAKE_KEY", "sk-test")
    cfg_path = _write(tmp_path, MINIMAL)

    scripted = model_mod.ScriptedModel(script=[model_mod.assistant_says("42")])
    monkeypatch.setattr(agent_config, "build_model", lambda profile: scripted)
    monkeypatch.chdir(tmp_path)  # memory.json lands in tmp_path, not the repo

    args = types.SimpleNamespace(
        goal="what is the answer", config=str(cfg_path), quiet=True, resume=None
    )
    assert cli._main_config(args) == 0


def test_main_config_wires_reflect_through_to_the_loop(tmp_path, monkeypatch):
    """runtime.reflect was added after agent_config.py and reflect.py were built
    independently — this pins that --config actually reaches loop.run(reflect=...)
    rather than silently defaulting to off."""
    from teacup_agent import cli

    monkeypatch.setenv("FAKE_KEY", "sk-test")
    text = MINIMAL.replace("plan: off", "plan: off\n  reflect: on")
    cfg_path = _write(tmp_path, text)

    scripted = model_mod.ScriptedModel(
        script=[model_mod.assistant_says("42")],
        reflection={"experience": "wiring reached the loop"},
    )
    monkeypatch.setattr(agent_config, "build_model", lambda profile: scripted)
    monkeypatch.chdir(tmp_path)

    args = types.SimpleNamespace(
        goal="what is the answer", config=str(cfg_path), quiet=True, resume=None
    )
    assert cli._main_config(args) == 0
    memory_file = tmp_path / "memory.json"
    assert "wiring reached the loop" in memory_file.read_text(encoding="utf-8")


def test_main_config_wires_delegate_a2a_through_the_cli(tmp_path, monkeypatch):
    """The --config path's own A2AHub construction (cli.py's _main_config, not the
    A2AHub unit tests in test_a2a_client.py) actually reaches loop.run(): a scripted
    model calls delegate_a2a against a real in-process a2a-sdk server."""
    import pathlib as _pathlib
    import sys as _sys

    import httpx

    from teacup_agent import cli
    from teacup_agent.a2a.client import A2AHub

    _sys.path.insert(0, str(_pathlib.Path(__file__).parent / "fixtures"))
    from demo_a2a_server import build_app  # noqa: E402

    monkeypatch.setenv("FAKE_KEY", "sk-test")
    # delegate_a2a requires approval by design (AGENTS.md rule 4); "allow" here is the
    # test explicitly exercising the executed path, not a change to that default.
    text = MINIMAL.replace("plan: off", "plan: off\n  approve: allow")
    text += "\na2a:\n  peers:\n    demo:\n      url: http://test\n"
    cfg_path = _write(tmp_path, text)

    test_transport = httpx.ASGITransport(app=build_app())
    real_init = A2AHub.__init__
    monkeypatch.setattr(A2AHub, "__init__", lambda self: real_init(self, transport=test_transport))

    scripted = model_mod.ScriptedModel(
        script=[
            model_mod.assistant_calls([("delegate_a2a", {"peer": "demo", "task": "ping"})]),
            model_mod.assistant_says("done"),
        ]
    )
    monkeypatch.setattr(agent_config, "build_model", lambda profile: scripted)
    monkeypatch.chdir(tmp_path)

    args = types.SimpleNamespace(
        goal="delegate something", config=str(cfg_path), quiet=True, resume=None
    )
    assert cli._main_config(args) == 0
    tool_result = next(
        m for m in scripted.calls[-1] if m.get("name") == "delegate_a2a"
    )
    assert tool_result["content"] == "echo: ping"
