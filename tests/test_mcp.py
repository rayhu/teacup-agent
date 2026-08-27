"""MCP integration, tested against a real MCP server over stdio.

The server is tests/fixtures/demo_mcp_server.py — real protocol, no network and no
npx, so this suite stays as hermetic as the rest.
"""

import json
import sys

import pytest

from teacup_agent import tools
from teacup_agent.mcp_tools import McpHub, _result_text, _tool_name, load_config

SERVER = {"command": sys.executable, "args": ["tests/fixtures/demo_mcp_server.py"]}


@pytest.fixture(scope="module")
def hub():
    h = McpHub()
    h.connect("demo", SERVER)
    yield h
    h.close()


# --- naming -------------------------------------------------------------------


def test_tool_names_are_namespaced_and_sanitized():
    """Two servers can each expose `search`, and OpenAI function names allow only
    [A-Za-z0-9_-] — MCP names may contain dots."""
    assert _tool_name("fetch", "fetch") == "fetch__fetch"
    assert _tool_name("gh", "admin.tools.list") == "gh__admin_tools_list"


# --- registration and the approval mapping ------------------------------------


def test_server_tools_land_in_the_registry(hub):
    assert {"demo__echo", "demo__wipe", "demo__unannotated"} <= set(tools.REGISTRY)
    spec = tools.REGISTRY["demo__echo"]
    assert spec.description.startswith("[demo]")  # the source server is visible
    assert spec.parameters["type"] == "object"  # the server's own JSON Schema


def test_read_only_hint_opens_a_tool(hub):
    assert tools.REGISTRY["demo__echo"].requires_approval is False


def test_anything_not_marked_read_only_is_gated(hub):
    assert tools.REGISTRY["demo__wipe"].requires_approval is True
    # The common case in the wild: a server that annotates nothing. Gating it is the
    # right default for a gate that exists because nobody is watching.
    assert tools.REGISTRY["demo__unannotated"].requires_approval is True


def test_approve_none_is_an_explicit_statement_of_trust():
    """The spec says annotations are untrusted unless the server is. Trust is
    something the config states, not something we infer."""
    h = McpHub()
    try:
        h.connect("trusted", {**SERVER, "approve": "none"})
        assert tools.REGISTRY["trusted__wipe"].requires_approval is False
    finally:
        h.close()


def test_approve_all_gates_even_read_only_tools():
    h = McpHub()
    try:
        h.connect("paranoid", {**SERVER, "approve": "all"})
        assert tools.REGISTRY["paranoid__echo"].requires_approval is True
    finally:
        h.close()


def test_allowlist_keeps_the_context_prefix_small():
    """Every tool schema costs prefix tokens on every request, so a server with 20
    tools should not force all 20 into the context."""
    h = McpHub()
    try:
        added = h.connect("small", {**SERVER, "tools": ["echo"]})
        assert added == ["small__echo"]
        assert "small__wipe" not in tools.REGISTRY
    finally:
        h.close()


# --- calling ------------------------------------------------------------------


def test_a_call_goes_through_the_normal_execute_path(hub):
    assert tools.execute("demo__echo", json.dumps({"text": "hello"})) == "echo: hello"


def test_tool_execution_errors_become_our_ERROR_string(hub):
    """MCP separates protocol errors from tool execution errors and says clients
    SHOULD hand the latter to the model. That is what execute() already does."""
    assert tools.execute("demo__explode", "{}").startswith("ERROR:")


def test_bad_arguments_come_back_as_a_correctable_error(hub):
    out = tools.execute("demo__echo", json.dumps({"wrong": 1}))
    assert out.startswith("ERROR:") and "echo" in out


def test_close_removes_the_tools_again():
    h = McpHub()
    h.connect("temp", SERVER)
    assert "temp__echo" in tools.REGISTRY
    h.close()
    assert not [n for n in tools.REGISTRY if n.startswith("temp__")]


# --- result conversion (unit, no server needed) --------------------------------


class _Block:
    def __init__(self, text=None, uri=None, type="text"):
        self.text, self.uri, self.type = text, uri, type


class _Result:
    def __init__(self, content=None, structured_content=None, is_error=False,
                 result_type="complete"):
        self.content = content
        self.structured_content = structured_content
        self.is_error = is_error
        self.result_type = result_type


def test_text_blocks_are_joined():
    assert _result_text(_Result([_Block("a"), _Block("b")])) == "a\nb"


def test_structured_content_is_used_when_there_is_no_text():
    assert _result_text(_Result(structured_content={"k": 1})) == '{"k": 1}'


def test_is_error_becomes_an_ERROR_prefix():
    assert _result_text(_Result([_Block("nope")], is_error=True)) == "ERROR: nope"


def test_input_required_is_reported_rather_than_hung_on():
    """Multi round-trip requests want interactive input mid-call. We do not do
    elicitation, so it becomes an error the model can route around."""
    out = _result_text(_Result(result_type="input_required"))
    assert out.startswith("ERROR:") and "interactive input" in out


def test_empty_results_say_so_instead_of_returning_nothing():
    assert "no content" in _result_text(_Result([]))


# --- config -------------------------------------------------------------------


def test_config_accepts_the_usual_shape(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"servers": {"fetch": {"command": "uvx"}}}), encoding="utf-8")
    assert load_config(str(path)) == {"fetch": {"command": "uvx"}}


def test_config_also_accepts_a_bare_mapping(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"fetch": {"command": "uvx"}}), encoding="utf-8")
    assert load_config(str(path)) == {"fetch": {"command": "uvx"}}


def test_config_drops_comment_keys(tmp_path):
    """JSON has no comments, and every config file grows them anyway."""
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps({"servers": {"fetch": {"command": "uvx", "_comment": "why"}}}),
        encoding="utf-8",
    )
    assert load_config(str(path)) == {"fetch": {"command": "uvx"}}


def test_the_shipped_example_config_parses():
    cfg = load_config("mcp.example.json")
    assert "fetch" in cfg and "_comment" not in cfg["fetch"]
    assert cfg["fetch"]["approve"] == "none"


# --- when MCP is used at all --------------------------------------------------


def test_mcp_is_off_when_there_is_no_config(tmp_path):
    """Zero-config runs must not start third-party processes."""
    from teacup_agent.cli import _resolve_mcp

    assert _resolve_mcp(None, root=tmp_path) is None


def test_an_mcp_json_in_the_project_is_the_opt_in(tmp_path):
    """The file's existence is the consent; it should not need a flag every run."""
    from teacup_agent.cli import _resolve_mcp

    (tmp_path / "mcp.json").write_text("{}", encoding="utf-8")
    assert _resolve_mcp(None, root=tmp_path) == str(tmp_path / "mcp.json")


def test_explicit_path_wins_and_off_disables(tmp_path):
    from teacup_agent.cli import _resolve_mcp

    (tmp_path / "mcp.json").write_text("{}", encoding="utf-8")
    assert _resolve_mcp("other.json", root=tmp_path) == "other.json"
    assert _resolve_mcp("off", root=tmp_path) is None
