"""Hooks (roadmap #13): a project-local hooks.py can veto a call by argument, rewrite
a result, and — the one thing that can say yes with nobody watching — approve a
normally-gated call under --approve hooks. See docs/threat-model.md for why the third
one is not a weakening of "deny by default when nobody is watching".
"""

from __future__ import annotations

import json

import pytest

from teacup_agent import hooks, loop, tools
from teacup_agent.cli import _make_approver
from teacup_agent.evals import tool_results_follow_their_call
from teacup_agent.memory import NullMemory
from teacup_agent.model import ScriptedModel, assistant_calls, assistant_says


def _write_hooks(tmp_path, body: str):
    path = tmp_path / "hooks.py"
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _clean_hooks():
    """Hooks state is module-global (same pattern as skills/subagent) — make sure a
    failed test never leaks a loaded hook into the next one."""
    yield
    hooks.unload()


class _Call:
    def __init__(self, name: str, arguments: dict):
        self.name = name
        self.arguments = json.dumps(arguments)


# --- before_tool_call: veto ---------------------------------------------------


def test_before_tool_call_vetoes_by_argument(tmp_path):
    hooks_path = _write_hooks(
        tmp_path,
        """
def before_tool_call(call):
    import json
    args = json.loads(call.arguments)
    if call.name == "send_email" and not args["to"].endswith("@example.com"):
        return "ERROR: only @example.com allowed"
    return None
""",
    )
    state = loop.run(
        "send an email",
        ScriptedModel(
            [
                assistant_calls([("send_email", {"to": "a@evil.com", "subject": "s", "body": "b"})]),
                assistant_says("handled"),
            ]
        ),
        memory=NullMemory(),
        hooks=hooks_path,
        approve=lambda call, spec: True,  # approval would say yes — the veto must fire first
    )
    assert state.trace[0].skip_reason == "vetoed"
    assert not state.trace[0].executed
    assert "only @example.com" in state.trace[0].result
    assert tool_results_follow_their_call(state)  # a veto still gets exactly one result


def test_before_tool_call_allows_when_it_returns_none(tmp_path):
    hooks_path = _write_hooks(
        tmp_path,
        """
def before_tool_call(call):
    return None
""",
    )
    state = loop.run(
        "do arithmetic",
        ScriptedModel([assistant_calls([("calculate", {"expression": "1+1"})]), assistant_says("2")]),
        memory=NullMemory(),
        hooks=hooks_path,
    )
    assert state.trace[0].executed
    assert state.trace[0].result == "2"


def test_before_tool_call_broken_hook_fails_closed(tmp_path):
    hooks_path = _write_hooks(
        tmp_path,
        """
def before_tool_call(call):
    raise RuntimeError("boom")
""",
    )
    assert hooks.load(hooks_path)
    result = hooks.veto(_Call("calculate", {"expression": "1+1"}))
    assert result is not None and "boom" in result  # a broken veto hook denies, not allows


# --- after_tool_result: rewrite ------------------------------------------------


def test_after_tool_result_rewrites_the_result(tmp_path):
    hooks_path = _write_hooks(
        tmp_path,
        """
def after_tool_result(call, result):
    return result + " [reviewed]"
""",
    )
    state = loop.run(
        "do arithmetic",
        ScriptedModel([assistant_calls([("calculate", {"expression": "1+1"})]), assistant_says("2")]),
        memory=NullMemory(),
        hooks=hooks_path,
    )
    assert state.trace[0].result == "2 [reviewed]"


def test_after_tool_result_broken_hook_is_a_no_op(tmp_path):
    hooks_path = _write_hooks(
        tmp_path,
        """
def after_tool_result(call, result):
    raise RuntimeError("boom")
""",
    )
    assert hooks.load(hooks_path)
    assert hooks.rewrite(_Call("calculate", {}), "2") == "2"  # unrewritten, not crashed


# --- approve_tool_call: the one hook that can say yes --------------------------


def test_approve_tool_call_grants_approval_with_no_tty(tmp_path, monkeypatch):
    hooks_path = _write_hooks(
        tmp_path,
        """
def approve_tool_call(call, spec):
    return call.name == "send_email"
""",
    )
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)  # nobody is watching
    assert hooks.load(hooks_path)
    approve = _make_approver("hooks", quiet=True)
    assert approve(_Call("send_email", {}), tools.REGISTRY["send_email"]) is True


def test_approve_tool_call_no_opinion_falls_back_to_deny_without_tty(tmp_path, monkeypatch):
    hooks_path = _write_hooks(
        tmp_path,
        """
def approve_tool_call(call, spec):
    return None  # no opinion on anything
""",
    )
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert hooks.load(hooks_path)
    approve = _make_approver("hooks", quiet=True)
    assert approve(_Call("send_email", {}), tools.REGISTRY["send_email"]) is False


def test_hooks_policy_with_no_hooks_loaded_behaves_like_auto(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    approve = _make_approver("hooks", quiet=True)
    assert approve(_Call("send_email", {}), tools.REGISTRY["send_email"]) is False


def test_approve_tool_call_broken_hook_is_no_opinion(tmp_path):
    hooks_path = _write_hooks(
        tmp_path,
        """
def approve_tool_call(call, spec):
    raise RuntimeError("boom")
""",
    )
    assert hooks.load(hooks_path)
    assert hooks.approve(_Call("send_email", {}), None) is None


# --- loading -------------------------------------------------------------------


def test_load_returns_false_when_no_hooks_defined(tmp_path):
    hooks_path = _write_hooks(tmp_path, "# no hooks defined here\n")
    assert hooks.load(hooks_path) is False


def test_load_returns_false_for_a_missing_file(tmp_path):
    assert hooks.load(tmp_path / "does-not-exist.py") is False


def test_unloaded_hooks_are_all_pass_through():
    assert hooks.veto(_Call("calculate", {})) is None
    assert hooks.rewrite(_Call("calculate", {}), "2") == "2"
    assert hooks.approve(_Call("send_email", {}), None) is None
