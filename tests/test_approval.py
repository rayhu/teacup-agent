"""The human-approval gate.

Read-only tools must never be gated (ask too often and people go numb, and numb
people approve with their eyes closed). Side-effecting tools must be **denied by
default** when nobody is watching, and the message protocol must survive a denial.
"""

import json

import pytest

from mini_agent import loop, tools
from mini_agent.cli import _make_approver
from mini_agent.evals import tool_results_follow_their_call
from mini_agent.memory import NullMemory
from mini_agent.model import ScriptedModel, assistant_calls, assistant_says


@pytest.fixture
def outbox(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # send_email writes into the working directory
    return tmp_path / "outbox.jsonl"


def _send(**kw):
    return loop.run(
        "send an email",
        ScriptedModel(
            [
                assistant_calls([("send_email", {"to": "a@b.c", "subject": "subject", "body": "body"})]),
                assistant_says("handled"),
            ]
        ),
        memory=NullMemory(),
        **kw,
    )


def test_default_policy_denies_and_nothing_is_sent(outbox):
    state = _send()  # no approve argument = the default deny_all
    assert not outbox.exists(), "a denied operation must leave no side effect"
    assert state.trace[0].skip_reason == "denied" and not state.trace[0].executed
    assert "was NOT executed" in state.trace[0].result
    assert tool_results_follow_their_call(state)


def test_approved_call_actually_runs(outbox):
    state = _send(approve=lambda call, spec: True)
    assert state.trace[0].executed and state.trace[0].skip_reason == ""
    assert json.loads(outbox.read_text().strip())["to"] == "a@b.c"


def test_approver_sees_which_tool_it_is_approving(outbox):
    seen = []
    _send(approve=lambda call, spec: seen.append((call.name, spec.requires_approval)) or False)
    assert seen == [("send_email", True)]


def test_read_only_tools_are_never_gated(outbox):
    asked = []
    state = loop.run(
        "do some arithmetic",
        ScriptedModel([assistant_calls([("calculate", {"expression": "1+1"})]), assistant_says("2")]),
        memory=NullMemory(),
        approve=lambda call, spec: asked.append(call.name) or True,
    )
    assert asked == []  # it was never asked
    assert state.trace[0].result == "2"


def test_cli_auto_policy_denies_without_a_terminal(monkeypatch):
    """In CI or a background job there is no terminal to ask, so it must deny."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    approve = _make_approver("auto", quiet=True)
    assert approve(type("C", (), {"name": "send_email", "arguments": "{}"})(), tools.REGISTRY["send_email"]) is False


def test_cli_allow_policy_is_explicit_yolo():
    approve = _make_approver("allow", quiet=True)
    assert approve(type("C", (), {"name": "send_email", "arguments": "{}"})(), tools.REGISTRY["send_email"]) is True


# --- the --plan flag ----------------------------------------------------------


def test_plan_auto_follows_the_run_mode():
    """auto is honest about what it does: on for live, off for the offline demo."""
    from mini_agent.cli import _resolve_plan

    assert _resolve_plan("auto", live=True) is True
    assert _resolve_plan("auto", live=False) is False


def test_plan_on_and_off_are_absolute():
    from mini_agent.cli import _resolve_plan

    assert _resolve_plan("on", live=False) is True  # plan even the offline demo
    assert _resolve_plan("off", live=True) is False
