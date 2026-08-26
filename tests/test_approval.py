"""人机确认门。

只读工具不该被拦（问多了会麻木，麻木了就闭眼点同意）；
有副作用的工具在无人值守时必须**默认拒绝**，而且拒绝后消息协议依然完整。
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
    monkeypatch.chdir(tmp_path)  # send_email 写在当前目录
    return tmp_path / "outbox.jsonl"


def _send(**kw):
    return loop.run(
        "发封邮件",
        ScriptedModel(
            [
                assistant_calls([("send_email", {"to": "a@b.c", "subject": "标题", "body": "正文"})]),
                assistant_says("处理完了"),
            ]
        ),
        memory=NullMemory(),
        **kw,
    )


def test_default_policy_denies_and_nothing_is_sent(outbox):
    state = _send()  # 不传 approve = 默认 deny_all
    assert not outbox.exists(), "被拒绝的操作绝不能留下副作用"
    assert state.trace[0].skip_reason == "denied" and not state.trace[0].executed
    assert "没有执行" in state.trace[0].result
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
        "算个数",
        ScriptedModel([assistant_calls([("calculate", {"expression": "1+1"})]), assistant_says("2")]),
        memory=NullMemory(),
        approve=lambda call, spec: asked.append(call.name) or True,
    )
    assert asked == []  # 压根没问
    assert state.trace[0].result == "2"


def test_cli_auto_policy_denies_without_a_terminal(monkeypatch):
    """CI / 后台任务里没有终端可问 —— 此时必须拒绝，而不是放行。"""
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    approve = _make_approver("auto", quiet=True)
    assert approve(type("C", (), {"name": "send_email", "arguments": "{}"})(), tools.REGISTRY["send_email"]) is False


def test_cli_allow_policy_is_explicit_yolo():
    approve = _make_approver("allow", quiet=True)
    assert approve(type("C", (), {"name": "send_email", "arguments": "{}"})(), tools.REGISTRY["send_email"]) is True
