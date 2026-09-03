"""Hooks: a project-local before_tool_call may veto by argument, an
after_tool_result may rewrite. Both are optional, both must not be able to sink
a run, and the veto must reach the model as an ordinary ERROR: result.
"""

import pytest

from teacup_agent import hooks, loop
from teacup_agent.evals import tool_results_follow_their_call
from teacup_agent.memory import NullMemory
from teacup_agent.model import ScriptedModel, assistant_calls, assistant_says

HOOKS_BOTH = '''
def before_tool_call(name, arguments):
    if name == "send_email" and not arguments.get("to", "").endswith("@allowed.com"):
        return f"recipient {arguments.get('to')!r} is not allowed"
    return None


def after_tool_result(name, arguments, result):
    if name == "calculate":
        return result + " (checked)"
    return None
'''

HOOKS_VETO_ONLY = '''
def before_tool_call(name, arguments):
    return "nope"
'''

HOOKS_REWRITE_ONLY = '''
def after_tool_result(name, arguments, result):
    return result.upper()
'''

HOOKS_BROKEN = '''
def before_tool_call(name, arguments):
    raise RuntimeError("boom")


def after_tool_result(name, arguments, result):
    raise RuntimeError("boom too")
'''


@pytest.fixture(autouse=True)
def _teardown():
    yield
    hooks.disable()


def _write(tmp_path, text) -> str:
    path = tmp_path / "hooks.py"
    path.write_text(text, encoding="utf-8")
    return str(path)


# --- the loader itself ---------------------------------------------------------


def test_a_hook_can_veto(tmp_path):
    hooks.load(_write(tmp_path, HOOKS_VETO_ONLY))
    assert hooks.veto("send_email", {}) == "ERROR: nope"


def test_no_veto_means_none(tmp_path):
    hooks.load(_write(tmp_path, HOOKS_BOTH))
    assert hooks.veto("send_email", {"to": "a@allowed.com"}) is None


def test_veto_is_prefixed_error_only_once(tmp_path):
    path = tmp_path / "hooks.py"
    path.write_text('def before_tool_call(name, arguments):\n    return "ERROR: already prefixed"\n', encoding="utf-8")
    hooks.load(str(path))
    assert hooks.veto("x", {}) == "ERROR: already prefixed"


def test_a_hook_can_rewrite(tmp_path):
    hooks.load(_write(tmp_path, HOOKS_REWRITE_ONLY))
    assert hooks.rewrite("any_tool", {}, "hello") == "HELLO"


def test_no_rewrite_means_unchanged(tmp_path):
    hooks.load(_write(tmp_path, HOOKS_BOTH))
    assert hooks.rewrite("search_web", {}, "unchanged") == "unchanged"


def test_a_file_defining_only_one_function_leaves_the_other_a_noop(tmp_path):
    hooks.load(_write(tmp_path, HOOKS_VETO_ONLY))
    assert hooks.rewrite("anything", {}, "same") == "same"

    hooks.load(_write(tmp_path, HOOKS_REWRITE_ONLY))
    assert hooks.veto("anything", {}) is None


def test_disable_resets_both(tmp_path):
    hooks.load(_write(tmp_path, HOOKS_BOTH))
    hooks.disable()
    assert hooks.veto("send_email", {"to": "x@other.com"}) is None
    assert hooks.rewrite("calculate", {}, "1") == "1"


# --- _resolve_hooks, same three-way logic as _resolve_mcp/_resolve_skills ------


def test_hooks_are_off_when_there_is_no_file(tmp_path):
    from teacup_agent.cli import _resolve_hooks

    assert _resolve_hooks(None, root=tmp_path) is None


def test_a_hooks_py_in_the_project_is_the_opt_in(tmp_path):
    (tmp_path / "hooks.py").write_text("", encoding="utf-8")
    from teacup_agent.cli import _resolve_hooks

    assert _resolve_hooks(None, root=tmp_path) == str(tmp_path / "hooks.py")


def test_explicit_path_wins_and_off_disables(tmp_path):
    (tmp_path / "hooks.py").write_text("", encoding="utf-8")
    from teacup_agent.cli import _resolve_hooks

    assert _resolve_hooks("other.py", root=tmp_path) == "other.py"
    assert _resolve_hooks("off", root=tmp_path) is None


# --- loop.run() level: the veto/rewrite actually change what the model sees ----


def test_a_veto_by_argument_reaches_the_model_as_an_error(tmp_path):
    """The roadmap's own DoD: block by argument, not just by tool name."""
    state = loop.run(
        "send an email",
        ScriptedModel(
            [
                assistant_calls([("send_email", {"to": "a@blocked.com", "subject": "s", "body": "b"})]),
                assistant_says("handled"),
            ]
        ),
        memory=NullMemory(),
        hooks=_write(tmp_path, HOOKS_BOTH),
    )
    assert state.trace[0].skip_reason == "hook_vetoed" and not state.trace[0].executed
    assert state.trace[0].result.startswith("ERROR:") and "blocked.com" in state.trace[0].result
    assert tool_results_follow_their_call(state)


def test_an_allowed_argument_is_not_vetoed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # send_email writes into the working directory
    state = loop.run(
        "send an email",
        ScriptedModel(
            [
                assistant_calls([("send_email", {"to": "a@allowed.com", "subject": "s", "body": "b"})]),
                assistant_says("handled"),
            ]
        ),
        memory=NullMemory(),
        hooks=_write(tmp_path, HOOKS_BOTH),
        approve=lambda call, spec: True,
    )
    assert state.trace[0].executed and state.trace[0].skip_reason == ""


def test_a_rewrite_changes_the_recorded_result(tmp_path):
    state = loop.run(
        "compute something",
        ScriptedModel(
            [
                assistant_calls([("calculate", {"expression": "1+1"})]),
                assistant_says("2"),
            ]
        ),
        memory=NullMemory(),
        hooks=_write(tmp_path, HOOKS_BOTH),
    )
    assert state.trace[0].result == "2 (checked)"


def test_a_broken_hook_does_not_crash_the_run(tmp_path):
    state = loop.run(
        "compute something",
        ScriptedModel(
            [
                assistant_calls([("calculate", {"expression": "1+1"})]),
                assistant_says("2"),
            ]
        ),
        memory=NullMemory(),
        hooks=_write(tmp_path, HOOKS_BROKEN),
    )
    assert state.status == "done"
    assert state.trace[0].executed and state.trace[0].result == "2"  # unchanged, not crashed
    assert tool_results_follow_their_call(state)


def test_the_hook_module_is_disabled_after_the_run(tmp_path):
    loop.run(
        "task", ScriptedModel([assistant_says("ok")]), memory=NullMemory(),
        hooks=_write(tmp_path, HOOKS_BOTH),
    )
    assert hooks.veto("send_email", {"to": "x@other.com"}) is None


def test_no_hooks_means_no_change_in_behavior():
    state = loop.run(
        "send an email",
        ScriptedModel([assistant_calls([("send_email", {"to": "a@b.c", "subject": "s", "body": "b"})]), assistant_says("done")]),
        memory=NullMemory(),
    )
    assert state.trace[0].skip_reason == "denied"  # the ordinary approval gate, unaffected
