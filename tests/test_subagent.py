"""Subagents: the one form of context compression that discards nothing.

The parent pays one step and one tool schema; the child reads the bulk in a context
the parent never sees. These tests pin down that isolation, and the guards that keep a
child from spending or recursing without limit.
"""

import pytest

from mini_agent import loop, subagent, tools
from mini_agent.evals import tool_results_follow_their_call
from mini_agent.memory import NullMemory
from mini_agent.model import ScriptedModel, assistant_calls, assistant_says

BIG = "PAGE TEXT " * 900  # ~9000 characters the parent must never see


@pytest.fixture(autouse=True)
def big_tool(monkeypatch):
    registry = dict(tools.REGISTRY)
    registry["read_big"] = tools.Tool(
        "read_big", "returns a wall of text", {"type": "object", "properties": {}},
        lambda **_: BIG,
    )
    monkeypatch.setattr(tools, "REGISTRY", registry)
    yield
    subagent.disable()


class TwoLevel(ScriptedModel):
    """Parent delegates once; the child reads the big page and summarises it."""

    def __init__(self, child_answer="Three sentences about the page.", child_reads=True):
        super().__init__([])
        self.child_answer = child_answer
        self.child_reads = child_reads
        self.child_tool_names: list[str] | None = None

    def complete(self, messages, tools_):
        goal = messages[1]["content"] if len(messages) > 1 else ""
        used_tools = any(m.get("role") == "tool" for m in messages)
        if goal.startswith("CHILD"):
            self.child_tool_names = [t["function"]["name"] for t in tools_]
            if self.child_reads and not used_tools:
                return assistant_calls([("read_big", {})])
            return assistant_says(self.child_answer)
        if not used_tools:
            return assistant_calls([("delegate", {"task": "CHILD: read and summarise"})])
        return assistant_says("Done, based on the subagent's summary.")


def _run(model, **kw):
    return loop.run(
        "summarise that page", model, memory=NullMemory(),
        subagents=True, budget=kw.pop("budget", 1.0), max_steps=kw.pop("max_steps", 4), **kw,
    )


# --- isolation ----------------------------------------------------------------


def test_the_bulk_never_reaches_the_parent_context():
    state = _run(TwoLevel())
    assert state.subagent_runs == 1
    assert BIG[:50] not in str(state.messages)  # the point of the whole feature
    assert "Three sentences" in state.trace[0].result


def test_the_child_still_read_it(tmp_path):
    """Isolation must not mean the work did not happen: the child's own trace and
    state.json are written under the parent's run directory."""
    _run(TwoLevel(), run_dir=tmp_path)
    child_state = tmp_path / "sub01" / "state.json"
    assert child_state.is_file()
    assert BIG[:50] in child_state.read_text(encoding="utf-8")


def test_delegation_keeps_the_message_protocol_valid():
    state = _run(TwoLevel())
    assert tool_results_follow_their_call(state) and state.status == "done"


# --- guards -------------------------------------------------------------------


def test_a_child_cannot_delegate_further():
    """One level only. A recursion here is a recursion that spends money."""
    model = TwoLevel()
    _run(model)
    assert model.child_tool_names is not None
    assert "delegate" not in model.child_tool_names
    assert "read_big" in model.child_tool_names  # everything else is still there


def test_the_child_spends_the_parents_money():
    state = _run(TwoLevel(), budget=1.0)
    assert state.remaining_budget < 1.0  # the child's cost landed on the parent
    assert state.input_tokens_total >= 0


def test_a_broke_parent_cannot_fund_a_child():
    tools.REGISTRY  # fixture applied
    subagent.enable(type("S", (), {"remaining_budget": 0.0, "time_left": lambda self: None,
                                   "subagent_runs": 0, "charge": lambda self, c: None,
                                   "input_tokens_total": 0, "cached_tokens_total": 0})(), None)
    out = tools.execute("delegate", '{"task": "anything"}')
    assert out.startswith("ERROR:") and "budget" in out


def test_delegate_is_absent_unless_the_run_enables_it():
    """An unbound tool schema would sit in the prefix of every request for a
    capability the run cannot use."""
    assert "delegate" not in tools.REGISTRY
    state = loop.run("no delegation", ScriptedModel([assistant_says("ok")]),
                     memory=NullMemory(), subagents=False)
    assert state.status == "done" and "delegate" not in tools.REGISTRY


def test_the_tool_is_removed_again_after_the_run():
    _run(TwoLevel())
    assert "delegate" not in tools.REGISTRY


def test_delegate_gets_a_longer_timeout_than_a_page_fetch():
    """A child run is not a single tool call; the loop's 30s default would kill it."""
    subagent.enable(type("S", (), {"remaining_budget": 1.0})(), None)
    assert tools.REGISTRY["delegate"].timeout > 30.0


# --- failure handling ---------------------------------------------------------


def test_a_child_without_an_answer_becomes_an_error_the_parent_can_read():
    state = _run(TwoLevel(child_answer="", child_reads=False))
    assert state.trace[0].result.startswith("ERROR:")
    assert state.status == "done"  # the parent carried on


def test_child_events_are_labelled_so_a_human_can_follow():
    seen = []
    _run(TwoLevel(), on_event=lambda e, d: seen.append((e, d.get("subagent"))))
    assert any(sub == 1 for _, sub in seen)  # child activity is visible
    assert any(sub is None for _, sub in seen)  # parent activity still is too
