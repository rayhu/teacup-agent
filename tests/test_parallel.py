"""Parallel tool execution, per-call timeouts, and backoff retries on model calls.

Parallelism is exactly what breaks the two message-protocol invariants (order, and
one result per id), so every case here re-checks them.
"""

import time

import pytest

from teacup_agent import loop, tools
from teacup_agent.evals import tool_results_follow_their_call
from teacup_agent.memory import NullMemory
from teacup_agent.model import ScriptedModel, assistant_calls, assistant_says


@pytest.fixture
def slow_tools(monkeypatch):
    """Register fake tools that sleep 0.3s each, to measure parallelism."""
    registry = dict(tools.REGISTRY)

    def make(name):
        def fn(**kwargs):
            time.sleep(0.3)
            return f"{name} done"

        return tools.Tool(name, "slow tool", {"type": "object", "properties": {}}, fn)

    for n in ("slow_a", "slow_b", "slow_c"):
        registry[n] = make(n)
    monkeypatch.setattr(tools, "REGISTRY", registry)
    return registry


def test_tools_run_in_parallel_and_keep_order(slow_tools):
    started = time.monotonic()
    state = loop.run(
        "parallelism test",
        ScriptedModel(
            [
                assistant_calls([("slow_a", {}), ("slow_b", {}), ("slow_c", {})]),
                assistant_says("done"),
            ]
        ),
        memory=NullMemory(),
    )
    elapsed = time.monotonic() - started

    assert elapsed < 0.6, f"serial would take 0.9s for three 0.3s tools, got {elapsed:.2f}s"
    assert [t.name for t in state.trace] == ["slow_a", "slow_b", "slow_c"]  # order kept
    assert [t.result for t in state.trace] == ["slow_a done", "slow_b done", "slow_c done"]
    assert tool_results_follow_their_call(state)


def test_slow_tool_times_out_but_still_gets_a_result_message(slow_tools):
    state = loop.run(
        "timeout test",
        ScriptedModel([assistant_calls([("slow_a", {}), ("calculate", {"expression": "1+1"})]), assistant_says("ok")]),
        memory=NullMemory(),
        tool_timeout=0.1,  # shorter than the slow tool's 0.3s
        clock=iter([0.0, 0.0, 0.0, 0.0, 0.0]).__next__,  # keep the time brake out of it
        time_budget=None,
    )
    slow_result = state.trace[0].result
    assert slow_result.startswith("ERROR:") and "did not return within" in slow_result
    assert "does **not** mean" in slow_result  # a timeout is not "it does not exist"
    assert state.trace[1].result == "2"  # the fast one still returns
    assert tool_results_follow_their_call(state)  # the timed-out call was filled too


# --- backoff retries on model calls (the other half of roadmap #3) -----------


class _FlakyModel:
    def __init__(self, errors):
        self.errors = list(errors)
        self.attempts = 0

    def complete(self, messages, tools_):
        self.attempts += 1
        if self.errors:
            raise self.errors.pop(0)
        return assistant_says("succeeded at last")

    def tool_result_item(self, call, result):
        from teacup_agent.model import chat_tool_result

        return chat_tool_result(call, result)


def _err(status):
    e = RuntimeError(f"status {status}")
    e.status_code = status
    return e


def test_retries_on_429_without_consuming_a_step(monkeypatch):
    model = _FlakyModel([_err(429), _err(503)])
    slept = []
    monkeypatch.setattr(loop.time, "sleep", slept.append)

    state = loop.run("retry test", model, memory=NullMemory())

    assert state.status == "done" and state.answer == "succeeded at last"
    assert model.attempts == 3
    assert state.step == 1  # a retry is not a step
    assert slept == [1, 2]  # backoff of 1s then 2s


def test_does_not_retry_client_errors(monkeypatch):
    model = _FlakyModel([_err(400), _err(400), _err(400)])
    monkeypatch.setattr(loop.time, "sleep", lambda *_: None)

    state = loop.run("retry test", model, memory=NullMemory())

    assert model.attempts == 1  # a 400 returns the same answer however often you ask
    assert state.status == "error" and "400" in state.answer
