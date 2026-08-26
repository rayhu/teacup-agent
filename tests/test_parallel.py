"""并行执行工具 + 单次超时 + 模型调用退避重试。

并行最容易打破的正是消息协议的两条不变量（顺序、每个 id 恰好一条结果），
所以这里每个用例都顺带验一遍。
"""

import time

import pytest

from mini_agent import loop, tools
from mini_agent.evals import tool_results_follow_their_call
from mini_agent.memory import NullMemory
from mini_agent.model import ScriptedModel, assistant_calls, assistant_says


@pytest.fixture
def slow_tools(monkeypatch):
    """注册两个各睡 0.3 秒的假工具，用来量并行度。"""
    registry = dict(tools.REGISTRY)

    def make(name):
        def fn(**kwargs):
            time.sleep(0.3)
            return f"{name} done"

        return tools.Tool(name, "慢工具", {"type": "object", "properties": {}}, fn)

    for n in ("slow_a", "slow_b", "slow_c"):
        registry[n] = make(n)
    monkeypatch.setattr(tools, "REGISTRY", registry)
    return registry


def test_tools_run_in_parallel_and_keep_order(slow_tools):
    started = time.monotonic()
    state = loop.run(
        "并行测试",
        ScriptedModel(
            [
                assistant_calls([("slow_a", {}), ("slow_b", {}), ("slow_c", {})]),
                assistant_says("好了"),
            ]
        ),
        memory=NullMemory(),
    )
    elapsed = time.monotonic() - started

    assert elapsed < 0.6, f"三个 0.3 秒的工具串行要 0.9 秒，实际 {elapsed:.2f} 秒"
    assert [t.name for t in state.trace] == ["slow_a", "slow_b", "slow_c"]  # 顺序不乱
    assert [t.result for t in state.trace] == ["slow_a done", "slow_b done", "slow_c done"]
    assert tool_results_follow_their_call(state)


def test_slow_tool_times_out_but_still_gets_a_result_message(slow_tools):
    state = loop.run(
        "超时测试",
        ScriptedModel([assistant_calls([("slow_a", {}), ("calculate", {"expression": "1+1"})]), assistant_says("好")]),
        memory=NullMemory(),
        tool_timeout=0.1,  # 比慢工具的 0.3 秒短
        clock=iter([0.0, 0.0, 0.0, 0.0, 0.0]).__next__,  # 别让时间刹车抢戏
        time_budget=None,
    )
    slow_result = state.trace[0].result
    assert slow_result.startswith("ERROR:") and "超过" in slow_result
    assert "不代表" in slow_result  # 超时 ≠ 该信息不存在
    assert state.trace[1].result == "2"  # 快的那个照常返回
    assert tool_results_follow_their_call(state)  # 超时的调用也回填了


# --- 模型调用的退避重试（roadmap #3 的另一半）--------------------------------


class _FlakyModel:
    def __init__(self, errors):
        self.errors = list(errors)
        self.attempts = 0

    def complete(self, messages, tools_):
        self.attempts += 1
        if self.errors:
            raise self.errors.pop(0)
        return assistant_says("终于成功了")

    def tool_result_item(self, call, result):
        from mini_agent.model import chat_tool_result

        return chat_tool_result(call, result)


def _err(status):
    e = RuntimeError(f"status {status}")
    e.status_code = status
    return e


def test_retries_on_429_without_consuming_a_step(monkeypatch):
    model = _FlakyModel([_err(429), _err(503)])
    slept = []
    monkeypatch.setattr(loop.time, "sleep", slept.append)

    state = loop.run("重试测试", model, memory=NullMemory())

    assert state.status == "done" and state.answer == "终于成功了"
    assert model.attempts == 3
    assert state.step == 1  # 重试不算 step
    assert slept == [1, 2]  # 退避 1s、2s


def test_does_not_retry_client_errors(monkeypatch):
    model = _FlakyModel([_err(400), _err(400), _err(400)])
    monkeypatch.setattr(loop.time, "sleep", lambda *_: None)

    state = loop.run("重试测试", model, memory=NullMemory())

    assert model.attempts == 1  # 400 重试多少次都一样，别浪费时间
    assert state.status == "error" and "400" in state.answer
