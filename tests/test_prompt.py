"""system prompt 与每轮状态注入。

这两件事是上一次失败运行的直接修复：模型既不知道今天几号（于是把真实新闻判为「可疑」），
也不知道自己还剩多少预算（于是把待办清单交回给了一个没人看的终端）。
"""

from mini_agent import loop
from mini_agent.memory import NullMemory
from mini_agent.model import ScriptedModel, assistant_calls, assistant_says


def _run(script, **kw):
    model = ScriptedModel(list(script))
    state = loop.run("测试目标", model, memory=NullMemory(), today="2026-08-25", **kw)
    return model, state


def test_system_prompt_carries_today_and_autonomy_rules():
    _, state = _run([assistant_says("好")])
    system = state.messages[0]["content"]
    assert "2026-08-25" in system
    assert "以检索结果为准" in system  # 别拿过期记忆否定检索结果
    assert "没有人会回答你的提问" in system  # 自动模式，别请求许可


def test_status_note_injected_before_every_model_call():
    _, state = _run(
        [assistant_calls([("calculate", {"expression": "1+1"})]), assistant_says("2")],
        max_steps=5,
        budget=0.5,
    )
    notes = [m for m in state.messages if str(m.get("content", "")).startswith("[运行状态]")]
    assert len(notes) == 2  # 两轮，每轮一条
    assert "第 1/5 轮" in notes[0]["content"] and "第 2/5 轮" in notes[1]["content"]
    assert "$0.5" in notes[0]["content"]  # 预算对模型可见


def test_status_note_is_the_last_thing_model_sees():
    model, _ = _run([assistant_says("好")])
    assert str(model.calls[0][-1]["content"]).startswith("[运行状态]")


def test_status_note_does_not_break_prefix_caching():
    """状态行追加在末尾，不改 system 前缀 —— 否则 prompt caching（#2）就废了。"""
    model, _ = _run(
        [assistant_calls([("calculate", {"expression": "1+1"})]), assistant_says("2")]
    )
    first, second = model.calls[0], model.calls[1]
    assert first[0] == second[0]  # system 消息逐字相同
    assert second[: len(first)] == first  # 第二轮请求是第一轮的严格扩展


# --- 资源耗尽时的行为 --------------------------------------------------------


def test_last_turn_has_no_tools_and_says_so():
    """光靠提示词不够硬，最后一轮直接把工具清单清空。"""
    model, state = _run(
        [assistant_calls([("calculate", {"expression": "1+1"})]) for _ in range(5)],
        max_steps=2,
    )
    assert model.tool_specs[0]  # 第 1 轮有工具
    assert model.tool_specs[1] == []  # 第 2 轮（最后一轮）没有
    last_note = [m for m in state.messages if str(m.get("content", "")).startswith("[运行状态]")][-1]
    assert "这是最后一轮" in last_note["content"]


def test_salvage_turn_runs_after_resources_exhausted():
    model, state = _run(
        [
            assistant_calls([("calculate", {"expression": "1+1"})]),
            assistant_calls([("calculate", {"expression": "1+1"})]),
            assistant_says("尽力给出的结论：1+1=2。"),
        ],
        max_steps=2,
    )
    assert state.status == "max_steps" and state.salvaged
    assert state.answer and not state.answer.startswith("（未得出最终答案")
    assert model.tool_specs[-1] == []  # 收尾轮也不给工具
    assert "[强制收尾]" in state.messages[-2]["content"]


# --- 时间预算 ----------------------------------------------------------------


def test_status_note_shows_time_left_and_urges_when_tight():
    model = ScriptedModel([assistant_says("好")])
    state = loop.run(
        "测试目标",
        model,
        memory=NullMemory(),
        today="2026-08-26",
        time_budget=60.0,
        clock=iter([0.0, 55.0, 55.0]).__next__,
    )
    note = [m for m in state.messages if str(m.get("content", "")).startswith("[运行状态]")][0]
    assert "剩余时间 5 秒" in note["content"]
    assert "请开始收尾" in note["content"]  # 步数还很富余，是时间在催


def test_time_brake_stops_run_with_budget_and_steps_left():
    """钱和步数都还有富余，时间照样能把它拦下来。"""
    state = loop.run(
        "测试目标",
        ScriptedModel(
            [
                assistant_calls([("calculate", {"expression": "1+1"})]),
                assistant_says("超时前的结论：1+1=2。"),
            ]
        ),
        memory=NullMemory(),
        today="2026-08-26",
        time_budget=10.0,
        clock=iter([0.0, 1.0, 30.0, 30.0]).__next__,
    )
    assert state.status == "out_of_time"
    assert state.remaining_budget > 0 and state.step < state.max_steps
    assert state.salvaged and state.answer  # 超时也不空手而归
    assert state.snapshot()["elapsed_s"] == 30.0
