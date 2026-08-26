"""落盘与恢复。

「每步一存」不是为了整洁，是为了**最需要它的那次崩溃**能留下东西 ——
跑完再存的话，崩掉的那次恰好什么都没有。
"""

import json

from mini_agent import loop, persist
from mini_agent.evals import tool_results_follow_their_call
from mini_agent.memory import NullMemory
from mini_agent.model import ScriptedModel, assistant_calls, assistant_says


def test_state_is_saved_after_every_step(tmp_path):
    loop.run(
        "落盘测试",
        ScriptedModel(
            [
                assistant_calls([("calculate", {"expression": "1+1"})]),
                assistant_calls([("calculate", {"expression": "2+2"})]),
                assistant_says("好"),
            ]
        ),
        memory=NullMemory(),
        run_dir=tmp_path,
    )
    saved = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert saved["status"] == "done"
    assert len(saved["trace"]) == 2
    assert saved["messages"][1]["content"] == "落盘测试"


def test_nothing_written_when_run_dir_is_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loop.run("不落盘", ScriptedModel([assistant_says("好")]), memory=NullMemory(), run_dir=None)
    assert list(tmp_path.iterdir()) == []


def test_resume_continues_without_redoing_work(tmp_path):
    # 第一段：步数只给 1，必然触顶
    first = loop.run(
        "恢复测试",
        ScriptedModel([assistant_calls([("calculate", {"expression": "1+1"})])] * 3),
        memory=NullMemory(),
        max_steps=1,
        run_dir=tmp_path,
    )
    assert first.status == "max_steps"
    tool_calls_before = len(first.trace)

    # 第二段：从盘上读回来，再给几轮
    resumed = persist.load(tmp_path)
    assert resumed.step == 1 and resumed.trace  # 上一段的工作还在
    resumed.max_steps = resumed.step + 3

    second = loop.run(
        "恢复测试",
        ScriptedModel([assistant_says("接着跑完了")]),
        memory=NullMemory(),
        resume=resumed,
        run_dir=tmp_path,
    )

    assert second.status == "done" and second.answer == "接着跑完了"
    assert second.step == 2  # 从第 2 轮接着数，没从头再来
    assert len(second.trace) == tool_calls_before  # 已完成的工具调用没有重做
    assert second.messages[1]["content"] == "恢复测试"  # 原始目标还在
    assert tool_results_follow_their_call(second)  # 恢复后消息协议依然完整


def test_resume_keeps_the_system_prefix_byte_identical(tmp_path):
    first = loop.run(
        "前缀测试",
        ScriptedModel([assistant_calls([("calculate", {"expression": "1+1"})])] * 3),
        memory=NullMemory(),
        max_steps=1,
        run_dir=tmp_path,
        today="2026-08-26",
    )
    resumed = persist.load(tmp_path)
    resumed.max_steps += 2
    model = ScriptedModel([assistant_says("好")])
    loop.run("前缀测试", model, memory=NullMemory(), resume=resumed, run_dir=tmp_path, today="2099-01-01")

    # 恢复时不重建 system 消息：重建会让前缀变化，攒下的 prompt cache 全作废
    assert model.calls[0][0]["content"] == first.messages[0]["content"]
    assert "2099" not in model.calls[0][0]["content"]
