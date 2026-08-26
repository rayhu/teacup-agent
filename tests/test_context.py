"""上下文管理：外置大结果 + 超限压缩。

压缩最危险的地方不是「摘要写得好不好」，而是**把工具调用和它的结果拆散** ——
那会让下一轮请求直接 400。所以这里每条用例都顺带验一遍消息协议不变量。
"""

import pytest

from mini_agent import context as ctx
from mini_agent import loop, tools
from mini_agent.evals import ScriptedWithSummarizer, tool_results_follow_their_call
from mini_agent.memory import NullMemory
from mini_agent.model import ScriptedModel, assistant_calls, assistant_says


def test_estimate_tokens_weighs_cjk_heavier():
    assert ctx.estimate_tokens("中文" * 100) > ctx.estimate_tokens("ab" * 100)


def test_safe_cut_points_never_split_a_call_from_its_result():
    messages = [
        {"role": "system"},
        {"role": "user"},
        {"role": "assistant", "tool_calls": [{"id": "a"}, {"id": "b"}]},
        {"role": "tool", "tool_call_id": "a"},
        {"role": "tool", "tool_call_id": "b"},
        {"role": "assistant", "content": "done"},
    ]
    points = ctx.safe_cut_points(messages)
    assert 3 not in points and 4 not in points  # 切在这里会拆散调用与结果
    assert 5 in points and 6 in points


def test_safe_cut_points_handles_responses_shape():
    messages = [
        {"role": "system"},
        {"role": "user"},
        {"type": "reasoning"},
        {"type": "function_call", "call_id": "fc_1"},
        {"type": "function_call_output", "call_id": "fc_1"},
    ]
    assert ctx.safe_cut_points(messages) == [1, 2, 3, 5]


# --- 外置 --------------------------------------------------------------------


@pytest.fixture
def big_tool(monkeypatch):
    registry = dict(tools.REGISTRY)
    registry["dump"] = tools.Tool(
        "dump", "吐一大坨", {"type": "object", "properties": {}}, lambda **_: "X" * 5000
    )
    monkeypatch.setattr(tools, "REGISTRY", registry)


def test_big_tool_result_is_written_to_disk_and_shrunk(big_tool, tmp_path):
    state = loop.run(
        "外置测试",
        ScriptedModel([assistant_calls([("dump", {})]), assistant_says("好")]),
        memory=NullMemory(),
        run_dir=tmp_path,
    )
    kept = [m for m in state.messages if m.get("role") == "tool"][0]["content"]
    assert len(kept) < 1200  # 上下文里只剩摘要
    assert "read_file" in kept  # 并且告诉模型怎么取回全文

    files = list(tmp_path.glob("*.txt"))
    assert len(files) == 1 and len(files[0].read_text()) == 5000  # 全文一个字没丢
    # 留痕记录的是「模型实际看到的东西」（精简版），全文在盘上 ——
    # trajectory eval 关心的是前者，复盘细节去读文件。
    assert state.trace[0].result == kept


# --- 压缩 --------------------------------------------------------------------


def test_compaction_replaces_history_and_keeps_protocol_valid():
    # 前几轮不断产生工具调用，把上下文撑大
    script = [assistant_calls([("calculate", {"expression": "1+1"})]) for _ in range(6)]
    script.append(assistant_says("结束"))
    model = ScriptedWithSummarizer(script)

    state = loop.run(
        "压缩测试",
        model,
        memory=NullMemory(),
        max_steps=8,
        context_limit=200,  # 故意设得很小，逼它压缩
        run_dir=None,
    )

    assert model.summaries >= 1 and state.compactions >= 1
    assert any("[上下文摘要]" in str(m.get("content", "")) for m in state.messages)
    assert tool_results_follow_their_call(state)  # 压缩后消息协议依然完整
    assert state.messages[0]["role"] == "system"  # 前缀没动，prompt caching 还在
    assert state.messages[1]["content"] == "压缩测试"  # 原始目标永远留着


def test_no_compaction_below_the_limit():
    model = ScriptedWithSummarizer([assistant_says("短任务")])
    loop.run("小任务", model, memory=NullMemory(), context_limit=100_000)
    assert model.summaries == 0
