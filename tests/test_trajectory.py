"""Trajectory eval：机械指标 + LLM 评委的解析。

评委本身不测（那是模型的判断），测的是**我们喂给它什么、以及解析失败时会不会假装成功**。
"""

import json

from mini_agent import trajectory as tj
from mini_agent.model import Reply, ScriptedModel, assistant_says
from mini_agent.state import AgentState, ToolTrace


def _state(**kw):
    st = AgentState(goal="研究 X", **{k: v for k, v in kw.items() if k != "trace"})
    st.trace = kw.get("trace", [])
    return st


def test_counts_duplicate_and_failed_tool_calls():
    st = _state(
        step=3,
        answer="结论：X 是 Y。来源 https://a.com 和 https://b.com",
        status="done",
        trace=[
            ToolTrace(1, "search_web", '{"q":"x"}', "结果"),
            ToolTrace(2, "search_web", '{"q":"x"}', "结果"),  # 一模一样，纯浪费
            ToolTrace(2, "search_web", '{"q":"y"}', "ERROR: 检索失败"),
            ToolTrace(2, "calculate", "{}", "未执行", executed=False, skip_reason="throttled"),
            ToolTrace(3, "send_email", "{}", "未执行", executed=False, skip_reason="denied"),
        ],
    )
    m = tj.mechanical(st)
    assert m["tool_calls"] == 3 and m["throttled"] == 1
    assert m["denied"] == 1  # 被拒绝和被限流要分开数
    assert m["duplicate_tool_calls"] == 1
    assert m["failed_tool_calls"] == 1
    assert m["answer_citations"] == 2
    assert m["delivered"] is True and m["asks_user_back"] is False


def test_detects_a_run_that_asks_the_user_back():
    """第一次实测失败的形态：交回一份请示而不是结论 —— 指标必须抓得到。"""
    st = _state(answer="我先做了检索……请确认是否同意我再跑 2-3 组检索。", status="done")
    assert tj.mechanical(st)["asks_user_back"] is True


def test_detects_empty_delivery():
    st = _state(answer="（未得出最终答案，停止原因：max_steps）", status="max_steps")
    assert tj.mechanical(st)["delivered"] is False


def test_judge_parses_json_even_with_surrounding_text():
    st = _state(answer="答案", trace=[ToolTrace(1, "calculate", "{}", "2")])
    model = ScriptedModel(
        [assistant_says('```json\n{"outcome":4,"grounding":3,"efficiency":5,"honesty":5,'
                        '"verdict":"还行","worst":"引用太少"}\n```')]
    )
    out = tj.judge(st, model)
    assert out["outcome"] == 4 and out["worst"] == "引用太少"
    assert "error" not in out


def test_judge_reports_failure_instead_of_faking_a_score():
    st = _state(answer="答案")
    out = tj.judge(st, ScriptedModel([assistant_says("我觉得挺好的")]))
    assert "error" in out and "outcome" not in out  # 解析不了就说解析不了


def test_rendered_trajectory_shows_tools_and_answer():
    st = _state(
        step=2,
        answer="最终结论",
        status="done",
        trace=[ToolTrace(1, "search_web", '{"q":"x"}', "很长的结果" * 200)],
    )
    text = tj.render_trajectory(st)
    assert "search_web" in text and "最终结论" in text
    assert len(text) < 1500  # 结果被截断，别把评委的上下文撑爆


def test_flags_citations_that_no_tool_ever_returned():
    """编造引用的确定性检测：答案里的链接必须在某次工具结果里出现过。"""
    st = _state(
        answer="见 https://real.com/a 和 https://made-up.com/b",
        trace=[ToolTrace(1, "search_web", "{}", "1. 标题 https://real.com/a 摘要")],
    )
    m = tj.mechanical(st)
    assert m["answer_citations"] == 2
    assert m["unsupported_citations"] == 1
