"""Trajectory eval: the mechanical metrics, and how the judge output is parsed.

The judge's opinions are not under test (those belong to the model). What is tested
is **what we feed it, and whether a parse failure is ever dressed up as a score**.
"""

import json

from teacup_agent import trajectory as tj
from teacup_agent.model import Reply, ScriptedModel, assistant_says
from teacup_agent.state import AgentState, ToolTrace


def _state(**kw):
    st = AgentState(goal="research X", **{k: v for k, v in kw.items() if k != "trace"})
    st.trace = kw.get("trace", [])
    return st


def test_counts_duplicate_and_failed_tool_calls():
    st = _state(
        step=3,
        answer="Conclusion: X is Y. Sources https://a.com and https://b.com",
        status="done",
        trace=[
            ToolTrace(1, "search_web", '{"q":"x"}', "results"),
            ToolTrace(2, "search_web", '{"q":"x"}', "results"),  # identical: pure waste
            ToolTrace(2, "search_web", '{"q":"y"}', "ERROR: search failed"),
            ToolTrace(2, "calculate", "{}", "not executed", executed=False, skip_reason="throttled"),
            ToolTrace(3, "send_email", "{}", "not executed", executed=False, skip_reason="denied"),
        ],
    )
    m = tj.mechanical(st)
    assert m["tool_calls"] == 3 and m["throttled"] == 1
    assert m["denied"] == 1  # denied and throttled are counted separately
    assert m["duplicate_tool_calls"] == 1
    assert m["failed_tool_calls"] == 1
    assert m["answer_citations"] == 2
    assert m["delivered"] is True and m["asks_user_back"] is False


def test_detects_a_run_that_asks_the_user_back():
    """The shape of the very first real failure: a request for permission instead of
    a conclusion. The metric has to catch it."""
    st = _state(answer="I ran a first round of searches. Please confirm whether I should run 2-3 more.", status="done")
    assert tj.mechanical(st)["asks_user_back"] is True


def test_detects_empty_delivery():
    st = _state(answer="(no final answer; stopped because: max_steps)", status="max_steps")
    assert tj.mechanical(st)["delivered"] is False


def test_judge_parses_json_even_with_surrounding_text():
    st = _state(answer="the answer", trace=[ToolTrace(1, "calculate", "{}", "2")])
    model = ScriptedModel(
        [assistant_says('```json\n{"outcome":4,"grounding":3,"efficiency":5,"honesty":5,'
                        '"verdict":"decent","worst":"too few citations"}\n```')]
    )
    out = tj.judge(st, model)
    assert out["outcome"] == 4 and out["worst"] == "too few citations"
    assert "error" not in out


def test_judge_reports_failure_instead_of_faking_a_score():
    st = _state(answer="the answer")
    out = tj.judge(st, ScriptedModel([assistant_says("looks fine to me")]))
    assert "error" in out and "outcome" not in out  # unparseable means unparseable


def test_rendered_trajectory_shows_tools_and_answer():
    st = _state(
        step=2,
        answer="final conclusion",
        status="done",
        trace=[ToolTrace(1, "search_web", '{"q":"x"}', "a very long result " * 200)],
    )
    text = tj.render_trajectory(st)
    assert "search_web" in text and "final conclusion" in text
    assert len(text) < 1500  # results are truncated so the judge context stays small


def test_flags_citations_that_no_tool_ever_returned():
    """Deterministic detector for invented citations: every link in the answer must
    have appeared in some tool result."""
    st = _state(
        answer="See https://real.com/a and https://made-up.com/b",
        trace=[ToolTrace(1, "search_web", "{}", "1. Title https://real.com/a snippet")],
    )
    m = tj.mechanical(st)
    assert m["answer_citations"] == 2
    assert m["unsupported_citations"] == 1


def test_flags_an_action_the_agent_never_attempted():
    """The failure this metric exists for: the goal said "email me", the agent
    researched well, never called the gated tool, and still reported done."""
    st = _state(
        answer="Here is the research and a draft email for you to send.",
        status="done",
        trace=[ToolTrace(1, "search_web", '{"q":"x"}', "results")],
    )
    st.goal = "research X and email me the result"
    assert tj.mechanical(st)["action_never_attempted"] is True


def test_a_denied_attempt_still_counts_as_attempted():
    """The model did its part; the human said no. That is not the same failure."""
    st = _state(
        answer="The email needs your approval.",
        status="done",
        trace=[
            ToolTrace(1, "search_web", '{"q":"x"}', "results"),
            ToolTrace(2, "send_email", "{}", "ERROR: ... was NOT executed",
                      executed=False, skip_reason="denied"),
        ],
    )
    st.goal = "research X and email me the result"
    assert tj.mechanical(st)["action_never_attempted"] is False


def test_a_pure_research_goal_is_not_flagged():
    st = _state(answer="Here is what I found.", status="done")
    st.goal = "research X for me"
    assert tj.mechanical(st)["action_never_attempted"] is False


def test_asking_after_a_denial_is_not_the_failure_mode():
    """Our own prompt tells the model to say what is left for the user once a gated
    call was denied. Only asking *without ever trying* deserves the flag."""
    st = _state(
        answer="I attempted the send and it was denied. Would you like me to retry?",
        status="done",
        trace=[ToolTrace(1, "send_email", "{}", "denied", executed=False, skip_reason="denied")],
    )
    st.goal = "research X and email me the result"
    m = tj.mechanical(st)
    assert m["asks_user_back"] is True  # the raw signal still fires
    assert m["asks_without_trying"] is False  # but this is the one that matters


def test_asking_without_trying_is_flagged():
    st = _state(answer="Would you like me to send it now?", status="done")
    st.goal = "research X and email me the result"
    assert tj.mechanical(st)["asks_without_trying"] is True
