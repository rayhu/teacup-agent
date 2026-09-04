"""Stage B's free half: the policy matrix, run offline.

Live, `bench.py` answers "where does the small model break". Offline it can still pin
the parts that are deterministic — that the matrix is sparse where it was declared
sparse, that a crashed cell produces a row instead of killing the table, and above all
that the **message protocol survives a run whose roles resolve to different backends**.

That last one is the claim Stage A rests on. Routing is fixed for the run, so `main` is
one model start to finish; there are exactly two places another model's output enters
the main message list:

* a compaction summary — a plain `{"role": "system", ...}` entry;
* a subagent's answer — a tool-result string.

Both are shape-neutral, which is *why* mixing a `chat` profile with a `responses` one is
safe today and why mid-run switching (roadmap #21 Stage C) is not.
"""

import pytest

from teacup_agent import bench, loop, persist, routing
from teacup_agent.evals import tool_results_follow_their_call
from teacup_agent.memory import NullMemory
from teacup_agent.model import Reply, ScriptedModel, ToolCall, assistant_calls, assistant_says


# --- the matrix ---------------------------------------------------------------


def _policies():
    return [bench.Policy("all-big", {}), bench.Policy("all-small", {"main": "small"})]


def _goal(name, policies=None, **kw):
    return bench.Goal(name=name, text=f"do {name}", policies=policies, plan=False, **kw)


def test_a_goal_runs_only_under_the_policies_it_names():
    """Sparse on purpose: a policy that differs only in `compact` tells you nothing on a
    goal that never compacts, and a live cell costs money."""
    goals = [_goal("everything"), _goal("narrow", policies=("all-big",))]
    got = [(g.name, p.name) for g, p in bench.cells(goals, _policies())]
    assert got == [
        ("everything", "all-big"),
        ("everything", "all-small"),
        ("narrow", "all-big"),
    ]


def test_a_goal_naming_an_unknown_policy_is_an_error():
    with pytest.raises(ValueError, match="unknown policy"):
        bench.cells([_goal("x", policies=("typo",))], _policies())


def test_a_policy_may_not_pin_the_judge():
    """Or the quality column varies with the thing being measured."""
    with pytest.raises(ValueError, match="may not set the judge role"):
        bench.Policy("bad", {"judge": "small"})


def _scripted_router(policy, script=None):
    """A fresh scripted model per profile, per cell — scripts are consumed."""
    return routing.Router(
        lambda name: ScriptedModel(list(script or [assistant_says(f"answer from {name}")])),
        policy.roles,
        default="big",
    )


def test_every_cell_reports_its_roles_metrics_and_protocol():
    matrix = bench.cells([_goal("one"), _goal("two", policies=("all-big",))], _policies())
    reports = bench.run_matrix(matrix, _scripted_router)
    assert [(r["goal"], r["policy"]) for r in reports] == [
        ("one", "all-big"),
        ("one", "all-small"),
        ("two", "all-big"),
    ]
    assert all(r["status"] == "done" for r in reports)
    assert reports[0]["roles"]["main"] == "big" and reports[1]["roles"]["main"] == "small"
    assert "judge" not in reports[0]["roles"]  # pinned outside the policy
    assert all("mechanical" in r and "judged" not in r for r in reports)
    assert "answer from small" in reports[1]["answer"]
    # Which roles the policy moves, and which ones actually ran (see vacuity_warnings).
    assert reports[0]["routed_roles"] == [] and reports[1]["routed_roles"] == ["main"]
    assert "main" in reports[1]["fired_roles"]
    assert "compact" not in reports[1]["fired_roles"]  # nothing compacted here

    table = bench.format_table(reports)
    assert "all-small" in table and "n=1 per cell" in table


def test_a_crashed_cell_is_a_row_not_a_dead_matrix():
    """A matrix that dies on cell 4 of 9 has spent the money and produced no table."""

    def explode(policy):
        raise RuntimeError("no such profile")

    reports = bench.run_matrix(bench.cells([_goal("one")], [bench.Policy("all-big", {})]), explode)
    assert reports[0]["error"].startswith("RuntimeError")
    assert "ERROR" in bench.format_table(reports)


def test_the_table_says_when_a_cell_proved_nothing():
    """A policy that moves only roles which never ran produced a copy of the baseline,
    at full price. Two numbers that match then mean nothing, and the table says so."""
    base = {"goal": "long", "status": "done", "steps": 3, "compactions": 0,
            "cost": 0.01, "mechanical": _empty_metrics()}
    vacuous = {**base, "policy": "compact-small", "routed_roles": ["compact"],
               "fired_roles": ["main", "plan"]}
    real = {**base, "policy": "compact-small", "compactions": 2,
            "routed_roles": ["compact"], "fired_roles": ["main", "plan", "compact"]}

    assert "identical to the baseline" in bench.format_table([vacuous])
    assert bench.vacuity_warnings([real]) == []
    assert bench.vacuity_warnings([{**base, "policy": "all-big", "routed_roles": []}]) == []


def test_the_table_says_when_one_cell_delegated_and_another_did_not():
    """Whether to call `delegate` is the model's choice, not the policy's, and one child
    run moves the cost more than any routing decision — so that row is not comparable."""
    base = {"goal": "research", "status": "done", "steps": 3, "compactions": 0,
            "cost": 0.01, "mechanical": _empty_metrics(), "routed_roles": []}
    rows = [
        {**base, "policy": "all-big", "fired_roles": ["main", "subagent"]},
        {**base, "policy": "all-small", "fired_roles": ["main"]},
    ]
    assert any("not comparable" in w for w in bench.vacuity_warnings(rows))
    # Both delegating, or neither, is comparable again.
    assert bench.vacuity_warnings([rows[0], {**rows[1], "fired_roles": ["main", "subagent"]}]) == []


def _empty_metrics():
    return {
        "tool_calls": 0, "failed_tool_calls": 0, "duplicate_tool_calls": 0,
        "pending_todos": 0, "unsupported_citations": 0, "delivered": True,
    }


# --- the protocol, across two backend shapes ----------------------------------


class ScriptedResponses:
    """Scripted output in the Responses shape (reasoning items, call_id,
    function_call_output), so a cell can mix the two backends."""

    def __init__(self, script):
        self.script = list(script)
        self.model = "fake-responses"

    def complete(self, messages, tools):
        return self.script.pop(0) if self.script else Reply(
            items=[{"type": "message", "role": "assistant", "content": "done"}], text="done"
        )

    def tool_result_item(self, call, result):
        return {"type": "function_call_output", "call_id": call.id, "output": result}


def _responses_call(call_id, name, arguments):
    return Reply(
        items=[
            {"type": "reasoning", "id": f"rs_{call_id}"},
            {"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments},
        ],
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
    )


def test_a_chat_subagent_under_a_responses_parent_keeps_both_lists_consistent(tmp_path):
    """The only thing that crosses between them is the child's answer, as a tool-result
    string. Both message lists must stay internally consistent in their own shape."""
    parent = ScriptedResponses(
        [
            _responses_call("fc_1", "delegate", '{"task": "read the long thing"}'),
            Reply(items=[{"type": "message", "role": "assistant", "content": "parent answer"}],
                  text="parent answer"),
        ]
    )
    child = ScriptedModel(  # chat-shaped: role="tool" results
        [assistant_calls([("calculate", {"expression": "6*7"})]), assistant_says("42")]
    )
    router = routing.from_models({"big": parent, "small": child}, {"subagent": "small"}, "big")

    state = loop.run(
        goal="delegate the reading",
        model=router,
        memory=NullMemory(),
        max_steps=4,
        subagents=True,
        run_dir=tmp_path,
    )

    assert state.status == "done" and state.subagent_runs == 1
    # Parent: Responses shape throughout, and the child's answer arrived as a plain string.
    assert tool_results_follow_their_call(state)
    outputs = [m for m in state.messages if m.get("type") == "function_call_output"]
    assert len(outputs) == 1 and "42" in outputs[0]["output"]
    assert not any(m.get("role") == "tool" for m in state.messages)  # no chat shape leaked in

    # Child: chat shape throughout, in its own message list under the parent's run dir.
    child_state = persist.load(tmp_path / "sub01")
    assert tool_results_follow_their_call(child_state)
    assert any(m.get("role") == "tool" for m in child_state.messages)
    assert not any(m.get("type") == "function_call_output" for m in child_state.messages)


def test_a_chat_compactor_under_a_responses_parent_produces_a_shape_neutral_summary():
    """The compaction summary re-enters the main list as a role="system" entry, which
    both backends accept — the second and last place another model's output crosses."""
    parent = ScriptedResponses(
        [_responses_call(f"fc_{i}", "calculate", '{"expression": "1+1"}') for i in range(4)]
    )
    compactor = ScriptedModel([])  # its summariser branch answers whatever it is asked
    router = routing.from_models(
        {"big": parent, "small": compactor}, {"compact": "small"}, "big"
    )

    state = loop.run(
        goal="compute repeatedly until the context is compacted",
        model=router,
        memory=NullMemory(),
        max_steps=5,
        context_limit=1,  # force compaction as soon as there is a safe cut point
        run_dir=None,
    )

    assert state.compactions >= 1
    summaries = [m for m in state.messages if "[context summary]" in str(m.get("content", ""))]
    assert summaries and summaries[0]["role"] == "system"
    assert tool_results_follow_their_call(state)
    assert state.spend_by_profile.get("small", 0) > 0  # the cheap model did the summarising
