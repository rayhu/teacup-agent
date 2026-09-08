"""Wire the cases from evals.py into pytest: one set of cases, two ways to run it."""

import pytest

from teacup_agent.evals import CASES, run_case


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_eval_case(case):
    ok, state = run_case(case)
    assert ok, f"final state: {state.snapshot()}"


def test_evals_force_offline_search_even_when_the_environment_asks_for_hosted(monkeypatch):
    """AGENTS.md advertises `python -m teacup_agent.evals` as free, and one case really
    does call search_web. setdefault left an exported TEACUP_AGENT_SEARCH=hosted in
    place, so the free health check would bill for anyone with a key exported."""
    import os

    from teacup_agent import evals

    monkeypatch.setenv("TEACUP_AGENT_SEARCH", "hosted")
    evals.run_case(evals.CASES[0])
    assert os.environ["TEACUP_AGENT_SEARCH"] == "offline"
