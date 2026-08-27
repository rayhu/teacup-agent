"""Wire the cases from evals.py into pytest: one set of cases, two ways to run it."""

import pytest

from teacup_agent.evals import CASES, run_case


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_eval_case(case):
    ok, state = run_case(case)
    assert ok, f"final state: {state.snapshot()}"
