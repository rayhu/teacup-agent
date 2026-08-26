"""把 evals.py 里的用例接到 pytest 上，一份用例两种跑法。"""

import pytest

from mini_agent.evals import CASES, run_case


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_eval_case(case):
    ok, state = run_case(case)
    assert ok, f"终态: {state.snapshot()}"
