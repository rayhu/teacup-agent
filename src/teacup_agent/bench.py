"""Bench — run the same work under different routing policies and compare.

`evals.py` asks "is the loop correct" with a scripted model, for free. This asks the
question routing (#21) actually turns on: **where does the small model break on your
workload**, and what does the split save. Those are not answerable from benchmark
scores or from taste; they are answerable from a table.

The shape is a sparse matrix. A *policy* is a named role->profile map — `all-big`,
`roles-split`, `all-small`, and one that differs from `all-big` in a single role, which
is how you isolate a question like "can the compactor be the small model". A *goal* is
one task, and it names the policies worth running it under: a policy that differs only
in `compact` tells you nothing on a run that never compacts, and a live run costs money.

Three rules this file exists to keep honest:

1. **The judge is pinned outside the policy.** If `all-small` were judged by the small
   model and `all-big` by the big one, the measurement would vary with the thing being
   measured. One judge profile for the whole matrix, and `judge` is rejected inside a
   policy's role map.
2. **`trajectory.mechanical()` is the column to trust** — free, deterministic, and
   model-independent. The judge's scores are a second opinion, and a weak one on long
   trajectories: decent at "did this go off the rails", poor at "is this subtly wrong".
3. **Say when a cell proved nothing.** A policy differs from the baseline in a set of
   roles; if none of those roles actually *ran* — no compaction happened, nothing was
   delegated — the cell is identical to the baseline by construction, and two numbers
   that match mean nothing. The table computes which roles fired and says so, rather
   than leaving it to be inferred.

Offline (scripted models, no key, no cost) it is a protocol check; live it is the
measurement. Run: `uv run python -m teacup_agent.bench --config agent.yaml`.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from teacup_agent import loop, reflect as reflect_mod, routing, trajectory
from teacup_agent.memory import NullMemory


@dataclass
class Policy:
    """A named role -> profile map. Roles left out fall back to the router's default."""

    name: str
    roles: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if "judge" in self.roles:
            raise ValueError(
                "a policy may not set the judge role: the judge is pinned for the whole "
                "matrix, or the quality column varies with the thing being measured"
            )


@dataclass
class Goal:
    """One task, and the policies worth running it under (None = all of them)."""

    name: str
    text: str
    policies: tuple[str, ...] | None = None
    max_steps: int = 8
    context_limit: int = 30_000
    plan: bool = True
    subagents: bool = False
    why: str = ""  # what this goal is here to find out


def default_policies(big: str, small: str) -> list[Policy]:
    """The three policies #21 named, plus one that differs from `all-big` in a single
    role — `compact` is the first question Stage B was written to answer, and it cannot
    be isolated by comparing against a policy that moves four roles at once."""
    return [
        Policy("all-big", {}),
        Policy("roles-split", {"reflect": small, "subagent": small}),
        Policy("compact-small", {"compact": small}),
        Policy("all-small", {r: small for r in routing.ROLES if r != "judge"}),
    ]


# The three goals, chosen to span the predictors routing is supposed to key on:
# checkable success, spec completeness, and context noise.
DEFAULT_GOALS = [
    Goal(
        name="verifiable",
        text=(
            "Compute the compound annual growth rate implied by 1200 growing to 3400 "
            "over 7 years, to two decimal places, and state the formula you used."
        ),
        policies=("all-big", "roles-split", "all-small"),
        max_steps=6,
        plan=True,
        why="complete spec, arithmetic a tool checks — the small model should hold here",
    ),
    Goal(
        name="underspecified",
        text=(
            "Find out how NVIDIA is currently positioned for *inference* workloads, as "
            "opposed to training. Say what you could confirm, from which sources, and "
            "what remains unverified."
        ),
        policies=("all-big", "roles-split", "all-small"),
        max_steps=8,
        subagents=True,  # or `roles-split` differs from `all-big` in nothing that runs
        why="ambiguous scope and source grading — judgment, where the gap should show",
    ),
    Goal(
        name="long-context",
        text=(
            "Research the main approaches to serving large language models cheaply in "
            "2026 (batching, quantization, caching, hardware), then give a short "
            "comparison with the trade-off each one makes."
        ),
        policies=("all-big", "compact-small", "all-small"),
        max_steps=8,
        subagents=True,
        context_limit=2_000,  # low on purpose: the compact cell means nothing if it never fires
        why="forces compaction — the `compact` question, and context noise",
    ),
]


def cells(goals: list[Goal], policies: list[Policy]) -> list[tuple[Goal, Policy]]:
    """The matrix, sparse: a goal runs only under the policies it names."""
    names = {p.name for p in policies}
    for goal in goals:
        for wanted in goal.policies or ():
            if wanted not in names:
                raise ValueError(f"goal {goal.name!r} names unknown policy {wanted!r}")
    return [
        (goal, policy)
        for goal in goals
        for policy in policies
        if goal.policies is None or policy.name in goal.policies
    ]


def roles_fired(state: Any, goal: Goal, metrics: dict[str, Any], reflect: bool) -> list[str]:
    """Which roles actually made a call in this run.

    Read off the state rather than instrumented in `Router`, because a subagent's turns
    run on a *derived* router the parent never sees, and because a fact recovered from
    the saved state can be re-derived later from `state.json` alone.
    """
    fired = ["main"]
    if goal.plan and state.todo:
        fired.append("plan")
    if state.compactions:
        fired.append("compact")
    if reflect and any(reflect_mod.should_reflect(state, metrics)):
        fired.append("reflect")
    if state.subagent_runs:
        fired.append("subagent")
    return fired


def run_cell(
    goal: Goal,
    policy: Policy,
    make_router: Callable[[Policy], routing.Router],
    *,
    budget: float = 0.10,
    judge: Any = None,
    run_dir: pathlib.Path | None = None,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """One goal under one policy. A crashed cell is recorded, never raised: a matrix
    that dies on cell 4 of 9 has spent the money and produced no table."""
    started = time.monotonic()
    router = None
    try:
        router = make_router(policy)  # inside the guard: a bad profile is a row, not a crash
        state = loop.run(
            goal=goal.text,
            model=router,
            memory=NullMemory(),  # a shared memory would leak one cell's findings into the next
            max_steps=goal.max_steps,
            budget=budget,
            context_limit=goal.context_limit,
            plan=goal.plan,
            subagents=goal.subagents,
            reflect=True,  # a real run reflects, and `reflect` is a role a policy may route
            run_dir=run_dir,
            on_event=on_event,
        )
    except Exception as e:  # noqa: BLE001 — the report is the deliverable
        return {
            "goal": goal.name,
            "policy": policy.name,
            "error": f"{type(e).__name__}: {e}",
            "elapsed_s": round(time.monotonic() - started, 1),
        }

    report: dict[str, Any] = trajectory.score(state, judge)
    report.update(
        goal=goal.name,
        policy=policy.name,
        roles={r: p for r, p in router.profiles().items() if r != "judge"},
        # The roles this policy moves, and the ones that actually ran: a policy whose
        # differing roles never fired produced a copy of the baseline, at full price.
        routed_roles=sorted(policy.roles),
        fired_roles=roles_fired(state, goal, report["mechanical"], reflect=True),
        status=state.status,
        steps=state.step,
        compactions=state.compactions,
        cost=round(budget - state.remaining_budget, 6),
        spend=dict(state.spend_by_profile),
        answer=state.answer,
        task=goal.text,
    )
    return report


def run_matrix(
    matrix: list[tuple[Goal, Policy]],
    make_router: Callable[[Policy], routing.Router],
    *,
    budget: float = 0.10,
    judge: Any = None,
    run_dir: pathlib.Path | None = None,
    on_cell: Callable[[Goal, Policy], None] | None = None,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    reports = []
    for goal, policy in matrix:
        if on_cell:
            on_cell(goal, policy)
        reports.append(
            run_cell(
                goal,
                policy,
                make_router,
                budget=budget,
                judge=judge,
                run_dir=(run_dir / f"{goal.name}--{policy.name}") if run_dir else None,
                on_event=on_event,
            )
        )
    return reports


# --- the table ----------------------------------------------------------------

PREAMBLE = """n=1 per cell: the cost and compaction columns are reliable, the quality
columns are anecdotes. LLM run-to-run variance will swamp a 30% difference, so read
`fail`/`dup`/`pend`/`unsup` as a failure **count** to look at, not as a score.

These are costs *of this comparison*, not of the same task in a real run: the bench uses
`NullMemory`, so no recalled facts sit in the prefix, and `$` is the run only — a judge
call happens outside `loop.run` and outside its budget ceiling."""

# `cites` sits next to `unsup` on purpose: `unsupported_citations` is a numerator with
# no denominator, and a cell that cited nothing scores a perfect 0 on it. Read together
# or the pair rewards vagueness — a lesson from the first live run that was written down
# before it was implemented.
_HEADERS = (
    ("goal", 14),
    ("policy", 13),
    ("status", 13),  # "out_of_budget" is 13 characters and used to shift the whole row
    ("steps", 5),
    ("comp", 4),
    ("$", 8),
    ("calls", 5),
    ("fail", 4),
    ("dup", 3),
    ("pend", 4),
    ("cites", 5),
    ("unsup", 5),
    ("deliv", 5),
    ("judge", 22),
)


def _row(report: dict[str, Any]) -> str:
    if "error" in report:
        cells_ = [report["goal"], report["policy"], "ERROR"] + [""] * 10 + [report["error"][:22]]
    else:
        m = report["mechanical"]
        j = report.get("judged") or {}
        judged = (
            f"out {j['outcome']} grd {j['grounding']} hon {j['honesty']}"
            if "outcome" in j
            else (f"error: {j['error'][:15]}" if "error" in j else "-")
        )
        cells_ = [
            report["goal"],
            report["policy"],
            report["status"],
            report["steps"],
            report["compactions"],
            f"{report['cost']:.4f}",
            m["tool_calls"],
            m["failed_tool_calls"],
            m["duplicate_tool_calls"],
            m["pending_todos"],
            m["answer_citations"],
            m["unsupported_citations"],
            "yes" if m["delivered"] else "NO",
            judged,
        ]
    return "  ".join(f"{str(v):<{w}}" for v, (_, w) in zip(cells_, _HEADERS)).rstrip()


def format_table(reports: list[dict[str, Any]]) -> str:
    head = "  ".join(f"{h:<{w}}" for h, w in _HEADERS).rstrip()
    lines = [PREAMBLE, "", head, "-" * len(head)]
    lines += [_row(r) for r in reports]

    lines += vacuity_warnings(reports)
    return "\n".join(lines)


def vacuity_warnings(reports: list[dict[str, Any]]) -> list[str]:
    """Two ways a row of the table can mean less than it looks like.

    A cell whose policy moves only roles that never ran is a copy of the baseline. And a
    goal where one cell delegated and another did not is not comparable on cost at all:
    whether to call `delegate` is the model's choice, and one subagent run moves the
    number far more than any routing decision does.
    """
    out = []
    by_goal: dict[str, set[bool]] = {}
    for r in reports:
        if "error" not in r and r.get("fired_roles") is not None:
            by_goal.setdefault(r["goal"], set()).add("subagent" in r["fired_roles"])
    for goal, delegated in by_goal.items():
        if len(delegated) > 1:
            out.append(
                f"WARNING: in {goal}, some cells delegated to a subagent and others did "
                "not. That is the model's choice, not the policy's, and it moves cost "
                "more than routing does — the $ column is not comparable across that row."
            )
    for r in reports:
        routed = set(r.get("routed_roles") or ())
        if not routed or "error" in r:
            continue  # the baseline policy moves nothing by definition
        fired = routed & set(r.get("fired_roles") or ())
        if not fired:
            out.append(
                f"WARNING: {r['goal']} x {r['policy']} moved {sorted(routed)}, and none of "
                "those roles ran — identical to the baseline by construction, so any "
                "difference in its numbers is noise."
            )
    return ["", *out] if out else []


# --- command line -------------------------------------------------------------


def _confirm(cell_count: int, budget: float, models: str, judge: str | None) -> bool:
    total = cell_count * budget
    print(f"\nAbout to run {cell_count} live agent runs ({models}).")
    print(f"Per run: --budget {budget:.2f}, which is checked **between turns** — a single")
    print("expensive turn can overshoot it. Measured overshoots on a $0.10 ceiling so far:")
    print("+51%, +70%, +29%. Treat the total below as an estimate, not a bound.")
    print(f"  about ${total:.2f} across the matrix, and plan for half again as much.")
    if judge:
        print(
            f"  PLUS one {judge} judge call per cell. Those happen outside loop.run and "
            "are NOT covered by --budget:\n"
            "  roughly 2-3k input tokens over the rendered trajectory, so small, but "
            "uncapped."
        )
    try:
        return input("Continue? [y/N] ").strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compare routing policies on the same goals")
    p.add_argument("--config", required=True, help="agent.yaml holding the model profiles")
    p.add_argument("--big", default="big", help="models.profiles entry for the strong model")
    p.add_argument("--small", default="small", help="models.profiles entry for the cheap model")
    p.add_argument("--policies", default=None, help="comma-separated subset of the policies to run")
    p.add_argument("--goals", default=None, help="comma-separated subset of the goals to run")
    p.add_argument("--budget", type=float, default=0.10, help="hard spending ceiling per run")
    p.add_argument(
        "--judge",
        action="store_true",
        help="add the LLM judge (extra calls, extra money). It runs on --judge-profile "
        "for every cell, never on the policy's own model",
    )
    p.add_argument("--judge-profile", default=None, help="judge model profile (default: --big)")
    p.add_argument("--run-dir", default=None, help="directory for the runs (default runs/bench-<timestamp>)")
    p.add_argument("--out", default=None, help="write the full report as JSON")
    p.add_argument("--dry-run", action="store_true", help="print the matrix and the ceiling, run nothing")
    p.add_argument("-y", "--yes", action="store_true", help="skip the spending confirmation")
    args = p.parse_args(argv)

    from dotenv import load_dotenv

    from teacup_agent import agent_config

    load_dotenv()
    cfg = agent_config.load(args.config)
    for name in (args.big, args.small):
        if name not in cfg.models:
            raise SystemExit(f"profile {name!r} is not in {args.config}'s models.profiles")
    os.environ["TEACUP_AGENT_SEARCH"] = cfg.runtime.search

    policies = default_policies(args.big, args.small)
    goals = DEFAULT_GOALS
    if args.goals:
        wanted = args.goals.split(",")
        goals = [g for g in goals if g.name in wanted]
        if not goals:
            raise SystemExit(f"--goals {args.goals} matched none of {[g.name for g in DEFAULT_GOALS]}")

    # The full matrix is validated first and *then* narrowed: --policies picks a slice to
    # run now, and must not turn a goal's own (valid) declaration into an unknown-policy
    # error just because this invocation left that policy out.
    matrix = cells(goals, policies)
    if args.policies:
        wanted = set(args.policies.split(","))
        if unknown := wanted - {p.name for p in policies}:
            raise SystemExit(f"--policies names unknown {sorted(unknown)}; known: {[p.name for p in policies]}")
        matrix = [(g, p) for g, p in matrix if p.name in wanted]
    if not matrix:
        raise SystemExit("nothing to run: --goals and --policies do not intersect")
    print(f"{len(matrix)} cells ({args.big} = strong, {args.small} = cheap):")
    for goal, policy in matrix:
        routed = ", ".join(f"{r}={p}" for r, p in policy.roles.items()) or "(everything on the default)"
        print(f"  {goal.name:<14} x {policy.name:<13} {routed}")
    if args.dry_run:
        print(f"\nDry run: nothing executed. Ceiling would be ${len(matrix) * args.budget:.2f}.")
        return 0
    if not args.yes and not _confirm(
        len(matrix), args.budget, f"{args.big} / {args.small}", (args.judge_profile or args.big) if args.judge else None
    ):
        print("Nothing ran.")
        return 1

    # One instance per profile for the whole matrix, so nine runs do not open nine
    # clients; a fresh Router per cell, because the role map is what changes.
    instances: dict[str, Any] = {}

    def make_router(policy: Policy) -> routing.Router:
        return routing.Router(
            lambda name: agent_config.build_model(cfg.models[name]),
            policy.roles,
            default=args.big,
            instances=instances,
        )

    judge_model = None
    if args.judge:
        judge_model = agent_config.build_model(cfg.models[args.judge_profile or args.big])

    run_dir = pathlib.Path(args.run_dir or f"runs/bench-{time.strftime('%Y%m%d-%H%M%S')}")
    reports = run_matrix(
        matrix,
        make_router,
        budget=args.budget,
        judge=judge_model,
        run_dir=run_dir,
        on_cell=lambda g, pol: print(f"\n--- {g.name} x {pol.name} ---", flush=True),
    )

    print("\n" + format_table(reports))
    print(f"\nTotal spent: ${sum(r.get('cost', 0.0) for r in reports):.4f}. Runs under {run_dir}/")
    out = pathlib.Path(args.out) if args.out else run_dir / "bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Full report: {out}")
    return 0 if all("error" not in r for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
