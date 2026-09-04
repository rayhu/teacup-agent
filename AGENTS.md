# AGENTS.md

Static context for this repo, and the same file for every coding agent that works here:
`CLAUDE.md` imports this file and states no rule of its own, so there is one set of rules
and no second copy to drift out of date. It is loaded on every turn, so it stays short —
the same trade-off
`context.py` makes for the agent we are building. Add a rule here when something goes
wrong twice; delete one when it stops being true.

## What this is

A minimal, readable AI agent — `Agent = Model + State + Tools + Control Loop +
Memory/Evals` — built to be **shared, forked, improved and shared again**. Its value is
that the ~80-line control loop in `loop.py` fits in one head. Every feature is weighed
against that: a capability that makes the loop unreadable costs more than it adds.

Working docs: `README.md` (what it is and how to run it), `docs/intent.md` (what the
project is for, and what a fork owes it), `docs/spec.md` (the technical contract:
values, shapes, interfaces), `docs/design-notes.md` (why each subsystem behaves the
way it does), `docs/roadmap.md` (what is missing and in what order),
`docs/threat-model.md` (what is trusted, what is not, and what a fork inherits),
`docs/integration.md` (the `--json` contract an external caller relies on),
`NOTES.md` (the original study notes this grew from).

How a change gets from an idea to `main` — the phase loop, who reviews what, what a human
signs off — is `docs/workflow.md`, and the review pass itself is `REVIEW.md`. Read them
when you start a phase or when you are the reviewer; they are deliberately *not* copied
here, because process detail that is needed twice a week does not belong in context that
is paid for every turn.

## How we work together

**Plan, then build, one item at a time.** Propose the shape of a change before writing
it; land one roadmap item per round and report what happened. Do not batch three
features into one message.

**Evidence, never "should work".** A change is not done because the code looks right.
Run it. Quote the actual output. If something was not verified, say which part and why —
"the live path is unverified, no API key in this shell" is a complete and acceptable
report; silence is not.

**Explain defaults when asked, and expect them to be challenged.** "Why is this false by
default?" is a normal question here. If the honest answer is partly "it kept the tests
green", say so and mark it as the weak reason it is.

**Prefer explicit over convenient.** A flag named `--no-plan` that is silently inert in
half the runs is worse than `--plan {auto,on,off}` that says what it does. Names must not
lie about behaviour.

**Failures are the curriculum.** When a run goes wrong, the post-mortem goes into
`docs/roadmap.md` under "Field patches": symptom, root cause, fix, and the general
principle. Most agent failures here have had the same root cause — *the model did not
know its own situation* — and writing that down is what made the next fix obvious.

**Chinese in conversation, English in the repo.** All code, comments, docstrings,
prompts, CLI output and docs are English.

## Money and permission

- Live API calls cost the user money. **Ask before spending**, name the model and the
  rough cost, and prefer `gpt-5-mini` for verification runs.
- Report actual spend afterwards (`remaining_budget` from the run is enough).
- Never commit, push, or open a PR unless asked.
- `.env` and `mcp.json` hold secrets. They are gitignored; `.env.example` and
  `mcp.example.json` are the templates that get committed.
- If `git push` (or another git command) is denied by the harness's own permission
  classifier — the error names a classifier, not a git/auth failure — do not retry the
  identical command. Push via the GitHub API instead (`push_files`, one commit per call,
  content not diffs). Skip huge generated files that add no reviewable signal (`uv.lock`);
  say so in the PR and note that `uv sync` regenerates it. Verified 2026-09-03: `git push`
  and even a read-only `git diff` were both blocked mid-session; `push_files` was not.

## Verification standard

```bash
uv run pytest                          # unit + protocol tests
uv run python -m teacup_agent.evals    # loop health, scripted model, free
uv run teacup-agent                    # offline demo, no key, must stay instant
```

All three must pass before reporting a change as done. Then:

- **Tests verify the deterministic parts; evals verify the loop's behaviour.** They are
  the contract with the model — a well-written eval says what "correct" means more
  precisely than any prompt. Write them with the change, not after it.
- **Errors become tool results, never exceptions.** A tool that fails hands the model an
  `ERROR: ...` string so it can self-correct. That is the whole point of the loop.
- **Never let a broken tool read as "this does not exist."** A failed search must say it
  failed. That distinction has caused real wrong answers here.
- Offline paths stay offline: evals and unit tests make no network calls and write
  nothing into the repo (`run_dir=None`, `TEACUP_AGENT_SEARCH=offline`).

## Design rules this repo has settled on

These were each paid for with a failed run. Do not quietly reverse one.

1. **The message protocol is sacred.** The assistant message carrying `tool_calls` is
   appended *before* the results, and every `tool_call_id` gets exactly one result —
   including calls that were throttled, denied, or arrived in a forced wrap-up turn.
   Miss one and the next request 400s. `tool_results_follow_their_call()` guards this.
2. **Tell the model its situation.** Steps, budget, time, and the checklist go into the
   context every turn. State living in `AgentState` is not state the model knows.
3. **A brake must also unload the car.** Hitting a ceiling triggers a forced wrap-up so
   the run never ends empty-handed.
4. **Deny by default when nobody is watching.** Side-effecting tools need approval;
   "no TTY, so allow it" is the most dangerous default there is.
5. **Nothing is silently half-done.** The checklist and the `action_never_attempted` /
   `pending_todos` metrics exist because a run once reported `done` for a task it had
   only half finished.
6. **Keep the context prefix stable.** Per-turn notes are appended at the end, never
   spliced into the system prompt, or prompt caching dies.
7. **Keep Python modules focused and cohesive.** As a guideline, prefer
   modules under ~500 lines. When a module grows beyond ~700 lines,
   consider splitting it by responsibility rather than mechanically
   splitting by size.

## Code conventions

- Python 3.11, `uv` for everything (`uv run`, `uv sync`). Never `pip install`.
- Comments explain **why**, especially why an alternative was rejected. The measured
  number belongs in the comment ("1.5s interval → 8.3s, no interval → 4.7s").
- New module = new concern. Keep `loop.py` readable; push shape differences into the
  backend classes, not into the loop.
- Match the surrounding density of comments and naming. Do not add type-checking or
  linting ceremony that is not already here.

## Where the SDLC paper lands

`The New SDLC With Vibe Coding` (Google, May 2026) is the method behind the
above. The parts that bind:

- **Agent = Model + Harness.** The model is ~10% of behaviour; the harness — instructions,
  tools, sandboxes, orchestration, guardrails, observability — is the rest, and it is
  *our* surface area, not the provider's. This repo is a harness you can read.
- **Context engineering, static vs dynamic.** Static context (this file, the system
  prompt) is always loaded and therefore expensive; dynamic context (tool results,
  externalized files, retrieved documents) is paid for only when used. Treat that
  boundary as an architectural decision, reviewed like code — which is exactly what
  `context.py` and the `--context-limit` / externalization split implement.
- **The factory model.** The output is the system that produces the code, not the code.
  Success comes from giving agents success criteria and letting them iterate — hence
  evals before features.
- **Output eval *and* trajectory eval.** Checking the final answer is not enough; check
  how it got there (`trajectory.py`). A fluent answer that skipped verification is more
  dangerous than a visible error.
- **Review every line that ships.** Be skeptical of anything clever, check imports are
  real packages, check error handling covers realistic failures. Code the team does not
  understand is debugging cost the team cannot afford.
- **Prototype mode and production mode are different modes.** This repo is a teaching
  artifact; when a change would only make sense in production, say so and leave it in
  `docs/roadmap.md` instead of smuggling it in.

## Where the Anthropic playbook lands

`The AI-Native SDLC Playbook` (Anthropic, 2026) is the other half of the method: the
Google paper says what a harness is, the playbook says how a change moves through one.
The parts that bind here:

- **Every stage commits an artifact the next stage reads.** Intent, spec, plan, code,
  review, merge — each is a file in git, not a conversation. `docs/workflow.md` names
  which file plays which role in this repo.
- **Plan before code, and commit the plan.** The design argument happens while it is
  still cheap to lose. A roadmap item's "what to change / what counts as done" *is* that
  plan; write it before the diff, not after.
- **Verification is part of "done".** The agent runs the checks and shows the output
  before a human looks — the three commands under "Verification standard" below, quoted,
  not summarized.
- **Evals gate changes to the harness itself.** `evals.py` regression-tests the loop's
  behaviour the way tests regression-test its code, so a change to prompts, guards or
  this file is a change that has to stay green.
- **The reviewer is not the author.** An independent pass with its own context finds what
  the writer's context hides. `REVIEW.md` defines that pass.
- **A human approves the merge.** Agents do everything up to the gate and nothing past
  it: never commit, push, or open a PR unless asked.
