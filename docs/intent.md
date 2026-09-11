# Intent

`README.md` says what this is. `CONTRIBUTING.md` says where the seams are. This file says
what the project is **for**, and what a fork owes it — the criteria a change can actually
fail, so "did this stay true to the thing?" is a question with an answer.

It is also the first artifact in the chain `docs/workflow.md` runs on — its table of
artifacts is the authority, and this is the part of it that matters here:

```
docs/intent.md  ->  docs/spec.md  ->  docs/roadmap.md item  ->  code + evals  ->  review  ->  human merges
```

That ordering is a working constraint, not a diagram. **`docs/spec.md` is downstream of
this file.** Every section of the spec exists to contract a capability named in §3 below,
and every capability in §3 names the spec section that contracts it. That is what
"machine-actionable" means here — not that a tool can parse the prose, but that an agent
given this file and the repo can regenerate the spec's table of contents, and can answer
three questions without asking a human:

- **Coverage.** Does every row in §3 name a spec section, and does every `## n.` in
  `docs/spec.md` appear in §3? A row with no section is unspecified. A section with no row
  is a capability no stated intent asked for — which is a scope question, not a bug.
- **Currency.** Does every criterion in §6 still print the number written beside it? A
  number that has moved is either the change or the criterion, and the change's author has
  to say which.
- **Guards.** Does every invariant in §5 still name an eval case or test that exists? An
  invariant nothing executes is a preference.

```bash
# coverage, both directions: what §3 claims a spec section for, and what the spec has
sed -n '/^## 3\./,/^## 4\./p' docs/intent.md | grep -o '§[0-9]\+' | sort -uV
grep -o '^## [0-9]\+' docs/spec.md
```

## 1. The intent in one sentence

Build the smallest agent harness that is still honest about what a real model on a real
network does to you — and make it cheap enough to fork, change and republish that the
harness, not this repository, is what spreads.

That is a deliberate ordering. The value here is not the feature list; every item on it
exists in a dozen frameworks. The value is that you can read the whole thing in an
afternoon, and therefore trust it, and therefore change it. A capability that makes the
loop unreadable costs more than it adds.

## 2. Who this is for

Each reader implies a surface, and the surface is what the spec has to pin down.

| Reader | What they want | The surface they touch |
| --- | --- | --- |
| **The reader** who wants to know what an agent actually is, and has bounced off frameworks where the loop is six layers down | to read it in one sitting and believe it | `loop.py`, `tools.py`, the offline demo (spec §2, §3, §6) |
| **The forker** who needs an agent for one job — a different model, three tools of their own, their own guardrails | a base they can hold in their head instead of a dependency they have to trust | the seams: `model.py`, `tools.py`, `context.py`, `state.py`, `skills/` (spec §5-§7, §11, §14) |
| **The publisher** who will rename it, make it theirs, and ship it | a rename that does not leave someone else's prefix in their error messages | §11 below, MIT license, `AGENTS.md` |
| **The integrator** who runs this from another program rather than a terminal | a contract that does not move under them | `--json`, the event stream, the run dir (spec §15, §16, `docs/integration.md`) |

## 3. What the system must be able to do

This is the generation map. Each row is a capability the intent above requires; the last
column is where its contract lives, which is how a spec gets written — or regenerated —
from this file rather than from someone's memory.

| # | Capability | Why the intent requires it | Contract |
| --- | --- | --- | --- |
| 1 | Drive a model through tool calls to an answer | this is the thing being taught | spec §3 |
| 2 | Four brakes — steps, budget, wall clock, context — and a forced wrap-up | "honest about what a real model does to you"; §5.4 | spec §3 (Termination, Forced wrap-up), §4 |
| 3 | Hold run state, and tell the model its situation every turn | it cannot choose between digging further and wrapping up without knowing where it stands; state in `AgentState` is not state the model knows | spec §4 |
| 4 | One model interface, several provider shapes behind it | a fork changes the model first | spec §5 |
| 5 | Per-role routing, so a cheap model can do cheap work | cost is part of honesty | spec §5 (Roles and routing) |
| 6 | Cost and token accounting the brake can see | a budget that cannot see a spend is not a budget | spec §5 (Cost), §18 |
| 7 | Tools whose failures are results, not exceptions | §5.2, §5.3 | spec §6 |
| 8 | An approval gate that denies by default when unattended | §5.5 | spec §6 (Approval policies) |
| 9 | Context management: compaction and externalization | the loop must survive a long run | spec §7 |
| 10 | Memory across runs | the M in Model + State + Tools + Loop + Memory | spec §8 |
| 11 | A checklist, so nothing is silently half-done | a run once reported `done` for half a task | spec §9 |
| 12 | Reflection on what a run learned | failures are the curriculum | spec §10 |
| 13 | Skills: procedural knowledge with no Python at all | the cheapest seam a forker has | spec §11 |
| 14 | Subagents, with the child's reading kept out of the parent | context is the scarce resource | spec §12 |
| 15 | MCP tools | tools a fork did not have to write | spec §13 |
| 16 | Declarative config, so an agent can be a file | "cheap to fork" includes not editing Python | spec §14 |
| 17 | A CLI, and an offline demo that needs no key | §6.4: the first thirty seconds | spec §2 |
| 18 | An event stream, `--json`, and a run directory | the integrator in §2 | spec §15, §16, `docs/integration.md` |
| 19 | Evaluation: unit tests, protocol evals, trajectory evals | §6.3, and the playbook's "evals gate changes to the harness" | spec §17 |
| 20 | Hooks | a fork's guardrails without a fork of the loop | spec §19 |
| 21 | Coding tools, so this repo can be driven against itself | `docs/case-studies.md` is the receipt | spec §20 |

The one stated exception to the coverage check: `docs/spec.md` §1 (Runtime — Python
version, dependencies, what a fresh checkout needs) is contracted by §4 below, not by a
capability row. Everything else pairs up.

**Known gap, stated rather than left to be discovered:** `src/teacup_agent/a2a/` and the
`teacup-agent-serve` console script ship, are tested, and have no `## n.` section in
`docs/spec.md` — only passing mentions in §6 and §14. Either it earns a section or it
earns an argument for why it is out of scope; today it has neither.

## 4. What binds any implementation

Constraints a spec must satisfy whatever else it says. These are not preferences; a change
that breaks one needs a paragraph, not a shrug.

1. **Readable beats capable.** Past the ceiling in §6.1 the next capability goes into a
   backend class or a new module, never into the loop.
2. **Offline by default.** Tests and evals make no network calls and write nothing into the
   repo (`run_dir=None`, `TEACUP_AGENT_SEARCH=offline`). Anything that needs a key is
   opt-in and says so.
3. **Live calls cost the user money.** Ask before spending, name the model, prefer
   `gpt-5-mini` for verification, report actual spend afterwards.
4. **Python 3.11 and `uv`**, no `pip install`, no type-checking or linting ceremony that is
   not already here.
5. **Modules stay focused** — a guideline of ~500 lines, a hard look at ~700.
6. **MIT, and it stays MIT.** The whole obligation of a fork is keeping the copyright line
   and adding its own.
7. **English in the repo**, whatever language the conversation happens in.

## 5. What a fork must keep to still be this thing

Everything else is yours. These five were each paid for by a run that failed, and each is
executed by something — which is why deleting `evals.py` is not a simplification, it is
forking something else.

| # | Invariant | What executes it |
| --- | --- | --- |
| 1 | **The message protocol.** The assistant message carrying `tool_calls` goes back before its results, and every `tool_call_id` gets exactly one result — including calls that were throttled, denied, or arrived in a forced wrap-up turn. Miss one and the next request is a 400. | `tool_results_follow_their_call()` in `evals.py`, asserted by the cases *"refill: every tool call in a turn must get a result message"* and *"tool calls in the wrap-up turn still need result messages (or resume 400s)"* |
| 2 | **Errors are tool results, not exceptions.** A failing tool hands the model `ERROR: ...` so it can correct itself. That is not defensive programming; it is the reason the loop exists at all. | eval *"self-healing: malformed JSON comes back as an ERROR result, loop continues"*, *"unknown tool: no crash, tell the model which tools exist"*; `tests/test_tools.py::test_bad_json_becomes_error_result_not_exception` |
| 3 | **A broken tool never reads as "this does not exist."** A failed search must say it failed, or the model concludes the fact is not real and says so confidently. | `tests/test_tools.py::test_search_failure_is_not_disguised_as_no_results`, `::test_hosted_search_failure_is_an_error_never_the_offline_corpus`, `::test_web_mode_reports_error_instead_of_pretending` |
| 4 | **A brake also unloads the car.** Hitting a ceiling triggers a forced wrap-up, so a run never ends empty-handed, and names what it never did. | evals *"never empty-handed: running out of steps forces a wrap-up conclusion"*, *"a burnt budget also wraps up instead of printing a stop reason"*, *"time brake: running out of wall-clock time also stops and wraps up"* |
| 5 | **Deny by default when nobody is watching.** Side-effecting tools need approval. "No TTY, so allow it" is the most dangerous default there is. | evals *"approval gate: unattended runs deny side effects and never execute them"*, *"...read-only tools are never gated"*, *"...a denial is followed by a different tool, not a stall"* |

## 6. Success criteria

Vague goals ("readable", "minimal") cannot be failed, so they do not constrain anything.
These can. Each carries the command that measures it, the threshold, and what the command
printed when this file was last checked (2026-09-11). If one goes red, the change is
usually what moved — but not always, and saying which is part of the job.

**6.1 The loop stays one sitting.** Past roughly 100 lines of code `_loop()` stops fitting
in one head, and the next capability belongs in a backend class or a module.

```bash
awk '/^def _loop/{f=1} f' src/teacup_agent/loop.py | grep -vc '^\s*#\|^\s*$'
```

Threshold: ≤ 100. **Measured: 117. Status: RED.** The loop has grown past its own ceiling
and both this file and `README.md` had been quoting 79 since before it did; the numbers are
corrected here rather than the ceiling raised, because raising a ceiling to meet the code
is how the criterion stops meaning anything. Nothing is scheduled yet — the next change to
`loop.py` either moves something out or argues the ceiling was wrong.

(`awk` rather than `sed -n '/^def _loop/,$p'`: both give the same answer today only because
`_loop` happens to be the last function in the file. The first function appended after it
would silently inflate the count.)

**6.2 Two files answer the central question.** "What happens when a tool fails?" must be
answerable from `loop.py` and `tools.py` alone — `execute()` returns the error as the tool
result, `execute_calls()` hands it back to the model. Other modules produce `ERROR:`
strings for their own tools, which is invariant §5.2 being obeyed, not the error path
spreading; what must not move is the dispatch. The executable form of this criterion is
that `tools.execute()` returns a string and never raises:

```bash
uv run pytest tests/test_tools.py -q -k "error_result or cannot_escape"
```

Threshold: green, and the two files above still contain the whole dispatch. **Status:
green.**

**6.3 Evaluation stays free.** The moment checking the loop costs money, people stop
checking the loop.

```bash
uv run python -m teacup_agent.evals     # 27 cases, scripted model, no key, no network
uv run pytest -q
```

Threshold: both green, offline, nothing written into the repo. **Measured: 27/27 eval
cases; 386 tests passed, ~10s (the counts are the claim; the seconds are the machine).
Status: green.** CI runs both plus the demo on every pull
request (`.github/workflows/verify.yml`).

**6.4 The offline demo stays instant and key-less.** `uv run teacup-agent` is the first
thing a new reader types. If it needs a key, a network round-trip or a wait, the project
has lost its first thirty seconds.

```bash
time uv run teacup-agent
```

Threshold: exits 0 with `OPENAI_API_KEY` unset, under one second. **Measured: 0.16s.
Status: green.**

## 7. Non-goals

Stated here so that "should we build it?" has an answer before the argument starts.
`docs/roadmap.md` ends with "Deliberately not doing", which is this list applied case by
case; the short version:

- **No framework wrapped around the loop.** The loop is the artifact.
- **No service layer, no control plane, no multi-tenancy.** Production concerns are named
  in `docs/roadmap.md` and left there rather than smuggled in.
- **No race for tool count.** Tools a fork can add in ten lines do not belong upstream.
- **Not a package format or a registry.** That is teacup-run's job — see §8.

## 8. The other half: teacup-run

[teacup-run](https://github.com/rayhu/teacup-run) is the sibling project, and the line
between them is what keeps both honest:

> **teacup-agent is one agent you can read and fork; teacup-run is the ecosystem around
> many agents.**

So: a feature that would make *this* repo a platform — a package format, a hub, lineage,
budgeted leaderboards, a way to run somebody else's agent — belongs there. A feature that
makes one loop more honest belongs here. teacup-run owns the package format; this repo is
one instance of it, and where the two touch, the contract is `docs/integration.md` (the
`--json` line an external caller parses) and `docs/threat-model.md` (what teacup-run's
sandbox does and does not isolate). The same statement from the other side is in
teacup-run's `docs/intent.md`.

## 9. Fork

The seams table in [CONTRIBUTING.md](../CONTRIBUTING.md) is the map: model and provider in
`model.py`, tools in `tools.py` or an MCP server, context policy in `context.py`, stopping
rules in `state.py`, procedural knowledge in `skills/` with no Python at all. The table is
a testable claim — when the Responses API backend was added, `loop.py` did not change. If
your change forces the loop to grow a special case, that is the signal to look again
before committing.

## 10. Improve

The habit this repo runs on is **before-and-after numbers**, not "should work". Parallel
execution 5.05s → 3.64s; subagents cutting parent context by a third while raising total
cost 35%; a skill catalog that went unused until its wording became an instruction. Those
numbers are why the decisions can be revisited by someone who was not in the room.

The process around that habit is [docs/workflow.md](workflow.md) (how a change gets
from an idea to `main`) and [REVIEW.md](../REVIEW.md) (the pass it goes through on the
way, written by one agent and reviewed by another). When a run goes wrong, the post-mortem
goes into [docs/roadmap.md](roadmap.md) under "Field patches": symptom, root cause, fix,
general principle. Almost every failure recorded there has had the same root cause —
*the model did not know its own situation* — and writing that down is what made the next
fix obvious.
A fork that keeps this habit will find its own patterns; one that does not will rediscover
these.

## 11. Publishing a fork

This is the part nothing else in the repo covers, and the rename is more mechanical than
it looks. There are two names — `teacup_agent`, the Python module, and `teacup-agent`,
the distribution and console script. One command finds every file carrying either:

```bash
grep -ril 'teacup[-_]agent' . --exclude-dir=.git --exclude=uv.lock
```

1. `git mv src/teacup_agent src/<your_module>`.
2. `pyproject.toml`: `name`, `[project.scripts]`, `[project.urls]`, and
   `[tool.hatch.build.targets.wheel] packages`.
3. Sweep both spellings across everything that grep listed — `src/`, `tests/`,
   `examples/`, `main.py`, `.env.example` and the docs.
4. `TEACUP_AGENT_SEARCH` and `TEACUP_AGENT_SEARCH_MODEL` — the two environment
   variables, in `tools.py`, `cli.py`,
   `evals.py`, the tests and the docs. Rename them or you will read someone else's prefix in
   your own error messages.
5. The prompt-cache key prefix — the `set_cache_key` call in `run()`. Cosmetic, but it
   groups cache entries; sharing a prefix with a project you have diverged from is a lie
   about behaviour.
6. `LICENSE` is MIT: keep the existing copyright line, add your own. That is the whole
   obligation.
7. **`AGENTS.md` is the first thing to make yours** (`CLAUDE.md` only imports it, so
   every coding agent reads one set of rules). It is the static context an agent gets in
   your repo, and it will teach your agent the rules you actually keep, not the ones
   inherited here.
8. **This file is the second.** Rewrite §1, §3 and §6 for what *your* fork is for; a
   success criterion inherited unchanged is one nobody will run.

Then `uv sync` — `uv.lock` still pins the old local package name, and until it is
re-locked the first `uv run` fails in a way that looks like your code. After that
`uv run pytest`, `uv run python -m <your_module>.evals` and `uv run <your-cli>` must all
still be green before you publish anything.

## 12. Upstream, or your fork?

Both are correct answers, and the criterion is narrow:

- **Upstream** if it makes the loop easier to understand or closes one of the criteria in
  §6 — a clearer trap comment, an eval case pinning down a rule that only fails against a
  real API, a field patch with its root cause written down.
- **Your fork** if it makes the agent better at *your* job: your provider, your tools,
  your domain guardrails, your service layer. Those are real work and they are welcome to
  exist — just not here, because each one costs a reader some of the afternoon this
  project is trying to sell them.

## Keeping this file true

An intent doc that drifts is worse than none: it launders a stale number as a current
commitment, which is exactly what §6.1 had been doing. So:

- **When a capability lands**, add its row to §3 with the spec section that contracts it.
  A capability with no spec section is the reviewer's finding, not a detail.
- **When the loop, the eval count, the test count or the demo changes**, re-run §6's four
  commands and paste what they print. The date at the top of §6 is the claim; a stale date
  is a stale file.
- **When an invariant gains or loses a guard**, fix §5's right-hand column. A named test
  that no longer exists is the same lie as a number that no longer holds.
- **Nothing in §6 is enforced by CI today.** The three commands in `AGENTS.md` are; the
  criteria here are not, which is why one of them could sit red. Wiring §6.1 into
  `.github/workflows/verify.yml` is the cheapest way to make this file stop needing
  discipline, and it is not done yet.
