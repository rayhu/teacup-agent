# Intent

`README.md` says what this is. `CONTRIBUTING.md` says where the seams are. This file says
what the project is **for**, and what a fork owes it — the criteria a change can actually
fail, so "did this stay true to the thing?" is a question with an answer.

## The intent in one sentence

Build the smallest agent harness that is still honest about what a real model on a real
network does to you — and make it cheap enough to fork, change and republish that the
harness, not this repository, is what spreads.

That is a deliberate ordering. The value here is not the feature list; every item on it
exists in a dozen frameworks. The value is that you can read the whole thing in an
afternoon, and therefore trust it, and therefore change it. A capability that makes the
loop unreadable costs more than it adds.

## Who this is for

- **The reader** who wants to know what an agent actually is, and has bounced off
  frameworks where the loop is six layers of abstraction down. Start at `loop.py`.
- **The forker** who needs an agent for one specific job — a different model, three tools
  of their own, their own guardrails — and wants a base they can hold in their head
  instead of a dependency they have to trust.
- **The publisher** who will rename it, make it theirs, and ship it. That path is the
  reason for the MIT license and for the "Publishing a fork" section below.

## Success criteria

Vague goals ("readable", "minimal") cannot be failed, so they do not constrain anything.
These can. Measure them; if one goes red, the change is what moved, not the criterion.

1. **The loop stays one sitting.** `_loop()` in `loop.py` is 79 lines of code today
   (115 with the comments that mark its traps). Past roughly 100 lines of code it stops
   fitting in one head, and the next capability belongs in a backend class or a module —
   not in the loop.
   ```bash
   sed -n '/^def _loop/,$p' src/teacup_agent/loop.py | grep -vc '^\s*#\|^\s*$'
   ```
2. **Two files answer the central question.** "What happens when a tool fails?" must be
   answerable from `loop.py` and `tools.py` alone — `execute()` returns the error as the
   tool result, the loop hands it back to the model. If that answer starts requiring a
   third file, the error path has been spread too thin.
3. **Evaluation stays free.** `uv run python -m teacup_agent.evals` runs 21 protocol
   cases with a scripted model: no API key, no network, nothing written into the repo
   (`run_dir=None`, `TEACUP_AGENT_SEARCH=offline`). CI runs it, plus the tests and the
   demo, on every pull request. The moment checking the loop costs money, people stop
   checking the loop.
4. **The offline demo stays instant and key-less.** `uv run teacup-agent` is the first
   thing a new reader types. If it needs a key, a network round-trip or a wait, the
   project has lost its first thirty seconds.

## What a fork must keep to still be this thing

Everything else is yours. These five were each paid for by a run that failed, and
`evals.py` guards them, which is why deleting `evals.py` is not a simplification — it is
forking something else.

1. **The message protocol.** The assistant message carrying `tool_calls` goes back before
   its results, and every `tool_call_id` gets exactly one result — including calls that
   were throttled, denied, or arrived in a forced wrap-up turn. Miss one and the next
   request is a 400.
2. **Errors are tool results, not exceptions.** A failing tool hands the model
   `ERROR: ...` so it can correct itself. That is not defensive programming; it is the
   reason the loop exists at all.
3. **A broken tool never reads as "this does not exist."** A failed search must say it
   failed, or the model concludes the fact is not real and says so confidently.
4. **A brake also unloads the car.** Hitting a ceiling triggers a forced wrap-up, so a run
   never ends empty-handed, and names what it never did.
5. **Deny by default when nobody is watching.** Side-effecting tools need approval.
   "No TTY, so allow it" is the most dangerous default there is.

## Fork

The seams table in [CONTRIBUTING.md](../CONTRIBUTING.md) is the map: model and provider in
`model.py`, tools in `tools.py` or an MCP server, context policy in `context.py`, stopping
rules in `state.py`, procedural knowledge in `skills/` with no Python at all. The table is
a testable claim — when the Responses API backend was added, `loop.py` did not change. If
your change forces the loop to grow a special case, that is the signal to look again
before committing.

## Improve

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

## Publishing a fork

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
4. `TEACUP_AGENT_SEARCH` — the one environment variable, in `tools.py`, `cli.py`,
   `evals.py`, the tests and the docs. Rename it or you will read someone else's prefix in
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

Then `uv sync` — `uv.lock` still pins the old local package name, and until it is
re-locked the first `uv run` fails in a way that looks like your code. After that
`uv run pytest`, `uv run python -m <your_module>.evals` and `uv run <your-cli>` must all
still be green before you publish anything.

## Upstream, or your fork?

Both are correct answers, and the criterion is narrow:

- **Upstream** if it makes the loop easier to understand or closes one of the four
  success criteria — a clearer trap comment, an eval case pinning down a rule that only
  fails against a real API, a field patch with its root cause written down.
- **Your fork** if it makes the agent better at *your* job: your provider, your tools,
  your domain guardrails, your service layer. Those are real work and they are welcome to
  exist — just not here, because each one costs a reader some of the afternoon this
  project is trying to sell them.

`docs/roadmap.md` ends with "Deliberately not doing", which is the same judgment applied
in advance. Read it before proposing a feature; the short version is no framework wrapped
around the loop, no service layer, and no race for tool count.
