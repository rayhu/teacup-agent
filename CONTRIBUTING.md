# Contributing, or: how to change this thing

This repo is meant to be forked. What follows is less a contribution process than a map
of where the seams are, so you can cut along them instead of across them.

## Before anything else

```bash
uv sync
uv run pytest                          # unit + protocol tests
uv run python -m teacup_agent.evals    # loop health, scripted model, free
uv run teacup-agent                    # offline demo, no API key, instant
```

All three must be green before and after your change. The evals in particular are the
contract: they use a scripted model, cost nothing, and pin down the message-protocol
rules that only fail against a real API, where they are expensive to discover.

## Where the seams are

| You want to | Change this | You should not need to touch |
| --- | --- | --- |
| Use a different model or provider | a class in `model.py` | `loop.py` |
| Add a tool | `tools.py`, or point at an MCP server | anything else |
| Change how context is spent | `context.py` | the loop's control flow |
| Add procedural knowledge | a folder under `skills/` | any Python at all |
| Change what stops a run | the guards in `state.py` | the tool execution path |

That table is the design goal, and it is testable: when the Responses API backend was
added, `loop.py` did not change. If your change forces the loop to grow a special case,
that is worth a second look before it is worth a commit.

## The rules that are not negotiable

Each was paid for by a run that failed. `evals.py` guards them, so you will find out
quickly, but knowing why saves the detour:

1. **The message protocol.** The assistant message carrying `tool_calls` goes back before
   its results, and every `tool_call_id` gets exactly one result — including calls that
   were throttled, denied, or arrived during a forced wrap-up. Miss one and the next API
   request fails with a 400.
2. **Errors are tool results, not exceptions.** A failing tool hands the model an
   `ERROR: ...` string so it can correct itself. That is what the loop is for.
3. **A broken tool must never read as "this does not exist."** A failed search says it
   failed, or the model concludes the fact is not real.
4. **Nothing is silently half-done.** If a run can end without doing what it was asked,
   something is missing: a checklist item, a metric, or a forced wrap-up.

## Things this project deliberately does not do

`docs/roadmap.md` ends with a list, and it is worth reading before proposing a feature.
The short version: no agent framework wrapped around the loop, no service layer, and no
race for tool count. The value here is that the loop fits in one head. A capability that
makes it unreadable costs more than it adds — which is a real trade-off, not a slogan, and
sometimes the right answer is to make the change in your fork rather than here.

## If your change is behavioural

Prompt changes, new guards, anything that alters how the agent decides: measure it. This
repo has a habit of before-and-after numbers (parallel execution 5.05s → 3.64s, subagents
cutting parent context by a third while raising total cost 35%, a skill catalog that went
unused until its wording became an instruction), and the habit is the reason those
decisions can be revisited later by someone who was not there.

`uv run python -m teacup_agent.trajectory runs/<timestamp>` scores a real run without an
LLM; add `--judge` for the qualitative half.

## Working with a coding agent on this repo

`AGENTS.md` is the static context an agent gets here — `CLAUDE.md` imports it and states
no rule of its own, so Claude Code and Codex work from the same conventions. It carries
the same rules as this file, plus how to verify, what to ask before spending money, and the
convention that failures get written down in `docs/roadmap.md` with their root cause. If
you fork this and work with an agent, that file is the first thing to make yours.

The rest of the process is two files: [docs/workflow.md](docs/workflow.md) for how a
change gets from an idea to `main`, and [REVIEW.md](REVIEW.md) for the review pass it goes
through on the way — written by one agent, reviewed by a different one, merged by a human.
CI (`.github/workflows/verify.yml`) runs the three commands above on every pull request
and on pushes to `main`, so "all three must be green" is checked and not just asked for.
On a feature branch with no pull request open yet, nothing checks it but you.
