# How a change gets to main

This repo is built by agents under human supervision, and the process is part of the
artifact: a harness you can read, built by a loop you can read. This file is that loop.
`AGENTS.md` is the static context every agent gets; `REVIEW.md` is the independent review
pass; this file says how the two fit together.

It is written for one human and two or three agents. It scales down to one of each
without changing, and it is not what a twelve-person team would run.

## The loop

```
intent  ->  phase  ->  plan  ->  implement  ->  independent review  ->  answer  ->  human merges
              ^                                                                        |
              +------------------- what the run taught, written down -------------------+
```

**1. Intent.** What the project is for, and what a change can fail against. Stated once,
not per change; `README.md` says what this is, and the success criteria a fork owes it
are in `docs/intent.md`.

**2. Phase.** One item from `docs/roadmap.md`, ordered by payoff over cost. An item is
the unit of work: it names what to change, what counts as done, and where to read more.
One item per round — batching three features into one message is how a review stops being
possible.

**3. Plan.** The design argument, while it is still cheap to lose. For an item already
written up in the roadmap, that write-up *is* the plan; for anything else, propose the
shape before writing the code and get agreement on it. Claude Code's plan mode is the
convenient way to do this, not the required one.

When a change defines something that outlives it — a file format, a contract another
project implements against — the argument goes in its own `docs/spec-*.md` and the
roadmap item points at it. A spec is not required for an ordinary item; it is what you
write when other people's code has to agree with yours.

**4. Implement.** Claude Code writes the change *and* its tests and evals — tests pin the
deterministic parts, evals pin the loop's behaviour, and both are the contract with the
model. Then it runs the three commands under "Verification standard" in `AGENTS.md` and
quotes the actual output. A part that was not verified is named, with the reason.

**5. Independent review.** A different agent — Codex here — reviews the diff against
`REVIEW.md` with its own context: bugs and logic, security, compliance with what the phase
promised. The author's context is what hides the author's mistakes, so the reviewer must
not inherit it.

**6. Answer the findings.** Claude Code fixes what is real and argues what is not, in the
thread rather than by silently dropping it. Re-run the verification. Findings that name a
rule this repo keeps re-learning go into `AGENTS.md` — the second time, not the first.

**7. A human merges.** Every significant change is read and approved by a person before it
lands. Agents do everything up to that gate and nothing past it: they never commit, push
or open a PR unless asked.

**8. What the run taught.** A failure gets a post-mortem in `docs/roadmap.md` under "Field
patches": symptom, root cause, fix, and the general principle. That section exists because
most failures here have had one root cause — the model did not know its own situation —
and writing it down is what made the next fix obvious.

## The artifacts

Each step commits a file the next step reads. The chain, and not anyone's memory, is the
record of why the code looks like this.

| Step | Artifact | What it holds |
| --- | --- | --- |
| Intent | `README.md`, `docs/intent.md` | what this is for, and the criteria a change can fail |
| Static context | `AGENTS.md` (imported by `CLAUDE.md`) | conventions, hard rules, how to verify, what to ask before spending money |
| Phase and plan | `docs/roadmap.md` | the item: what to change, what counts as done, and afterwards what happened |
| Spec *(when a change defines a format or contract)* | `docs/spec-*.md` | the format itself, its failure modes, and what other code may rely on |
| Implementation | code, `tests/`, `evals.py` | the change and its contract |
| Review | `REVIEW.md` + the review thread | the passes, and the findings from this one |
| Verification | the three commands' output | quoted in the report, and re-run in CI |
| Merge | git history | who approved what, and when |
| Post-mortem | `docs/roadmap.md` "Field patches" | symptom, root cause, fix, principle |

## Why two instruction files are one file

`AGENTS.md` is the canonical file. `CLAUDE.md` exists because Claude Code looks for that
name; it imports `AGENTS.md` and states no rule of its own, so there is one set of rules
whichever name a tool goes looking for. What it does carry is a note about the import
mechanism itself — which is about the file, not about the project, and is repeated below
so that the reviewer's copy is not the short one. Two real rule files would drift, and a
reviewer holding last month's conventions is worse than one holding none: it produces
confident findings about rules that no longer exist.

A symlink would do the same job in fewer bytes and was the first attempt, but it checks out
as a plain text file on Windows without `core.symlinks`, and this repo is meant to be
forked by people whose filesystem we do not get to choose. An import is portable.

The import must sit on **a line of its own**. Inline in a sentence, `@AGENTS.md.` takes the
trailing period as part of the path and imports nothing — and the failure is silent: the
session answers from an empty ruleset without saying so. Verified against Claude Code
2.1.179, both ways round:

```bash
claude -p --model haiku "Do NOT use tools. From the project instructions already in your \
  context only: which language is used in conversation vs in the repo? \
  If nothing is loaded reply exactly NONE LOADED." < /dev/null
```

Inline: `NONE LOADED`. On its own line: the rule, quoted back. The same thing shows in an
interactive session under `/memory`, which lists the import as a child of the file that
pulled it in:

```
1. Project memory     Checked in at ./CLAUDE.md
2. L AGENTS.md        @-imported
```

Check one or the other after any change to how the two files reference each other — a
broken import cannot be seen by reading.

## What is enforced, and what is only written down

Honest accounting, because a rule that nothing checks is a preference:

- **Enforced by code.** The message-protocol rules, the brakes, the approval gate and the
  checklist are guarded by `evals.py` and `tests/`, and CI runs both on every pull request
  and on pushes to `main` (`.github/workflows/verify.yml`). A feature branch with no pull
  request open is checked by nobody.
- **Enforced by the platform, once switched on.** "A human approves the merge" is a branch
  protection rule on GitHub — protect `main`, require a pull request, require the *Verify*
  check to pass. Until that switch is flipped, step 7 is a habit, not a gate. This repo has
  it written down and not yet switched on; that is a choice about a single-author teaching
  repo, not an oversight.
- **Written down only.** Steps 1-6 are conventions between a human and their agents. They
  hold because the files exist and are read, which is the same reason `AGENTS.md` works.
