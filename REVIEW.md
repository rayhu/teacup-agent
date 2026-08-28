# REVIEW.md

The independent review pass. One agent writes the change; a **different** agent, with its
own context and no memory of the arguments that produced the diff, reviews it. That
separation is the whole point — the author's context is exactly what hides the author's
mistakes.

In practice: Claude Code implements a roadmap item, Codex reviews the diff against this
file, Claude Code answers the findings, and a human decides what merges. The reviewer may
be any agent or person who did not write the code; the passes below do not change.

## How to run it

```bash
git diff main...HEAD          # or the phase's diff, if the phase is several commits
```

Give the reviewer this file, `AGENTS.md`, and the roadmap item the phase was supposed to
land. Nothing else is needed: a review that has to ask what the change was for is
reviewing the wrong thing.

## The three passes

Run all three. They find different failures, and skipping one is how the cheap bug ships.

**1. Bugs and logic.** Does it do what it says under inputs that are not the happy path?
Empty result, network error, a tool that returns a string where a dict was assumed, a
`None` where a dataclass was rebuilt from JSON. Check that imports are real packages and
that error handling covers a failure someone will actually hit.

**2. Security.** What can this reach that it should not? New file reads against the
`read_file` deny-list, new tools against the approval gate, anything that widens what an
unattended run can do, secrets in code or in a run directory, and any path where user
text becomes a shell command or a file path.

**3. Compliance.** Does the diff match what the phase said it would do? Compare against
the roadmap item: work that was not asked for is as much a finding as work that was
skipped. Then check the rules this repo does not negotiate, each of which was paid for by
a failed run:

- The message protocol: the assistant message carrying `tool_calls` goes back *before*
  its results, and every `tool_call_id` gets exactly one result — including throttled,
  denied and forced-wrap-up calls.
- Errors are tool results, never exceptions.
- A broken tool must never read as "this does not exist".
- Nothing is silently half-done: a run that cannot finish says so.
- The loop stays readable. If the change made `loop.py` grow a special case, say so;
  that is a finding even when the code is correct.

## What counts as a finding

**Important**, and worth blocking a merge: wrong behaviour, a protocol violation, a
security hole, a claim in the report that the diff does not support, a documented default
that the code does not implement.

**Nit**, and capped at five per review: naming, comment wording, a tidier way to write a
correct line. Past five, summarize the rest in one line and move on — a review that
returns thirty nits gets read as thirty nits and its two real findings are lost.

**Not a finding**: style already settled by the surrounding code, type-checking or linting
ceremony this repo has decided not to carry, and anything under `docs/roadmap.md`'s
"Deliberately not doing".

## What the reviewer must verify, not assume

- Run the three commands from `AGENTS.md`'s verification standard and quote the output.
  A green report from the author is a claim; the reviewer's job includes checking it.
- If the change is behavioural, look for the before-and-after number. "It feels better" is
  not a measurement, and this repo has a habit of measuring.
- If the report says something is unverified, that is acceptable — check the reason is
  the real one ("no API key in this shell" is; "should work" is not).

## Output

One list, most severe first. Each finding: file and line, what breaks, and the input or
state that breaks it. No summary of what the diff does — the author knows, and the human
reading the review has the diff.

Findings the author disagrees with stay in the thread rather than being silently dropped;
the human deciding the merge reads both sides. When a finding turns out to name a rule
this repo keeps re-learning, it goes into `AGENTS.md` — the second time, not the first.
