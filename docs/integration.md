# External invocation: `--json`

**Status:** shipped. This is the contract an external caller — teacup-run's sandboxed
subprocess launcher is the first one — parses. Everything else on stdout is free to
change shape; this is not.

## Why

`uv run teacup-agent "<goal>"` prints a human log: progress lines, `Answer: ...`, then a
`State: {...}` line only when not `--quiet`. None of that is meant to be parsed — the
wording changes freely, and the state line is opt-out, not guaranteed. A caller that is
another program, not a person watching a terminal, needs something stable instead.

## The contract

`--json` implies `--quiet` (set once, in `main()`, right after `argparse.parse_args`) and
prints **exactly one JSON object on stdout and nothing else**, whether the run used plain
flags or `--config agent.yaml` — both paths end in the same `_finish()` (`cli.py`), so the
shape does not depend on which one launched the run.

```json
{
  "goal": "2+2",
  "step": 1,
  "max_steps": 8,
  "remaining_budget": 0.0169,
  "elapsed_s": 4.2,
  "context_tokens": 812,
  "compactions": 0,
  "cache_hit": "n/a",
  "status": "done",
  "salvaged": false,
  "subagents": 0,
  "skills_loaded": 0,
  "todo_done": "n/a",
  "messages": 4,
  "tool_calls": 1,
  "throttled": 0,
  "spend": {"gpt-5": 0.0031},
  "answer": "4",
  "exit_code": 0
}
```

Every field through `throttled` is `AgentState.snapshot()` (`state.py`) unchanged —
reusing it rather than hand-picking a subset means this contract and the human `State:
...` line can never silently drift apart. `answer` and `exit_code` are the two fields
`snapshot()` omits on purpose (it is a summary "without the full message list", and exit
codes are a CLI concept, not agent state).

`spend` is dollars per model, keyed by **profile name** under `--config agent.yaml`
(`big`, `small`, whatever `models.profiles` calls them) and by **model name** on the
flag-driven path, which has no profile names to use — `{"gpt-5": ...}` with `--live`,
`{"default": ...}` for the offline demo. It is a diagnostic, not a second ledger:
`remaining_budget` is the ledger and `spend` is not: they can disagree, and since
`--search hosted` landed they can disagree by a lot rather than in the last decimal.
Two reasons. A subagent charges its parent one rounded delta. And a **tool** that
spends money — today only `search_web`'s hosted backend, at roughly $0.01 a search —
is charged against `remaining_budget` without a profile name, because it is not any
model's spend and filing it under whichever profile happened to run the turn would be
a wrong attribution rather than a missing one. So a run that made three hosted
searches can report `remaining_budget` down $0.033 with `spend` showing $0.003.
Reconcile against `remaining_budget`; treat `spend` as the per-model diagnostic it
says it is. `{}` when nothing named a profile.

`exit_code` mirrors the process's actual exit code: `0` if `status == "done"`, `1`
otherwise (`max_steps` / `out_of_budget` / `out_of_time` / `error`). A caller can trust
either one; they are computed from the same value.

## What this is not

Not a replacement for `state.json` (`persist.py`) — that has the full message history and
tool trace, for resuming or auditing a run. `--json`'s stdout line is a summary for a
caller that only wants the outcome, not the transcript.

Not versioned by a schema field. If a field is ever renamed or removed here, that is a
breaking change to be called out explicitly (grep every caller of `snapshot()` first) —
adding a field is not, and a caller should not fail on an unrecognized key.
