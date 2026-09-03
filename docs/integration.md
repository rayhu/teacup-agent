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
  "answer": "4",
  "exit_code": 0
}
```

Every field through `throttled` is `AgentState.snapshot()` (`state.py`) unchanged —
reusing it rather than hand-picking a subset means this contract and the human `State:
...` line can never silently drift apart. `answer` and `exit_code` are the two fields
`snapshot()` omits on purpose (it is a summary "without the full message list", and exit
codes are a CLI concept, not agent state).

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
