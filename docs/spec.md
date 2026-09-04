# Technical specification

The contract: exact values, exact shapes, exact interfaces. Every number here was read
out of the code, and each one names the symbol it came from so the next reader can
re-check it rather than trust this file.

This is deliberately not a narrative. **Why** any of it behaves this way is
[docs/design-notes.md](design-notes.md); what it is *for* is [docs/intent.md](intent.md);
what is missing is [docs/roadmap.md](roadmap.md).

Names prefixed with `_` are private and may change without notice — the values are given
because they are load-bearing, not because they are stable interfaces.

## 1. Runtime

| | |
| --- | --- |
| Python | >= 3.11 (`pyproject.toml: requires-python`) |
| Package manager | `uv` only — never `pip install` |
| Distribution | `teacup-agent`; module `teacup_agent`; console script `teacup-agent` |
| Runtime deps | `openai>=1.60`, `python-dotenv>=1.0`, `ddgs>=9.0`, `mcp>=2.1`, `pyyaml>=6.0` |
| Dev deps | `pytest>=8.0` |
| Build backend | `hatchling`, wheel packages `src/teacup_agent` |
| License | MIT |

Environment variables:

| Variable | Read by | Meaning |
| --- | --- | --- |
| `OPENAI_API_KEY` | `model.py`, via `.env` | required only for `--live` |
| `TEACUP_AGENT_SEARCH` | `tools.py` | `auto` \| `web` \| `offline`; set by `cli.py` from `--search` |
| any name in `api_key_env` | `agent_config.py` | per-profile key when running `--config agent.yaml` |

## 2. CLI surface

`teacup-agent [goal] [options]`. The positional `goal` defaults to a built-in demo goal
(`cli.py: DEFAULT_GOAL`).

| Flag | Type | Default | Effect |
| --- | --- | --- | --- |
| `--live` | flag | off | call the real OpenAI API; without it the scripted offline model runs |
| `--model` | str | `gpt-5` | model name used with `--live` |
| `--api` | `responses` \| `chat` | `responses` | which OpenAI API backend to construct |
| `--max-steps` | int | `8` | loop turns before the step brake fires |
| `--max-tool-calls` | int | `3` | tool calls executed per turn; `0` = unlimited |
| `--budget` | float | `0.05` | spending ceiling in USD |
| `--deadline` | float | `600.0` | wall-clock ceiling in seconds; `0` = unlimited |
| `--search` | `auto` \| `web` \| `offline` | `auto` with `--live`, else `offline` | search backend mode |
| `--tool-timeout` | float | `30.0` | per-tool-call timeout in seconds |
| `--context-limit` | int | `30000` | compact once the context exceeds this estimate |
| `--run-dir` | str | `runs/<timestamp>` | state + externalized results; `off` disables both |
| `--memory` | str | `memory.json` | long-term memory file |
| `--approve` | `auto` \| `deny` \| `allow` \| `hooks` | `auto` | gate policy; `auto` = ask on a TTY, deny without one; `hooks` defers to `hooks.py`'s `approve_tool_call`, falling back to `auto` when it has no opinion |
| `--resume` | path | none | continue from a `state.json` or the directory holding it |
| `--mcp` | path | `./mcp.json` if present | MCP config; `off` disables |
| `--skills` | path | `./skills` if present | skills directory; `off` disables |
| `--hooks` | path | `./hooks.py` if present | project-local hooks module; `off` disables. §19 |
| `--subagents` | flag | off | offer the `delegate` tool |
| `--subagent-steps` | int | `4` | step ceiling for one child run |
| `--coding-tools` | flag | off | offer `list_files`/`edit_file`/`write_file`/`run_command`. §20 |
| `--plan` | `auto` \| `on` \| `off` | `auto` | upfront checklist; `auto` = on for `--live`, off offline |
| `--reflect` | `auto` \| `on` \| `off` | `auto` | write an experience/lesson note after a qualifying run; `auto` = on for `--live` |
| `-q`, `--quiet` | flag | off | print only the final answer |
| `--json` | flag | off | one machine-readable JSON object on stdout, nothing else; implies `--quiet`. [`docs/integration.md`](integration.md) |
| `--config` | path | none | take everything from a YAML file instead of the flags above |
| `--project-root` | path | current directory | boundary `read_file` may not read outside of; resolved once at startup and applies whether or not `--config` is set |

`--config` is a **parallel track, not a merge**: once it is set, every flag except the
goal, `--quiet`, `--resume` and `--project-root` is ignored (section 14).

On `--resume`, the ceilings mean **additional** allowance: steps and elapsed time already
spent live in the saved state, so the flags are added to it rather than replacing it
(`cli.py`).

## 3. Control loop

`loop.run(...) -> AgentState` sets up; `loop._loop(...)` is the loop itself, split out so
`run()` can guarantee teardown. One turn, in order:

1. **Guards.** `state.elapsed = clock() - started_at`; if `not state.can_continue()`,
   set `status`, run the forced wrap-up when `step > 0`, and break.
2. `state.step += 1`.
3. **Compact** if `state.context_tokens > context_limit`. A compaction that raises is
   caught and the turn continues.
4. **Status note** appended at the end of `messages` — never spliced into the system
   prompt, which would void the prompt cache.
5. **Model call** through `complete_with_retries`. On the final turn (`step >= max_steps`)
   the tool list passed in is empty: wording can be ignored, an empty list cannot.
6. **Append `reply.items`** (trap 1 — extend, not append: one Responses turn can produce
   several entries).
7. **No tool calls = done** (trap 3), unless the checklist has open items and the
   push-back has not fired yet, in which case one `COMPLETION_CHECK` system message is
   appended and the loop continues.
8. **Execute** every call, in parallel, refilling results in the original order (trap 2).
9. **Persist** `state.json` when `run_dir` is set.

After the loop exits, and **before** the final persist so its cost lands in the saved
state, `reflect.maybe_record()` runs when `reflect=True` (section 10).

### Termination

`state.stop_reason()` is evaluated in this fixed precedence:

| Order | Condition | `Status` |
| --- | --- | --- |
| 1 | `step >= max_steps` | `max_steps` |
| 2 | `remaining_budget <= 0` | `out_of_budget` |
| 3 | `time_left() <= 0` | `out_of_time` |
| — | otherwise | `running` |

Plus two loop-level exits: the model returned no tool calls (`done`), or the model call
raised after retries (`error`, with the exception text as the answer).

Time is checked **between turns only**; a wedged tool call is `tool_timeout`'s problem.

### Forced wrap-up

`finalize()` appends a `[forced wrap-up]` system message naming the stop reason and any
unfinished checklist items, then makes one more model call **with no tools**. If that
produces text, it becomes the answer and `state.salvaged = True`. Tool calls arriving in
this turn still receive one result message each — dropping them 400s the next resume.

### Retries

`complete_with_retries(model, messages, tools, emit, attempts=3)`: up to 3 attempts with
`delay = 2**attempt` slept **between** them, so 1s then 2s, and the third failure raises.
**A retry is not a step.** `is_retryable()` returns True for HTTP 429,
any status >= 500, and any error carrying no status code at all; a 4xx is not retried.

## 4. `AgentState` (`state.py`)

| Field | Type | Default | Written by |
| --- | --- | --- | --- |
| `goal` | str | — | caller |
| `messages` | list[dict] | `[]` | loop |
| `step` | int | `0` | loop, once per turn |
| `max_steps` | int | `8` | caller |
| `max_tool_calls_per_step` | int | `3` | caller; `0` = unlimited |
| `remaining_budget` | float | `0.05` | `charge()`, per turn |
| `time_budget` | float \| None | `None` | caller |
| `elapsed` | float | `0.0` | loop, top of each turn |
| `context_tokens` | int | `0` | real `input_tokens`, else an estimate |
| `compactions` | int | `0` | `context.compact()` |
| `input_tokens_total` | int | `0` | loop |
| `cached_tokens_total` | int | `0` | loop |
| `status` | `Status` | `idle` | loop |
| `answer` | str | `""` | loop |
| `salvaged` | bool | `False` | `finalize()` |
| `todo` | list[TodoItem] | `[]` | `plan.decompose()`, `update_todo` |
| `completion_checked` | bool | `False` | loop, at most once |
| `subagent_runs` | int | `0` | `subagent._delegate()` |
| `loaded_skills` | list[str] | `[]` | `skills._load_skill()` |
| `trace` | list[ToolTrace] | `[]` | `execute_calls()` |
| `spend_by_profile` | dict[str, float] | `{}` | `charge(cost, profile)` |

`Status` = `idle` \| `running` \| `done` \| `max_steps` \| `out_of_budget` \|
`out_of_time` \| `error`.

`ToolTrace(step, name, arguments, result, executed=True, skip_reason="")`, where
`skip_reason` is `""` \| `"throttled"` \| `"denied"`.
`TodoItem(text, done=False, note="")`.

`charge(cost, profile="")` subtracts from `remaining_budget` always, and adds to
`spend_by_profile` only when a profile is named. The breakdown is a **diagnostic** for
routing decisions, not a second ledger: a subagent charges its parent one rounded delta
and merges the child's own breakdown in, so the two can disagree in the last decimal.

`snapshot()` returns the 17 human-readable keys printed at the end of a run; it omits
`messages` content, reports `todo_done` as `"n/a"` when there is no checklist, and
`spend` as `{}` when nothing named a profile.

## 5. Model interface

A fork replaces this Protocol and nothing else (`model.py`):

```python
class Model(Protocol):
    def set_cache_key(self, key: str) -> None: ...          # optional
    def complete(self, messages, tools) -> Reply: ...
    def tool_result_item(self, call: ToolCall, result: str) -> dict: ...
```

`ToolCall(id, name, arguments)` — `arguments` is a JSON **string**, not a dict.

`Reply(items, text="", tool_calls=[], cost=0.0, input_tokens=0, cached_tokens=0)`.
`items` is "the entries to append back", not "one assistant message": Chat produces one
per turn, Responses can produce several (a reasoning item plus function calls), and
carrying them back verbatim is what preserves reasoning state.

Three implementations ship: `ResponsesModel` (default), `OpenAIModel` (Chat Completions),
`ScriptedModel` (offline, deterministic, used by the demo and every eval).
`agent_config.build_model(profile)` constructs one of the first two from an `agent.yaml`
profile, including any OpenAI-compatible endpoint via `base_url`.

### Roles and routing (`routing.py`)

An agent makes six kinds of model call, and `agent.yaml`'s `models.roles` maps each to a
profile:

| Role | Call site | What it does |
| --- | --- | --- |
| `main` | `loop.py` | the agent's own turns, and the forced wrap-up |
| `plan` | `plan.py` | decompose the goal into the checklist |
| `compact` | `context.py` | summarize the early context |
| `reflect` | `reflect.py` | write the experience/lesson note |
| `judge` | `trajectory.py` | the LLM judge over a finished run (resolved from the config by `trajectory.main()`, not through a `Router` — scoring a saved run builds one model and nothing else) |
| `subagent` | `subagent.py` | a delegated child run's own `main` |

`Router(build, roles, default)` resolves a role to a profile name (`profile_for`) and to
a model (`for_role`), building each profile **at most once per run** — a fresh instance
would mean a fresh client and an empty cache key every turn. `child(role)` derives a
router whose `main` is that role's profile, which is how a subagent runs elsewhere.
`as_router(model)` accepts a bare `Model` and answers every role with it, so
`loop.run(model=...)` is unchanged. A role name outside `ROLES`, or a profile name not in
`models.profiles`, is an error at config load.

There is no classifier and no per-turn switching: routing is fixed for the run.

### Cost

`estimate_cost(model, input_tokens, output_tokens, cached_tokens)` where `input_tokens`
is the **total** input, cache hits included:

```
(max(0, input - cached) * p_in + cached * p_cached + output * p_out) / 1_000_000
```

`PRICES` in USD per million tokens, `(input, cached input, output)`:

| Model | in | cached | out |
| --- | --- | --- | --- |
| `gpt-5` | 1.25 | 0.125 | 10.00 |
| `gpt-5-mini` | 0.25 | 0.025 | 2.00 |
| `gpt-4.1-mini` | 0.40 | 0.10 | 1.60 |
| anything else (`_DEFAULT_PRICE`) | 1.25 | 0.125 | 10.00 |

An unknown model is priced as `gpt-5` — deliberately pessimistic, so the budget brake
never under-charges. Prices are a local table and go stale; they bound spending, they do
not bill.

### Prompt cache

`loop.run()` calls `set_cache_key("teacup-agent-" + sha256(model_id + "\n" +
system_message)[:16])` on the `main` role's model, derived from the context prefix so
separate runs of the same configuration share cache entries, and from the model id
because caches are per model. A prefix under ~1024 tokens never enters the cache at all,
which is why the other roles — each with its own short, constant system prompt — get no
key. On `--resume` the system message is **not** rebuilt, or every entry earned so far is
voided.

## 6. Tools

Registration is a decorator writing into a module-level `REGISTRY: dict[str, Tool]`:

```python
@tool(description, parameters, requires_approval=False, timeout=None, externalize=True)
def my_tool(...) -> str: ...
```

`specs()` exports the registry in the OpenAI Chat `tools` shape. `execute(name, arguments)`
runs one call and **never raises** — every failure comes back as an `ERROR: ...` string.

### Built-ins

| Tool | Parameters | Approval | Notes |
| --- | --- | --- | --- |
| `search_web` | `query`, `max_results=5` | no | three modes, below |
| `calculate` | `expression` | no | AST-walked arithmetic, not `eval` |
| `read_file` | `path` | no | project-relative, full content (§18's `EXTERNALIZE_OVER` truncates, not this tool), deny-list |
| `remember` | `fact` | no | writes long-term memory |
| `update_todo` | `index` (1-based), `status`, `note=""` | no | see below |
| `send_email` | `to`, `subject`, `body` | **yes** | the gated example; the demo appends to `outbox.jsonl` |

Conditionally registered: `load_skill` (with `--skills`), `delegate` (with
`--subagents`), `delegate_a2a` (with `a2a.peers` non-empty in `--config agent.yaml`,
**yes** approval), one entry per MCP tool, and `list_files`/`edit_file`/`write_file`/
`run_command` (with `--coding-tools`, §20).

`update_todo` declares `status` as the enum `done` | `blocked`, but the value is **not
validated**: the implementation sets `item.done = True` unconditionally and only uses
`status` to decide whether to keep `note`. Any value other than `"blocked"` — including a
typo — is therefore treated as `done`. The enum is advisory, not enforced.

### `search_web` modes

Selected by `TEACUP_AGENT_SEARCH`: `web` always hits the network, `offline` always uses a
built-in corpus and makes zero network calls, `auto` tries the network and falls back
to the corpus **only when the corpus has something** — a broken search over an empty
corpus returns an ERROR, never "no results".

The real backend is DuckDuckGo via `ddgs`, no API key. `_RETRIES = 3` attempts with 1s
then 2s backoff; `_MIN_INTERVAL = 0.5`s between real searches, enforced under a lock because
tools run in parallel. **A failed search says it failed** — it must never read as "there
is nothing to find".

### `read_file` deny-list

Path must resolve inside `cwd` (traversal guard), then `DENIED_FILES` / `DENIED_DIRS`
apply:

- files: `.env`, `.env.*`, `*.env`, `mcp.json`, `memory.json`, `state.json`, `*.pem`,
  `*.key`, `id_rsa*`, `*.p12`
- directories: `.git`, `.ssh`, `.aws`, `.venv`

A denial returns an ERROR that tells the model this is a fixed rule, not a grantable
permission, so it does not retry a different spelling.

### Execution semantics

- **Per-turn cap.** Calls beyond `max_tool_calls_per_step` are refused with an ERROR
  explaining they can be re-sent next turn — and still get a result message each
  (`skip_reason="throttled"`).
- **Approval runs first, serially,** before the thread pool: it either asks a human or
  denies. A denial returns `loop.DENIED` and records `skip_reason="denied"`.
- **Parallelism.** `ThreadPoolExecutor(max_workers=len(to_run))`, results collected
  against **absolute** deadlines so waiting on them in sequence does not add the timeouts
  together. Results are appended strictly in the original call order.
- **Per-call limit** = the tool's own `timeout` if set, else `tool_timeout`, further
  clamped to `max(1.0, time_left())`. On timeout the model gets an ERROR saying the call
  was abandoned — explicitly not that the operation failed — and the thread is left to
  finish. Python cannot kill a thread; real isolation would need a subprocess.
- **Externalization.** With a `run_dir`, a result longer than `EXTERNALIZE_OVER = 2000`
  chars is written to disk and replaced in context by a `context.EXCERPT = 600`-char
  excerpt plus the path, which the model can read back with `read_file`. Tools with
  `externalize=False` (`load_skill`) are exempt — a procedure is an instruction, not raw
  material.

### Approval policies

`deny_all` is the default passed into `loop.run()`. `cli.py` maps `--approve` to
`auto` (interactive prompt when `stdin.isatty()`, otherwise deny), `deny`, or `allow`.

## 7. Context management (`context.py`)

- `estimate_tokens(text)` ≈ `cjk/1.5 + (len - cjk)/4 + 1`. Used **only** to decide
  whether to compact; billing uses the real usage numbers.
- `safe_cut_points(messages)` returns every index where a cut cannot break the message
  protocol: no announced tool-call id still unfilled (both API shapes —
  `tool_calls`/`role=tool` and `function_call`/`function_call_output`), **and** the kept
  previous entry is not a Responses `reasoning` or `message` item — a turn arrives as a
  group and a `function_call` later in the same group needs that reasoning item, so a
  cut must land on a turn boundary (Field patch G).
- `compact(state, model, limit, keep_recent=8, profile="")` keeps `head = 2` entries (system + the
  original goal) and the last 8, summarizes everything between the newest safe cut point
  and replaces it with one `[context summary]` system message. Returns estimated tokens
  saved. Returns `0` — changing nothing — when there is no safe cut point, or when the
  summarizer produced no text. The summarizer call is charged to the run's budget.

## 8. Memory (`memory.py`)

Two layers of lifetime, and inside the long-term layer two tiers of **trust**. Short
term is `AgentState.messages` and dies with the task. Long term is a JSON file:

```json
{
  "facts": ["...", "..."],
  "notes": [{ "kind": "experience", "text": "..." }]
}
```

`Memory(path="memory.json", limit=20, note_limit=10)`.

| | `facts` | `notes` |
| --- | --- | --- |
| Written by | the model, via the `remember` tool | the harness, via `reflect.py` |
| When | mid-task, deliberately | after the run has already ended |
| Reviewed | chosen by the model | unreviewed by construction |
| Kept | last 20 | last 10 |
| Entry | a string | `{"kind": "experience" \| "lesson", "text": ...}` |

Both de-duplicate exactly and evict oldest-first. `recall()` returns up to two labelled
blocks, and the notes block says in the prompt that they are auto-generated and not
human-verified, so the model can weigh them differently. A corrupt file loads as empty
rather than taking the agent down. `NullMemory` never touches disk — what evals and unit
tests use.

The replaceable surface is `remember()` + `note()` + `recall()`.

## 9. Checklist (`plan.py`)

`decompose(goal, model)` makes **one** model call with no tools and returns 1–5
`TodoItem`s. A planner that fails returns `[]`, which degrades exactly to the unplanned
behaviour — a broken planner must never stop a run. `render()` produces the `[x]`/`[ ]`
block carried in every turn's status note; `pending()` returns items still open.

The model ticks items off with `update_todo(index, status, note)`. `blocked` is settled:
it stops being outstanding but keeps its reason. The completion push-back
(`COMPLETION_CHECK`) fires **at most once per run**, guarded by
`state.completion_checked`.

## 10. Reflection (`reflect.py`)

`--reflect` (default `auto`: on for `--live`, off offline). Shaped exactly like
`plan.py` — one extra model call, no tools, and any failure writes nothing rather than
sinking an already-finished run.

Triggers are computed for free from `trajectory.mechanical()`; **no model call is made
unless one fires**:

| Note | Condition |
| --- | --- |
| `experience` | `status == "done"` and not `salvaged` and `pending_todos == 0` and `duplicate_tool_calls == 0` and not `action_never_attempted` |
| `lesson` | `failed_tool_calls > 0` and `delivered` |

The model is asked for JSON with exactly the requested keys; the first `{...}` in the
reply is parsed, and anything unparseable means nothing is written. `state.charge()` is
called on the reply, so the cost is honest even though `status` is already final. Notes
land in `Memory.notes`, never in `facts` (section 8). The kinds written are returned and
emitted as the `reflected` event.

The intended path for a good note is a human promoting it into `docs/roadmap.md`'s
"Field patches" — this is the candidate feed for that, not a replacement.

## 11. Skills (`skills.py`)

A skill is a directory under `--skills` (default `./skills`) containing `SKILL.md` with
YAML frontmatter:

```markdown
---
name: web-research
description: One line. This is the only part always in context.
---
<the procedure>
```

`catalog()` builds the always-loaded block: one `- name: description` line each, prefixed
by an instruction to call `load_skill(name)` first when the task matches. The body arrives
only as a `load_skill` tool result, and never enters the system prompt. Loaded names are
recorded in `state.loaded_skills`.

## 12. Subagents (`subagent.py`)

`--subagents` registers `delegate(task, wanted="")`. Defaults from
`subagent.enable(...)`: `max_steps=4` (`--subagent-steps`), `budget_share=0.4`,
`timeout=300.0`s for the tool call itself.

- The child's budget is `parent.remaining_budget * budget_share`, read **at call time**,
  so a nearly-spent parent cannot fund an expensive child. Zero budget returns an ERROR.
- The child inherits `time_left()`, the approval policy, and every tool except `delegate`
  — **one level of delegation only**.
- Child artifacts go to `<run_dir>/sub<NN>/`.
- On return the parent is charged `budget - child.remaining_budget` and accumulates the
  child's token counts; `parent.subagent_runs += 1`.
- Only `child.answer` crosses back into the parent context. A child that ends without one
  returns an ERROR naming its status; one that stopped early appends
  `[subagent stopped early: <status>]`.

## 13. MCP (`mcp_tools.py`)

Config file (`mcp.json`, template `mcp.example.json`), one entry per server under
`"servers"`:

| Key | Meaning |
| --- | --- |
| `url` | HTTP transport — mutually exclusive with `command` |
| `command`, `args`, `env` | stdio transport |
| `tools` | optional allowlist; every schema costs prefix tokens on every request |
| `approve` | `auto` (default) \| `all` \| `none` |
| `stderr` | `hide` (default) \| `show` |

Tools register as `server__tool`, with anything outside `[A-Za-z0-9_-]` replaced by `_`.
`CALL_TIMEOUT = 60.0`s per call. Approval under `auto` opens **only** tools the server
explicitly annotates `read_only_hint`; a server that annotates nothing gets everything
gated. `McpHub` owns one asyncio loop on a daemon thread and blocks the sync caller on it.

## 14. Declarative config (`agent_config.py`)

`--config agent.yaml` replaces the flags rather than merging with them: everything except
the goal, `--quiet` and `--resume` comes from the file. Template: `agent.example.yaml`.
Secrets never go in it — name an env var with `api_key_env`, or embed `${VAR}` — and
`agent.yaml` is gitignored for the same reason as `mcp.json`.

| Block | Keys |
| --- | --- |
| `models.default` | the profile every unmapped role falls back to |
| `models.roles` | `main` \| `plan` \| `compact` \| `reflect` \| `judge` \| `subagent` -> a profile name. Omit the block for single-model behaviour |
| `models.profiles.<name>` | `provider` (`openai` \| `openai-compatible`), `api` (`responses` \| `chat`), `model`, `api_key_env`, optional `base_url`, optional `reasoning_effort` |
| `mcp` | the same per-server shape as `mcp.json`'s `servers`, nested one level deeper |
| `tools` | `exclude: [names]`, `subagents.enabled`, `subagents.max_steps` |
| `skills` | `dir:` a path, or `off` |
| `runtime` | `max_steps`, `max_tool_calls_per_step`, `budget`, `deadline`, `tool_timeout`, `context_limit`, `approve`, `plan`, `reflect`, `search`, `memory`, `run_dir` |
| `a2a` | `peers.<name>` (`url`, optional `api_key_env`) offers `delegate_a2a`; `card` (`name`, `description`, `version`) is this agent's identity when served with `teacup-agent-serve` |

`load(path) -> AgentConfig` expands `${VAR}` from the environment;
`build_model(profile)` returns a `Model`; `build_router(cfg)` returns a `routing.Router`
that builds profiles lazily, so a profile no role uses is never constructed (and its
`api_key_env` never needed); `resolve_run_dir(value)` turns `runs` into a fresh
`runs/<timestamp>` and `off` into `None`.

`trajectory.py` takes the same file: `--config agent.yaml [--judge-profile <name>]`
sources the judge model from `models.profiles` instead of `--model`, so the eval cannot
silently diverge from what the agent itself runs. The default is `models.roles.judge`
when the config sets one, else `models.default`; `--judge-profile` overrides it for one
invocation, the same split `goal`/`--quiet`/`--resume` have.

## 15. On-disk formats

| Path | Written by | Shape |
| --- | --- | --- |
| `runs/<timestamp>/state.json` | `persist.save()` after every step | `dataclasses.asdict(AgentState)`, UTF-8, indent 2 |
| `runs/<timestamp>/sub<NN>/` | subagent runs | same, one directory per child |
| `runs/<timestamp>/step<NN>_<i>_<name>.txt` | `context.externalize()` | the full tool result |
| `memory.json` | `Memory.save()` | `{"facts": [...], "notes": [{"kind", "text"}]}` |
| `outbox.jsonl` | `send_email` | one JSON line per approved send (nothing is really sent) |
| `mcp.json` | you | section 13 |
| `agent.yaml` | you | section 14 |
| `skills/<name>/SKILL.md` | you | section 11 |

`persist.save()` writes a `.tmp` file and renames it, so a crash mid-write cannot leave a
half-written state. `persist.load()` rebuilds `trace` and `todo` into dataclasses by hand
— `asdict()` flattened them on the way out and everything downstream expects objects.

`runs/` and `memory.json` are gitignored, as are `.env`, `mcp.json` and `agent.yaml`.

## 16. Event stream

`loop.run(on_event=...)` receives `(event: str, data: dict)`. The full set:

| Event | Emitted when |
| --- | --- |
| `planned` | the checklist was decomposed |
| `skills` | the skill catalog was enabled |
| `compacted` | context was compacted |
| `retry` | a model call is being retried |
| `tool_call` | a call is about to run |
| `tool_result` | a call returned |
| `throttled` | a turn exceeded the per-turn cap |
| `approval_required` / `approved` / `denied` | the gate |
| `vetoed` | a project-local `hooks.py`'s `before_tool_call` blocked a call |
| `hooks_loaded` | a project-local `hooks.py` was loaded |
| `externalized` | a result went to disk |
| `completion_check` | the checklist push-back fired |
| `answer` | the model finished |
| `stopped` | a ceiling was hit |
| `salvaged` | the forced wrap-up produced an answer |
| `reflected` | a run wrote an experience and/or lesson note |
| `saved` | the final state was written |
| `error` | a model call failed, or a compaction raised |

This is the observability contract; `cli.py` is one consumer of it.

## 17. Evaluation

Two kinds, and conflating them is the mistake this repo names explicitly.

**Protocol evals** — `uv run python -m teacup_agent.evals`. 22 cases against
`ScriptedModel`: no API key, no network, `run_dir=None`, nothing written into the repo.
They pin the message protocol, the brakes, the wrap-up, compaction, the approval gate,
the checklist, delegation and skills. They must stay green and must stay free.

All three commands run in CI (`.github/workflows/verify.yml`) on every pull request
and on pushes to `main`, with `TEACUP_AGENT_SEARCH=offline` forced and read-only token
permissions.

**Trajectory scoring** — `uv run python -m teacup_agent.trajectory runs/<timestamp>`,
`--judge` adds the LLM half, `--config` + `--judge-profile` source the judge from
`agent.yaml` instead of `--model` (section 14).

`mechanical(state)` is deterministic and returns:

| Key | Meaning |
| --- | --- |
| `status`, `steps`, `elapsed_s` | how the run ended and what it used |
| `tool_calls`, `failed_tool_calls` | executed calls; those whose result began `ERROR:` |
| `duplicate_tool_calls` | same tool, same arguments, more than once |
| `throttled`, `denied` | calls that never ran, by reason |
| `retried_after_denial` | an identical call re-sent after being denied |
| `action_never_attempted` | the goal used an action verb and no gated tool was ever called |
| `pending_todos` | checklist items still open |
| `compactions`, `salvaged`, `cache_hit`, `cost_hint` | run mechanics |
| `answer_chars`, `answer_citations` | answer size and link count |
| `unsupported_citations` | links in the answer that appear in no tool result |
| `asks_user_back` | the answer asks the user a question (EN and ZH phrasings) |
| `asks_without_trying` | asked for authorization without ever attempting the call |
| `delivered` | an answer exists, the run did not end in `error`, and it is not the "(no final answer" placeholder |

`unsupported_citations` is the deterministic invented-citation detector, and it is more
accurate than an LLM judge for that one question. `action_never_attempted` counts a
*denied* call as an attempt — the model did its part.

**Routing bench** — `uv run python -m teacup_agent.bench --config agent.yaml`
(`bench.py`). Runs the same goals under several *policies* (named role -> profile maps)
and prints one table: cost, steps, compactions, the `mechanical()` columns, and
optionally one judge's scores.

| Concept | Meaning |
| --- | --- |
| `Policy(name, roles)` | a role -> profile map. Setting `judge` here is an error: the judge is pinned for the whole matrix with `--judge-profile`, or the quality column varies with the thing being measured |
| `Goal(name, text, policies, ...)` | one task, and the policies worth running it under — the matrix is **sparse**, because a policy that differs only in `compact` measures nothing on a run that never compacts |
| `routed_roles` / `fired_roles` | which roles the policy moves, and which ones actually ran. When those sets do not intersect, the cell is a copy of the baseline and `format_table()` says so |
| `cites` next to `unsup` | `unsupported_citations` is a numerator with no denominator — a cell that cited nothing scores a perfect 0 — so the table prints `answer_citations` beside it |

`--dry-run` prints the matrix and the ceiling without running anything; without `--yes`
it asks before spending, quoting `cells x --budget` as a **hard** ceiling (each run
stops at `--budget`). Offline it is exercised by `tests/test_bench.py`, which also pins
the claim Stage A rests on: a `chat` subagent under a `responses` parent leaves both
message lists internally consistent, because the only things that cross between models
are a compaction summary (a `role="system"` entry) and a subagent's answer (a string).

## 18. Tuned constants

Every one of these was set by a measurement; the reasoning is in
[design-notes.md](design-notes.md).

| Constant | Value | Where |
| --- | --- | --- |
| `EXTERNALIZE_OVER` | 2000 chars | `loop.py` |
| `EXCERPT` | 600 chars | `context.py` |
| `compact(keep_recent=)` | 8 entries | `context.py` |
| `head` kept by `compact` | 2 entries | `context.py` |
| retry `attempts` | 3, sleeps 1s then 2s | `loop.py` |
| `_MIN_INTERVAL` | 0.5s between searches | `tools.py` |
| `_RETRIES` (search) | 3, sleeps 1s then 2s | `tools.py` |
| `CALL_TIMEOUT` (MCP) | 60.0s | `mcp_tools.py` |
| `Memory(limit=)` | 20 facts | `memory.py` |
| `Memory(note_limit=)` | 10 notes | `memory.py` |
| `budget_share` (subagent) | 0.4 of remaining | `subagent.py` |
| `timeout` (delegate tool) | 300.0s | `subagent.py` |
| checklist size | 1–5 items | `plan.py` |
| `run_command` default timeout | 60.0s | `coding_tools.py` |
| `run_command` max timeout | 300.0s | `coding_tools.py` |

## 19. Hooks (`hooks.py`)

Loaded from a project-local `hooks.py` (`--hooks`, default `./hooks.py` if present;
`off` disables), the same opt-in-by-file convention as `mcp.json`/`skills/` — but
**not** gitignored, since it carries policy, not secrets (`docs/threat-model.md`).

| Callback | Signature | Return | Effect |
| --- | --- | --- | --- |
| `before_tool_call` | `(call) -> str \| None` | a string vetoes; `None` allows | checked before the approval gate; a veto becomes the call's `ERROR:` result |
| `after_tool_result` | `(call, result) -> str` | the (possibly rewritten) result | applied to executed calls only, before emit/externalize/trace |
| `approve_tool_call` | `(call, spec) -> bool \| None` | `True`/`False` decides; `None` = no opinion | only consulted under `--approve hooks`; `None` falls back to `auto`'s behaviour |

Failure handling per callback, all in `hooks.py`'s module docstring: `before_tool_call`
fails **closed** (an exception becomes a veto), `approve_tool_call` fails to "no
opinion" (also closed), `after_tool_result` fails to a no-op.

`hooks.example.py` is the committed template, demonstrating an argument-aware
allowlist ("`send_email` only to these domains" — roadmap #14's own example) using
only the built-in `send_email` tool.

## 20. Coding tools (`coding_tools.py`)

Registered dynamically by `--coding-tools` (`loop.run(coding_tools=True)`), the same
`enable()`/`disable()`-on-`tools_mod.REGISTRY` shape `subagent.py`'s `delegate` uses —
these four tools do not exist in the registry at all, and cost no prefix tokens, unless
the flag is passed.

| Tool | Parameters | Approval | Notes |
| --- | --- | --- | --- |
| `list_files` | `path="."`, `recursive=False` | no | top-level unless `recursive`; deny-list applies, pruned before descending |
| `edit_file` | `path`, `old_string`, `new_string` | **yes** | exact-substring replace; errors on zero or >1 matches; always re-reads the file fresh |
| `write_file` | `path`, `content` | **yes** | new files only — errors if `path` already exists |
| `run_command` | `command`, `timeout=None` | **yes** | `shell=True`, cwd = project root; `timeout` passed to `subprocess.run` itself (actually kills the child, unlike the loop's generic per-call timeout) |

`list_files`/`edit_file`/`write_file` all resolve `path` via `tools._resolve_project_path()`
(the shared traversal guard, using `Path.is_relative_to()`) and `tools._is_denied()`
(the deny-list) — the same two checks `read_file` uses, not a second copy of either.
`read_file` and these three tools all had, until an independent review caught it, their
own copies of a naive `str(target).startswith(str(root))` traversal check that a sibling
directory sharing the root as a string prefix could defeat; `_resolve_project_path()` is
the one place that check now lives.

`run_command`'s `timeout` is clamped to `_MAX_COMMAND_TIMEOUT` (300.0s) regardless of
what is requested; its registered `Tool.timeout` (310.0s) is a backstop above that, since
`subprocess.run`'s own timeout is what actually bounds the call.

`docs/threat-model.md`'s "What #20 (coding tools) added" section states what changed in
this repo's trust boundary and what did not; `hooks.example.py`'s `run_command`
allowlist (§19) is the concrete mechanism for using it unattended.
