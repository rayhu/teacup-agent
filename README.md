# mini-agent

A **minimal AI agent that still has all the organs** — one formula turned into code
you can actually run:

```
Agent = Model + State + Tools + Control Loop + Memory/Evals
```

Every part really does its job (no placeholder modules), and every implementation is
deliberately kept to a few dozen lines so you can replace them one at a time.

## Quick start

```bash
uv sync                      # create the environment (.venv + uv.lock)

uv run mini-agent            # offline demo: no API key, no cost
uv run mini-agent "compute (3200-450)*0.6 and tell me what CUDA is"

uv run mini-agent --live "research NVIDIA's GPU strategy"   # real OpenAI call, needs .env
uv run mini-agent --live --api chat "..."                   # compare the older Chat Completions path

uv run python -m mini_agent.evals   # evals: offline assertions about the control loop
uv run pytest                       # the same cases + tool/memory unit tests
```

`--live` needs a `.env`:

```bash
cp .env.example .env   # then fill in OPENAI_API_KEY
```

## Layout

```
src/mini_agent/
├── model.py    Model        the only part that thinks. Responses / Chat / scripted
├── state.py    State        goal / messages / step / budget / status + tool trace
├── context.py  context work externalize big results, compact history over the limit
├── tools.py    Tools        function + JSON Schema + safe execution (errors become
│                            tool results, not exceptions)
├── memory.py   Memory       short term = messages; long term = memory.json
├── plan.py     checklist    decompose the goal into action items, hold the run to them
├── persist.py  persistence  writes runs/<timestamp>/state.json every step, --resume
├── loop.py     Control Loop LLM -> tool call -> tool result -> LLM
├── evals.py    Evals        health check on the loop with a scripted model, free
├── trajectory.py            score a real run: mechanical metrics + an LLM judge
└── cli.py                   command-line entry point
tests/                       pytest: the eval cases + tool/memory/context unit tests
NOTES.md                     the original study notes, annotated with what implements what
docs/roadmap.md              what this is still missing, and in what order to add it
```

## What each of the five parts does

| Part | File | Today | When you outgrow it |
| --- | --- | --- | --- |
| Model | `model.py` | Responses API (default) + Chat Completions + cache-aware pricing, plus an offline scripted model | Claude / local models / multi-model routing |
| State | `state.py` | dataclass: steps, budget, time, status machine, trace; saved every step and resumable | distributed or concurrent runs |
| Context | `context.py` | externalize big results + compact over the limit (protecting call/result pairing) | relevance-based recall, hierarchical summaries |
| Tools | `tools.py` | search_web (**real network**), calculate, read_file, remember, send_email (gated) | browser, SQL, code execution, MCP |
| Control Loop | `loop.py` | one loop + four brakes + parallel tools + retries + a completion check | subagents, delegated planning |
| Plan | `plan.py` | goal decomposed into a checklist the loop enforces | hierarchical plans, replanning mid-run |
| Memory | `memory.py` | JSON file + dedupe + keep the last N | vector store, summarization, relevance recall |
| Evals | `evals.py` + `trajectory.py` | 15 offline protocol cases + real-trajectory scoring | fixed task suites, cross-version regression |

## The three search_web modes

Web search goes through [`ddgs`](https://pypi.org/project/ddgs/) (DuckDuckGo) and
needs **no API key**; `uv sync` installs it. Switch with the `MINI_AGENT_SEARCH`
environment variable:

| Value | Behaviour | Use for |
| --- | --- | --- |
| `auto` (default) | search the web; on failure fall back to the local corpus and say why | everyday use |
| `web` | web only; on failure return `ERROR:` | when offline material must not stand in |
| `offline` | the three local corpus entries only, zero network calls | evals, unit tests, demos |

`--search` overrides it on the command line. The default follows the run mode:
`auto` (real network) with `--live`, `offline` (network-free and instant) for the
offline demo.

```bash
uv run mini-agent --live "research OpenAI's funding and competition"  # really goes online
MINI_AGENT_SEARCH=offline uv run mini-agent                           # force offline
```

**A design rule: returning "nothing found" always beats returning the wrong thing.**
An early version matched the offline corpus with `any()` (any keyword in the query
counted as a hit), so "OpenAI strategy" matched the "nvidia gpu strategy" entry and
the model nearly answered an OpenAI question with NVIDIA material. It uses `all()`
now, with a regression test watching it. For the same reason, a failed search in
`web` mode must return `ERROR:` rather than nothing — otherwise the model reads "the
tool is broken" as "this does not exist in the world".

## Two OpenAI backends

| | `--api responses` (default) | `--api chat` |
| --- | --- | --- |
| Tool definitions | flat `{"type":"function","name":...}` | nested `{"function":{...}}` |
| One turn of output | an `output` list (reasoning item + several function_calls) | a single assistant message |
| Call id | `call_id` | `id` |
| Result shape | `{"type":"function_call_output",...}` | a `role="tool"` message |
| Reasoning state | **preserved across tool calls** | discarded every turn |

All four differences are sealed inside the two classes in
[`model.py`](src/mini_agent/model.py); the control flow in
[`loop.py`](src/mini_agent/loop.py) did not move. The loop only needed two
generalizations: `state.messages.extend(reply.items)` (a turn can produce several
entries) and letting the backend's `tool_result_item()` decide the result shape.

Measured on the same task (gpt-5-mini): the responses context carries an extra
`reasoning` entry that goes back out with the next request; the chat side has no
equivalent. OpenAI's migration guide reports roughly +3% on SWE-bench with the same
prompt.

## The checklist

Ask for "research X, **then email me the result**" and an agent will happily do the
first half. A real run did exactly that: excellent research, no email, `status: done`,
and it stopped at turn 6 of 14 with 97% of the budget unspent. It was never a resource
problem — **nothing was keeping track of the second half of the request**.

So the goal is decomposed once at the start (one extra model call, `--no-plan` to skip
it) into 1-5 action items, and from then on:

```
[checklist] 1. research the Model Context Protocol | 2. write a summary | 3. email it to a@b.c
```

- Every turn's `[run status]` line carries the checklist with `[x]` / `[ ]` marks, so
  the open items stay in front of the model.
- The model ticks items off with the `update_todo` tool — `done`, or `blocked` with a
  reason. Blocked counts as settled: it stops being outstanding but keeps the reason,
  so the final answer can say what was left and why.
- **The loop refuses to finish while an item is untouched.** If the model stops calling
  tools with items still open, it gets one push-back (`[completion check]`) listing
  them, and then the answer stands either way. Once, never a loop.
- The forced wrap-up turn names the unfinished items too, so a run that ran out of
  resources says plainly what it never did.

This is the same lesson as the run-status line, one level up: **the model did not know
it had missed something, because nobody was remembering.** Keeping the list in
`AgentState` is what lets the loop remember on its behalf.

## The human-approval gate

Read-only tools just run. Tools with **external, irreversible side effects**
(`send_email` in this repo) pass through a gate first:

```python
@tool(description="...", parameters={...}, requires_approval=True)
def send_email(to, subject, body): ...
```

Three `--approve` policies:

| Value | Behaviour |
| --- | --- |
| `auto` (default) | ask you when there is a terminal (prints tool, arguments, description, waits for y/N); **deny when there is not** |
| `deny` | always deny |
| `allow` | always allow (explicit yolo mode) |

**"Nobody is watching, so allow it" is the most dangerous default there is** — the
run that goes wrong is precisely the unattended one (CI, cron, batch jobs). So `auto`
denies when it cannot find a TTY, instead of choosing convenience.

A denied call **still gets a result message** (same as a throttled one, or the
protocol breaks). It tells the model: not executed, do not re-send, take another
route or say in the answer that the user must do this step. Read-only tools **never**
carry the flag — ask too often and people go numb, and numb people approve with
their eyes closed.

**The gate is where authorization happens, so the model must attempt the call.** An
earlier version of the prompt offered "or state in your final answer that this step is
for the user" as an alternative, and the model took that exit *before ever trying* —
it wrote a draft email and asked for permission in an answer nobody reads before the
run ends. The prompt now spells out the order: call the tool, and only if the call
comes back denied take another route. Two live runs confirm the change: `send_email`
is now attempted (and denied by policy), then marked blocked with the reason.

Trajectory eval counts `denied` and `throttled` separately, plus
`retried_after_denial`: re-sending an identical call after a denial means the model
did not read the denial.

## Two kinds of evaluation — do not conflate them

| | `evals.py` | `trajectory.py` |
| --- | --- | --- |
| Model | fake (scripted) | a real trajectory (`runs/*/state.json`) |
| Question | **is the machine broken?** message protocol, brakes, compaction cut points | **how well did this run go?** outcome, grounding, efficiency, honesty |
| Cost | zero | mechanical metrics free; the LLM judge costs per run |
| Verdict | must be 15/15 green | relative scores, for comparing two versions |

```bash
uv run python -m mini_agent.trajectory runs/20260826-xxxx           # mechanical, free
uv run python -m mini_agent.trajectory runs/* --judge --out r.json  # add the LLM judge
```

**Mechanical metrics** (deterministic; read these first): steps, tool calls, failures,
**duplicate calls**, throttled, denied, compactions, elapsed, tokens, cache hits,
whether a conclusion was actually delivered, **whether it asked the user back**
(the very first real run failed exactly this way), citation count,
**`unsupported_citations`: links that appear in the answer but in no tool result** —
the deterministic detector for invented citations — and two signals for half-finished
work:

- **`action_never_attempted`**: the goal used an action verb (send, email, submit,
  delete...) and no approval-gated tool was ever called. "Looks done, is not."
  A *denied* attempt does not count as a failure here: the model did its part and a
  human said no.
- **`asks_without_trying`**: it asked the user for authorization in the answer without
  ever attempting the call. Asking *after* a denial is what the prompt asks for, so
  the raw `asks_user_back` signal is kept separately and this is the sharper one.
- **`pending_todos`**: checklist items still open at the end.

**The LLM judge**: outcome / grounding / efficiency / honesty from 0-5, plus a
one-line verdict and "the single thing most worth fixing". If the JSON does not
parse, it says so instead of pretending to have scored.

One real comparison makes the point. The judge scored a run's grounding 3, reasoning
that "it cited four sources but the trajectory shows only one search, so the evidence
is missing". The mechanical check reported `unsupported_citations = 0` — all four
links did appear in that search's results. **The judge saw a 300-character excerpt;
the mechanical check saw the full text.** So the order is: trust the deterministic
metrics first, then listen to the judge's qualitative read.

## Persistence and resume

Every step writes the full state to `runs/<timestamp>/state.json` (temp file plus
rename, so no half-written files). **Saving every step rather than at the end is
deliberate** — save only at the end and the crash that most needed the data is the
one that leaves nothing.

```bash
uv run mini-agent --max-steps 1 --run-dir runs/demo "..."   # hits the ceiling and stops
uv run mini-agent --resume runs/demo --max-steps 3          # continues from turn 2
```

On resume, the command-line ceilings mean "**give it this much more**" (steps and
time already spent live in the state, so reusing them verbatim would hit the ceiling
again immediately). One easily-missed detail: **resume does not rebuild the system
message** — rebuilding changes the context prefix and voids every prompt-cache entry
earned so far. A test watches that byte for byte.

What gets saved is also the input to trajectory eval: full messages, every tool
call's arguments and result, the spend, the final state. Reviewing a run used to mean
guessing from the 120 characters the terminal printed.

## Prompt caching

The caching is OpenAI's job. Ours is two things:

1. **Do not dirty the prefix.** The per-turn `[run status]` line is always appended
   at the **end**, the system message never changes, and `--resume` reuses the
   original system message.
2. **Say which requests belong together**: `prompt_cache_key` is a hash of the system
   prompt, so separate runs with the same configuration reuse each other's cache.

Cached input tokens bill at **a tenth**, `estimate_cost()` accounts for them
separately, and `cache_hit` in `state.snapshot()` reports the rate.

Measured (gpt-5-mini, 5 turns with real search): `"cache_hit": "38%"`. Short tasks
showing 0% is normal — a prefix under ~1024 tokens never enters the cache at all.

## Context management

`state.messages` only grows. After 20-30 turns three things go wrong together: the
context window overflows, every turn is resent at full price, and attention is
diluted by stale output. Two mechanisms, and **externalizing comes before
compacting**:

**1. Externalize** (`--run-dir`, default `runs/<timestamp>`, `off` to disable).
Tool results over 2000 characters go to a file; the context keeps the first 600
characters, the path, and a line saying "read that path with read_file for the full
content". **Nothing is lost**, and the cost of fetching it is paid only when needed.

```
[externalized] search_web returned 2022 characters; saved to a file, only an excerpt stays in context
```

**2. Compact** (`--context-limit`, default 30000 tokens). If the context is still
over the limit, an earlier slice is summarized and replaced by one
`[context summary]` message. The system prefix (or prompt caching dies), the original
goal and the last 8 entries are kept. The summarizer prompt explicitly demands
"verified facts + source links + what was tried and failed" — the last one matters
most, or the agent walks into the same dead end again after compaction.

**The dangerous part of compaction is not summary quality, it is the cut point.**
Split a tool call from its result and the next request fails with a 400. So
`safe_cut_points()` only cuts where nothing is dangling — the same scan as the
ordering invariant in evals. If there is no safe cut, it does not compact. One eval
case watches "the protocol survives compaction".

The decision uses the **real token count from the model** (`usage.input_tokens`),
falling back to a character estimate (CJK at 1.5 chars/token, everything else at 4)
only when there is none.

## Parallel tool execution

Multiple tool calls from one turn run **in parallel** (thread pool) and are fed back
**strictly in the original order**. Parallelism is exactly what breaks the two
message-protocol invariants, so both are pinned by tests: results in `tool_calls`
order, and exactly one result per id.

`--tool-timeout` (default 30s) covers a single call: on timeout the model gets an
`ERROR:` result and the loop continues instead of the whole turn wedging. Honest
caveat: Python cannot kill a stuck thread, so after a timeout that thread is left to
finish on its own — real isolation would need a subprocess. If less time remains than
the timeout, the remaining time wins (the time brake reaches tools too).

**Measured** (three real searches): serial 5.05s, parallel 3.64s — **1.39x**. The
modest speedup is worth recording: the bottleneck was not the network but **our own
search throttle**. At a 1.5s interval the parallel run took 8.3s; with no interval,
4.7s. Settled on 0.5s (six back-to-back searches, zero failures) and left the rest to
backoff retries. This only opens up properly with a real search API (roadmap #11).

## Four brakes

A model can fire ten searches in one turn, or bounce between tools until the budget
is gone. The loop has four independent guards:

| Guard | Flag | Default | On trigger |
| --- | --- | --- | --- |
| Turns | `--max-steps` | 8 | `status="max_steps"`, stop |
| Budget | `--budget` (USD) | 0.05 | `status="out_of_budget"`, stop |
| Wall clock | `--deadline` (seconds) | **600 (10 min)**, 0 = unlimited | `status="out_of_time"`, stop |
| Tool calls per turn | `--max-tool-calls` | 3 (0 = unlimited) | run the first N, **push the rest to the next turn** |

Money and time measure different things: **dollars measure model compute, time
measures human waiting.** Search throttling (0.5s spacing), backoff retries (up to
7s) and slow networks burn time without burning money, and only the time brake
catches them. The per-turn `[run status]` line puts **the tightest** brake in front
of the model, so it does not keep digging while watching the most generous one (in a
real run it was misled by "91% of the budget left" and never noticed its steps were
gone).

Time is only checked **between turns**; a single wedged tool call is what
`--tool-timeout` is for.

Hitting a ceiling is not the same as giving up — see "A brake must do more than
stop" below.

The fourth brake has a detail that must be right: **a call that was held back still
gets a `role="tool"` message**, saying "the per-turn limit is reached, not executed,
send it again next turn". Drop them instead and those `tool_call_id`s have no
results, so the next request fails with a 400 — the semantics here are "refused", not
"ignored". Reading that, the model converges on its own and squeezes ten queries into
the few that matter. `evals.py` pins this behaviour, and `state.snapshot()` reports
`tool_calls` (actually run) and `throttled` (held back) separately.

### A brake must do more than stop

Lesson from a real run: the model spent all 8 turns searching and produced not one
line of conclusion — ten searches paid for, nothing to show. So the brakes also have
to **unload the car**. Two mechanisms:

1. **No tools on the final turn**: when `state.step >= max_steps` the tool list goes
   out empty. Wording can be ignored, an empty list cannot — the model's only
   remaining option is to talk.
2. **A forced wrap-up turn**: when a ceiling is actually hit, ask once more, again
   with no tools, and require "conclusion + confidence + unverified items" from what
   is already there. `salvaged` is only set to true if something was actually
   rescued.

Each turn also appends a `[run status]` message telling the model which turn it is on
and how much budget is left — it cannot choose between digging and wrapping up
without knowing where it stands. That line is appended at the **end** rather than
written into the system prompt; otherwise the context prefix changes every turn and
prompt caching is void.

```bash
uv run mini-agent --live --max-tool-calls 2 "research OpenAI's funding and competition"
```

## The three traps in the control loop

All three are marked in the code and covered by `evals.py`:

1. **Order**: the assistant message carrying `tool_calls` must be written back into
   `messages` **before** the tool results. Reversed, the next API call errors.
2. **Count**: a turn may contain several `tool_call`s, and **every `tool_call_id`
   needs** its own `role="tool"` message. Miss one and the next request is a 400.
3. **Termination**: there is no mysterious `check_completion()`. A run ends when the
   model stops requesting tools (i.e. it gave a final answer), or when a ceiling is
   hit.

And: when a tool fails (bad JSON, wrong arguments, an exception inside the tool) the
error text is **returned as the tool result** so the model can correct itself. That
is not defensive programming — it is the whole reason the loop exists.

## The two bugs in the original notes

`NOTES.md` keeps the pre-refactor pseudo-code, including:

```python
result = fn(**item.arguments)   # 1. arguments is a JSON string, not a dict
                                # 2. the result never goes back to the model, and there is no loop
```

(1) is a missing `json.loads()`; (2) is the absence of appending the tool result and
asking the model again — which makes that snippet a single function call, not an
agent. The difference is the 40 lines in `loop.py`.

## Where this sits today

The core is not dated: an agent in 2026 is still this loop. The engineering layer has
come a long way from where it started — Responses API, prompt caching, context
management, parallel execution, persistence and resume, an approval gate and
trajectory eval are all in. What is still missing: MCP, subagents with isolated
context, and a serious search backend.

What is missing, why it matters, and in what order to add it — see
[docs/roadmap.md](docs/roadmap.md).
