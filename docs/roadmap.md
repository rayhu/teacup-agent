# teacup-agent roadmap

**Baseline assessment (2026-08-25)**: the core is not dated; the engineering layer
was roughly where the field stood in late 2023 / early 2024.
**Progress**: #1-#10, #12, #13, #15, #17, #18 and #19 are done (except the
fine-grained permissions part of #6). #14 is half done: `docs/threat-model.md`
exists, but `read_file`'s root is still `Path.cwd()`, not an explicit project root
captured at startup. Five items that were never on the roadmap were added after
reviewing real runs (see "Field patches" at the end).
Next up: the remaining half of #14 (root capture); then #11 (a better search
backend) and #16 (multi-provider price overrides, a native Anthropic path), both
scoped but not started.

The loop `LLM -> tool call -> tool result -> LLM` is still the core of every agent in
2026, and not one line of [`loop.py`](../src/teacup_agent/loop.py) is "old technology".
What was missing is the whole layer around it.

Subsystem-level rationale lives in [design-notes.md](design-notes.md); this file is
about what is still missing.

This roadmap is ordered by **payoff divided by cost**. Each item says what to change,
what counts as done, and where to read more. You do not have to do all of it — as a
learning skeleton the repo is fine as it is, and every layer added makes those 40
lines a little harder to see.

---

## P0 - payoff from changing one class

### 1. Move to the Responses API — DONE (2026-08-25)

**Before**: `OpenAIModel` in [`model.py`](../src/teacup_agent/model.py) used Chat
Completions, throwing away the model's reasoning each turn and pushing only the final
message back into `messages`.

**The cost**: for reasoning models like gpt-5 the reasoning state cannot survive
across tool calls. OpenAI's migration guide reports about +3% on SWE-bench with the
same prompt, and new capabilities (hosted tools such as built-in web_search and
code_interpreter) only land on Responses.

**How**: touch only `OpenAIModel`; `loop.py` and `state.py` do not change — that is
the payoff for abstracting the model as `complete(messages, tools) -> Reply`. Key
points:

- Tool definitions are **flat**: `{"type": "function", "name": ..., "parameters": ...}`,
  without the extra `function` nesting Chat Completions uses.
- Output is the `response.output` list; tool calls have `type == "function_call"` and
  use `call_id`.
- Tool results go back as
  `{"type": "function_call_output", "call_id": ..., "output": ...}`, not a
  `role="tool"` message.
- `previous_response_id` chains the context and preserves reasoning state.

**Caveat**: this asks more of the normalized `Reply` — `Reply.message` assumes one
chat message, and may need to become "the things to append back to the context".
That is the one design cost of this change.

**Definition of done**:
- copy `tests/test_model.py` into `test_responses_model.py` and verify parsing with a
  fake client;
- all evals still green (they only depend on the `Model` interface);
- both backends selectable with `--api {chat,responses}`, same task working on each.

**What actually shipped**:

- A new `ResponsesModel` alongside `OpenAIModel`, selected with
  `--api {responses,chat}`, **defaulting to responses**.
- The loop structure did not change, but two shape differences were factored out:
  - `Reply.message` -> `Reply.items` (a turn can produce several entries: reasoning
    plus multiple function_calls), so the loop does
    `state.messages.extend(reply.items)`;
  - the tool-result shape is decided by the backend (`Model.tool_result_item`):
    `role="tool"` for Chat, `function_call_output` for Responses.
- The ordering invariant `tool_results_follow_their_call()` in `evals.py` now
  recognises both shapes.
- New `tests/test_responses_model.py`: fake-client parsing plus the whole loop driven
  in the Responses shape.

**Measured** (gpt-5-mini, same task on each backend, about $0.002 total):

```
### responses                    ### chat
  0. system                        0. system
  1. user                          1. user
  2. reasoning      <- reasoning    2. assistant (tool_calls=1)
  3. function_call                 3. tool
  4. function_call_output          4. assistant
  5. message
```

Entry 2, `reasoning`, goes back out with the next request — that is where the gain
comes from, and Chat has no equivalent.

**In hindsight**: the predicted design cost (`Reply.message`'s shape assumption) was
indeed the only thing that had to change; the control flow in `loop.py` did not move.
The `Model` abstraction held.

**Reference**: <https://developers.openai.com/api/docs/guides/migrate-to-responses>

---

### 2. Prompt caching — DONE (2026-08-26)

**Before**: a long system prompt plus an ever-growing history, resent at full price
every turn.

**How**: keep the message prefix stable (never splice changing content into the
system prompt — note that `memory.recall()` is spliced in there, which is fine as
long as the memory itself is stable; do not start appending timestamps). Then split
cached and uncached tokens in `Reply.cost`.

**Definition of done**: run the same task twice and the second run's
`remaining_budget` drops noticeably less, with the snapshot reporting cache hits.

**What shipped**: `PRICES` became a triple (input / cached input / output) with cached
input billed at a tenth; both backends dig `cached_tokens` out of usage (Chat under
`prompt_tokens_details`, Responses under `input_tokens_details`); `snapshot()` reports
`cache_hit`. `prompt_cache_key` is a hash of the system prompt, so runs with the same
configuration reuse each other's cache.

**Measured**: gpt-5-mini on a 5-turn task with real search, `cache_hit: 38%`.
**One trap**: the first measurement was 0%, because the first request was about 972
tokens — **below OpenAI's ~1024-token threshold for caching at all**. Short tasks
showing 0% is normal, not a bug.

---

### 3. Retries, backoff, timeouts — DONE (2026-08-26)

**Before**: any exception from a model call in
[`loop.py`](../src/teacup_agent/loop.py) set `status="error"` and ended the run. One 429
or network hiccup threw away everything.

**How**: wrap the model call in exponential backoff (retry on 429/5xx/timeouts, not on
4xx) and count retries in the state for observability. Do not confuse this with
`max_steps`: **a retry is not a step**.

**Definition of done**: a unit test where a fake client raises 429 twice then
succeeds; the loop completes normally and `step` is unchanged.

**Search side**: measured, DuckDuckGo rate-limits after 4-5 rapid queries — and an
agent loves to fire several per turn. [`tools.py`](../src/teacup_agent/tools.py) now has
a minimum interval plus backoff retries (1s/2s/4s), and **when retries are exhausted
it reports `ERROR:` instead of silently degrading to the corpus's "nothing found"** —
disguising "the search broke" as "there is nothing" makes the model conclude the fact
does not exist.

**Model side**: `complete_with_retries()` backs off on 429/5xx/network errors (1s,
2s) and re-raises 4xx immediately rather than wasting time. The key detail:
**a retry is not a step** — steps measure how many decisions the model made, and one
rate limit should not eat its thinking allowance. Per-call timeouts also landed
(`--tool-timeout`, default 30s).

---

## P1 - what decides whether long tasks are possible

### 4. Context engineering (compaction / externalization) — DONE (2026-08-26)

**Before**: `state.messages` grew without bound. 20-30 turns would blow the context
window, and everything got more expensive and less focused as it grew.

**How**, two things together:

- **Compaction**: past a token threshold, summarize the early tool results and replace
  the originals. Keep the last N turns verbatim, plus the goal and key conclusions.
- **Externalization**: write large tool results (page text, file contents) into
  `runs/<id>/` and keep only "excerpt + file path" in the context; the model fetches
  detail with `read_file`. This beats any summarization algorithm.

**Definition of done**: a task that needs 15+ turns finishes without blowing the
context; `state.snapshot()` shows the token count dropping across a compaction; no key
conclusion is lost in the process (which trajectory eval has to verify, see #7).

**What shipped**: a new [`context.py`](../src/teacup_agent/context.py).

- Externalization: tool results over 2000 characters go into `runs/<timestamp>/`, and
  the context keeps 600 characters, the path, and "read it back with read_file". A real
  2022-character search result came down to roughly 900 tokens of context, with the
  full text on disk.
- Compaction: over `--context-limit` (default 30000), the early history becomes one
  summary message. The system prefix, the original goal and the last 8 entries stay.
- **The critical detail is the cut point, not summary quality**: `safe_cut_points()`
  reuses the ordering-invariant scan and only cuts where nothing is dangling; if there
  is no safe cut it does not compact. Splitting a call from its result costs a 400 on
  the next request.
- The decision prefers the model's real `usage.input_tokens` (now exposed on `Reply`)
  and only falls back to a character estimate.
- `state.snapshot()` gained `context_tokens` and `compactions`.

**Left open**: compaction itself costs a model call and currently re-summarizes from
scratch every time; hierarchical summaries (summaries of summaries) are not done.

---

### 5. Parallel tool execution — DONE (2026-08-26)

**Before**: the model fired three searches in one turn and
[`loop.py`](../src/teacup_agent/loop.py) ran them one at a time. With real web search
that became the most obvious wall-clock waste.

**How**: run tool execution through `asyncio.gather` (or a thread pool, since most
tools are synchronous IO).

**Invariant that must hold**: results are fed back **in `tool_calls` order** and every
`tool_call_id` has its result — the thing
`tool_results_follow_their_call()` in `evals.py` pins down. Concurrency is the change
most likely to break it, so check that case first.

**Definition of done**: three fake tools that sleep 1s each take ~1s per turn rather
than 3s, with all evals still green.

**What shipped**: `execute_calls()` runs a thread pool and refills in the original
order; in `tests/test_parallel.py` three 0.3s fake tools finish in under 0.6s, with
order and invariants verified. Per-call timeouts came along too (`--tool-timeout`,
default 30s), returning an ERROR result instead of wedging the whole turn — the "one
stuck tool" debt left over from the time brake. The search throttle's global counter
in `tools.py` got a lock (under parallelism it was a race).

**Surprise finding**: real searches only sped up 1.39x (5.05s -> 3.64s), because the
bottleneck was not the network but the interval we added ourselves to dodge
DuckDuckGo's rate limiter. Measured: 8.3s in parallel at a 1.5s interval, 4.7s with no
interval. Settled at 0.5s. **The ceiling on this optimization is set by #11 (a better
search backend), not by concurrency** — measure before optimizing, or you push hard in
the wrong place.

---

### 6. Human-in-the-loop approval and permission tiers — DONE (2026-08-26)

**Before**: every tool was read-only or harmless, so there was no approval step. Add
`send_email`, `shell`, or file writes and that is a hole with real consequences.

**How**: give `Tool` a `requires_approval: bool` (or a danger level). Emit an
`approval_required` event before executing and confirm interactively in the CLI; in
non-interactive mode deny by default and hand "denied" back as the tool result so the
model can take another route.

**Definition of done**: a tool marked dangerous never executes in automatic mode, and
the loop keeps running with the message protocol intact.

**What shipped**: `Tool.requires_approval` plus a `run(approve=...)` callback,
defaulting to `deny_all`. CLI `--approve auto|deny|allow`: auto asks when there is a
terminal and **denies when there is not**. A new example tool `send_email` (writes
outbox.jsonl, sends nothing) — the last entry in the original notes' tool list, and
the first one that needs a gate: a read-only tool going wrong wastes one call, this one
going wrong means the mail has left.

**Deliberate choices**:
- Deny by default rather than allow. "Nobody is watching, so allow it" is the most
  dangerous default there is; the run that goes wrong is the unattended one.
- The approval check runs serially **before** the thread pool (it either asks a human
  or denies, and cannot be parallelized).
- A denied call still gets a result message that says "not executed, do not re-send".
- Read-only tools never carry the flag: ask too often and people go numb, and numb
  people approve with their eyes closed.
- Trajectory eval gained `denied` and `retried_after_denial` (re-sending an identical
  call after a denial = the denial was not understood).

**Left open**: finer permission tiers (argument-aware rules, e.g. "only addresses on
this allowlist").

---

## P2 - from "it runs" to "it can be trusted"

### 7. Trajectory eval — DONE (2026-08-26)

**Before**: the cases in [`evals.py`](../src/teacup_agent/evals.py) pin down whether the
loop protocol is correct — necessary, but the lowest tier, equivalent to unit tests.

**What an agent eval looks like in 2026**: it scores the **whole trajectory**, not
just the final answer. The same correct answer reached in 3 clean steps and in 12
flailing ones are not worth the same. Common dimensions: outcome correctness, tool-use
correctness (right tool, right arguments), efficiency (steps / tokens / cost), and
path safety, scored with an LLM-as-judge rubric.

**How**:
- build a small task set (10-20 items), each with expected conclusions and a
  reasonable tool-use path;
- keep `ScriptedModel` for the protocol layer and use a real model for the trajectory
  layer (that part costs money, so it gets its own command);
- add a judge: hand `state.trace` and the final answer to a model with a rubric;
- persist results under `runs/` so two versions can be compared.

**Definition of done**: change the system prompt and be able to say in numbers whether
it got better or worse, instead of guessing.

**What shipped**: a new [`trajectory.py`](../src/teacup_agent/trajectory.py), taking the
`runs/*/state.json` files persisted by #8 as input. Two layers:

- **Mechanical metrics** (free, deterministic): steps, tool calls / failures /
  **duplicates**, throttled, compactions, elapsed, tokens, cache hits, `delivered`
  (a real conclusion or nothing), `asks_user_back` (handing the question back — the
  shape of the very first real failure), and `unsupported_citations` (links in the
  answer that appear in no tool result).
- **The LLM judge**: outcome / grounding / efficiency / honesty from 0-5, plus a
  verdict and the single thing most worth fixing. Unparseable JSON is reported as an
  error rather than a fake score.

**The most interesting moment in testing**: the judge scored one run's grounding 3,
saying "it cited four sources but only searched once, so the evidence is missing",
while the mechanical check computed `unsupported_citations = 0` — all four links were
in that search's results. **The judge saw a 300-character excerpt; the mechanical
check saw the full text.** Conclusion: **trust the deterministic metrics first, then
the judge's qualitative read** — a judge is good at "how good is this", not at fact
checking.

**Left open**: a fixed task set and automated cross-version regression comparison
(right now it is one run, one score).

**Reference**: <https://qaskills.sh/blog/agent-trajectory-evaluation-guide-2026>

---

### 8. Observability and resumability — DONE (2026-08-26)

**Before**: a finished run left nothing behind. A crash meant starting over, and
debugging meant scrolling the terminal.

**How**: persist each run (`runs/<timestamp>/`: messages, trace, spend, final state),
add structured logging / OpenTelemetry spans; one step further is checkpoint resume —
`AgentState` is already a dataclass, so JSON is enough to continue from where it
stopped.

**Definition of done**: Ctrl-C halfway through, then resume from the checkpoint
without redoing completed tool calls.

**What shipped**: a new [`persist.py`](../src/teacup_agent/persist.py). Every step writes
the whole `AgentState` to `runs/<timestamp>/state.json` (temp file plus rename, so no
half-written files), and `--resume` continues from there. Resume **does not rebuild the
system message** (rebuilding voids the prompt cache), and command-line ceilings stack
as "give it this much more". `run_dir=None` disables persistence — the path evals and
unit tests take, so they leave nothing in the repo.

**A real bug caught along the way**: writing the resume test revealed that if the model
still emits tool calls during the forced wrap-up turn, those calls have **no result
messages** and the protocol breaks — the next request (including a later `--resume`)
fails with a 400. The wrap-up turn now fills every dangling call with "the run has
entered wrap-up, tools are unavailable", and an eval case watches it. This class of bug
only surfaces once you actually implement resume.

**Left open**: structured tracing such as OpenTelemetry spans.

---

## P3 - opening up the ecosystem

### 9. MCP support — DONE (2026-08-26)

**Now**: tools are hardcoded Python functions, and adding one means editing
[`tools.py`](../src/teacup_agent/tools.py).

**Why it is worth it**: MCP is the de facto standard for tool integration in 2026.
Connect to it and off-the-shelf servers (filesystem, databases, GitHub, browsers) work
immediately instead of being written one at a time. The 2026 roadmap focuses on
transport scalability, agent-to-agent communication, governance and enterprise
readiness.

**How**: write an `mcp_tools.py` that pulls the tool list from an MCP server and turns
each entry into the existing `Tool` structure in `REGISTRY`. Because `tools.execute()`
takes "a name plus a JSON string of arguments", which matches MCP's call shape
naturally, `loop.py` still does not change.

**Definition of done**: connect to a local MCP server at startup, see its tools in
`tools.specs()`, and have the model call them normally.

**What shipped**: a new [`mcp_tools.py`](../src/teacup_agent/mcp_tools.py) plus `--mcp <config>`,
which defaults to `./mcp.json` when that file exists and to no MCP at all when it does not.
Off by default because connecting starts third-party processes and inflates the context
prefix of every request; auto-loaded once the file exists because at that point the file
*is* the consent.
`loop.py` did not change, exactly as predicted — MCP's "name + JSON arguments" call shape
lines up with `tools.execute()`.

**Which protocol revision**: **2026-07-28**, the stateless one. It removed the
`initialize`/`notifications/initialized` handshake and the `Mcp-Session-Id` header, so a
tool call is one stateless RPC; the connection-lifecycle code that used to dominate an MCP
client is simply absent. We use the official Python SDK v2, which also speaks the earlier
handshake-based revisions — verified against the real `mcp-server-fetch`, which is still
on the old protocol: the SDK probed with `server/discover`, the server rejected it, the
client fell back, and the call succeeded.

**Design decisions**:
- Tools register as `server__tool`, sanitized — the spec warns that aggregating clients
  will hit name collisions, and OpenAI function names allow only `[A-Za-z0-9_-]`.
- `annotations.read_only_hint` opens a tool; anything else is gated, **including servers
  that annotate nothing** (the common case). The spec says annotations are untrusted
  unless the server is, so `"approve": "none"` is how a config *states* trust.
- A per-server `tools` allowlist, because every schema sits in the context prefix of
  every request.
- Server stderr is hidden by default: legacy servers print a wall of pydantic validation
  errors when probed with `server/discover`. `"stderr": "show"` when a server will not
  start.
- One background event loop owns every session, so the SDK's async API stays invisible to
  the rest of the codebase.
- A server that fails to connect is reported and skipped, never fatal.

**Verified**: 20 tests against a real MCP server over stdio
([`tests/fixtures/demo_mcp_server.py`](../tests/fixtures/demo_mcp_server.py), no network),
plus a live run where gpt-5-mini used `fetch__fetch` to read the spec page and answer
with a citation.

**Left open**: resources, prompts, subscriptions, tasks; elicitation (a
`resultType: "input_required"` result comes back as an `ERROR:` the model can route
around); remote servers over Streamable HTTP work by URL but OAuth is untested.

**Reference**: <https://modelcontextprotocol.io/specification/2026-07-28/changelog> ·
<https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/>

---

### 10. Subagents / context isolation — DONE (2026-08-26)

**Now**: everything shares one context.

**How**: orchestrator-worker. The main agent delegates subtasks to subagents with their
own contexts, and a subagent returns only its **conclusion** (not every page it read).
This is also the most effective form of context compression.

**Definition of done**: for a task that needs five sources, the main context uses
significantly fewer tokens than the single-context version with no loss of quality.

**What shipped**: [`subagent.py`](../src/teacup_agent/subagent.py) plus `--subagents`, off by
default. `loop.run(subagents=True)` registers a `delegate` tool for the duration of the
run; calling it starts a child `loop.run()` with a blank context, a slice of the parent's
remaining budget, its own step ceiling, and every tool except `delegate` itself.

**Design decisions**:
- One level of delegation, enforced by omitting the tool from the child's list rather than
  by a depth counter: a recursion here is a recursion that spends money.
- The child is out of context but not invisible: its trace and `state.json` land in
  `runs/<timestamp>/sub01/`, its events are forwarded with a `[sub1]` marker, and its cost
  and tokens are charged to the parent.
- Budget share is read at call time, so a parent that has already spent most of its budget
  cannot fund an expensive child.
- The tool is registered per run, not globally, so its schema is not in the prefix of
  every request for runs that cannot use it.
- `Tool.timeout` was added for this: a child run is not a page fetch, and the loop's
  30-second per-call default would kill it. Per-call deadlines are now absolute, so
  waiting on parallel calls one after another no longer adds their timeouts together.

**Verified**: 11 tests plus an eval case. The sharpest one gives a fake tool that returns
9,000 characters: the child reads it, the parent's context never contains a byte of it,
and the child's `state.json` on disk does.

**Measured** (gpt-5-mini, two spec pages via MCP fetch): raw characters entering the
parent context fell from 4,700 to 916 and its final `context_tokens` from 6,515 to 4,333,
while total input tokens rose from 28,866 to 38,465 and cost from $0.0110 to $0.0148.
Isolation is not free: each child carries its own system prompt and re-reads material the
parent would have had anyway. The trade pays only when the parent would otherwise carry
that bulk for the rest of a long run, since context spent on turn 2 is still being resent
on turn 12. See [design-notes.md](design-notes.md#measured-including-the-part-that-is-not-a-win).

**Also learned**: on the first attempt the model never called the tool at all
(`subagents: 0`) even though it was available. An available tool is not a used tool, the
same lesson as the email that never got sent.

**Left open**: several subagents in parallel (they run one per tool call today, so the
per-turn cap applies), and passing curated context down instead of a blank slate.

---

### 11. A better search backend

**Now**: `search_web` scrapes DuckDuckGo through ddgs — free and key-less, but average
in both quality and stability.

**Options**: the model's own hosted web search (via the Responses API, see #1), or a
dedicated agentic search API. The interface does not change; only the inside of
`search_web` does — the three-mode structure (auto/web/offline) is already there.

---

### 12. Agent Skills — DONE (2026-08-26)

**Why**: the system prompt is static context, so every token in it is in every request
whether the task calls for it or not. That is right for rules the agent must never forget
and wrong for knowledge it needs once an hour. Measured before this landed: 842 tokens of
system prompt plus 643 of tool schemas, resent every turn, and both only grow as servers
and tools are added.

**What shipped**: [`skills.py`](../src/teacup_agent/skills.py), `--skills <dir>` defaulting
to `./skills` when it exists, and two real skills (`web-research`, `long-document`).

Progressive disclosure, three levels:

1. **Startup**: name and one-line description spliced into the system prompt, about 25
   tokens per skill.
2. **On match**: the model calls `load_skill`, and the procedure arrives as a tool result,
   which is dynamic context by construction.
3. **Deep reference**: a skill points at files beside it, read with `read_file` only if
   needed. No new mechanism.

**Measured**: the catalog block costs 180 tokens (about 42 of it marginal per skill) and
covers 920 tokens of procedure, so static context per request is 1,665 instead of the
2,405 it would be with the bodies inlined. Loading the same skill twice returns a pointer
rather than the text, because resending a 600-token procedure would undo the saving the
feature exists for.

**The live test that mattered**: with gpt-5-mini on a research task and no instruction to
use skills, the first catalog wording ("procedures you can load when the task matches")
produced `loaded_skills: []`. Rewritten as an instruction ("if the task matches one of
these descriptions, call load_skill first and follow it"), the same model loaded
`web-research` as its first action. A control task, `(3200-450)*0.6`, loaded nothing.
Third time in this repo that an available capability stayed unused until it was phrased
as an obligation.

**A bug that run exposed**: the skill body is 2,420 characters and the externalizer moves
anything over 2,000 to a file, so the model received the first 600 characters of the
procedure it was meant to follow. Tool results are either raw material (an excerpt plus a
path is right) or instructions (truncation defeats the point). `Tool.externalize` now
marks the difference and `load_skill` sets it to `False`.

**Design decisions**:
- A skills directory is the opt-in, the same convention as `mcp.json`: its metadata costs
  prefix tokens and its contents are instructions the model will follow.
- A skill without a description is skipped. Without one the model cannot know when to load
  it, so it is pure cost.
- Frontmatter is parsed by hand rather than adding a YAML dependency to read two fields.
- Skills are knowledge, not code. They are text the model follows, which makes the skills
  directory a trust boundary in the same way tool descriptions are.

**Left open**: nothing moves *out* of the system prompt yet. The recency and
query-anchoring rules (roughly 300 tokens) only matter for research and are the obvious
candidates. The model does now load the skill on a research task, so the move is a real
option, but it needs its own before-and-after: those rules fix a failure that was
expensive to find, and one successful load is not proof of reliable loading.

---

### 13. Hooks — DONE (2026-09-03)

**Now**: every guardrail was hardcoded in the loop. The per-turn cap, the approval gate and
the forced wrap-up are all good behaviour, but a user who wants their own rule had to edit
`loop.py`. It is also the mechanism #14 needs: an allowlist and an output audit are both
hooks, and adding them as hooks means the loop does not grow a security section.

**What shipped**: [`hooks.py`](../src/teacup_agent/hooks.py) — a small registry of three
callbacks, loaded from a project-local `hooks.py` (the same opt-in-by-file convention as
`mcp.json`/`skills/`, wired via a new `--hooks` flag and `_resolve_hooks()` in `cli.py`,
mirroring `_resolve_mcp`/`_resolve_skills`):

- `before_tool_call(call) -> str | None` — a string vetoes the call (it becomes the
  call's `ERROR:` result, same shape a denial already gets); `None` allows it through.
  Wired into `execute_calls()` in `loop.py`, checked **before** the approval gate, so a
  vetoed call never reaches "ask for approval."
- `after_tool_result(call, result) -> str` — may rewrite the result that reaches the
  model. Wired in right before a result is emitted/externalized/traced.
- `approve_tool_call(call, spec) -> bool | None` — the one hook that can say **yes**
  with nobody watching, and only because the project itself declared that trust. Kept
  separate from `before_tool_call` on purpose: a veto hook can only refuse, this one can
  approve, which is a materially different kind of power and needed its own name and its
  own opt-in (`--approve hooks`, a new fourth policy alongside `auto`/`deny`/`allow` in
  `_make_approver`). Returning `None` (including "no `hooks.py` was loaded at all") falls
  through to `auto`'s own ask-if-there-is-a-TTY / deny-otherwise behaviour, so `--approve
  hooks` is safe to leave on even for calls the project's `hooks.py` never mentions.

**Failure handling is deliberately asymmetric**, because the three callbacks are not
symmetric risks: a broken `before_tool_call` fails **closed** (an exception becomes a
veto — a broken safety check must not silently stop being one); a broken
`approve_tool_call` fails to "no opinion" (also closed, since that already means deny
without a TTY); a broken `after_tool_result` fails to a no-op (it is a transform, not a
gate, so silence is the safe fallback — the same "a broken planner must never stop the
run" discipline `plan.py`/`reflect.py` already hold).

**A departure from the mcp.json/agent.yaml convention, stated on purpose**: those two are
gitignored because they can carry credentials. `hooks.py` carries policy, not secrets, and
an unattended run trusting it to approve calls is exactly the kind of change that should
be reviewed in version control, not hidden from it — so it is **not** added to
`.gitignore`. See `docs/threat-model.md`.

`hooks.example.py` demonstrates the mechanism end to end using the existing `send_email`
tool and zero new tools — exactly the example this item originally asked for
("send_email only to these domains"): `before_tool_call` refuses a recipient outside the
allowlist, `approve_tool_call` says yes for one that is on it, so an unattended run can
actually send the approved mail instead of every send being denied for lack of a TTY.

**Definition of done**: a project-local hook blocks a tool call by argument (not just by
tool name) without touching `loop.py`'s own code, and the veto reaches the model as a
normal `ERROR:` result — pinned by `tests/test_hooks.py`.

**Verification**:
```
uv run pytest                          # includes tests/test_hooks.py
uv run python -m teacup_agent.evals    # loop health, scripted model, free
uv run teacup-agent                    # offline demo unaffected (no hooks.py by default)
```

---

### 14. Threat model, and tool-call allowlists — half done (2026-09-03)

**Update**: `docs/threat-model.md` now exists, built alongside #13's hooks mechanism
(the argument-aware allowlist example below, `hooks.example.py`, is real and runnable).
The other half of this item's definition of done — `read_file` rooted at an explicit
project root instead of `Path.cwd()` — is still open; see "Definition of done" below.

**Now**: the docs state that web content and MCP tool descriptions are untrusted text the
model reads, and then nothing is done about it. Documenting a risk without marking the
boundary is worse than not raising it.

**Three exposures that exist today**, all verified rather than hypothetical:

1. **`read_file` can read `.env`.** Before the deny-list below, its only guard was
   "stay inside the project directory", and the project directory is where the API keys
   live. Combined with
   `send_email` under `--approve allow`, or simply with a model that quotes it into an
   answer, that is an exfiltration path built entirely from existing, intended features.
2. **MCP servers are third-party processes with our user's privileges.** The SDK passes
   them a minimal environment (`HOME`, `LOGNAME`, `PATH`, `SHELL`, `USER`, and notably
   *not* `OPENAI_API_KEY`), which is better than we would have managed by hand, but the
   child still has the filesystem and the network. `mcp.example.json` demonstrates a
   filesystem server pointed at `.`.
3. **`read_file`'s root is wherever the process was launched**, not the project.
   [`tools.py`](../src/teacup_agent/tools.py) opens with `root = pathlib.Path.cwd()
   .resolve()`. The traversal guard on top of it holds — `../../../etc/hosts` is
   refused — but the root it guards moves with the shell. Launched from `~`, every file
   under the home directory is "inside the project" except what the deny-list patterns
   happen to catch; launched from a subdirectory, the deny-list's root-relative entries
   (`.env`, `.git`, `.venv`) stop covering the tree the user thinks they are in.
   Measured by `chdir`-ing into a scratch directory and reading a file there: the
   traversal error fired for `../../etc/hosts`, and `notes.txt` in the new directory was
   returned in full.

**Where the field is**: the 2026 consensus on prompt injection is containment rather than
prevention. Assume an injection lands and make sure it cannot do much: minimum capability
per tool, authorization enforced outside the agent's reach, allowlists on tool calls,
output auditing, and process isolation (Seatbelt on macOS, bubblewrap on Linux) for
anything that executes code.

**On sandboxing — what is out of scope, and what only looks it.** This section used to
say "this agent executes no code, so there is nothing to isolate". Half of that is true,
and the half that is not is the more interesting half:

- **Our own tools execute no code**, and that was a deliberate choice at the smallest
  possible scale: `calculate` is a hand-written `ast` walker (`_eval_node`), not `eval`.
  So there is no exec surface of ours to isolate, and the rule stands unchanged — **a
  code-execution tool must not be added until a sandbox exists to run it in.**
- **But a stdio MCP server *is* code execution — someone else's, spawned by us.**
  `MCPTools.connect()` runs `spec["command"]` as a child process holding our user's
  privileges, the filesystem and the network. Exposure 2 above already says this; the old
  "nothing to isolate" line contradicted it. MCP isolation stays out of scope by **cost**
  — a Seatbelt or bubblewrap wrapper around the child command is real work and buys a
  teaching repo little — not because there is nothing there to isolate. If you fork this
  and point it at servers you did not write, that is the sentence to re-read.

**"Docker or Seatbelt?" is the wrong first question.** The word "sandbox" bundles four
properties that different tools buy separately:

| Property | The question it answers | Bought by |
| --- | --- | --- |
| Isolation | can it damage the host? | Seatbelt (`sandbox-exec` — `DEPRECATED` in its own man page, still working), bubblewrap, containers, microVMs |
| Reproducibility | does it see the same world every run? | an image plus a lockfile; Seatbelt gives none of this |
| Reversibility | can what it did be undone? | a git worktree, or a container over a *copied* tree — a bind-mounted one gives nothing |
| Credential and egress scope | which secrets and which network does the process hold? | none of the above |

Every exposure listed above lives in the last row, which is why the answer here is
deny-lists, approval gates and allowlists rather than a container. A container holding
`OPENAI_API_KEY` with open egress is a perfectly good exfiltration channel. These are
different axes, not strong and weak versions of one thing.

**Proportionate here** (this is a teaching repo, not a production runtime):
- ✅ **done**: a deny-list in `read_file` covering `.env*`, `mcp.json`, `memory.json`,
  `state.json`, key files and `.git` / `.ssh` / `.aws` / `.venv`, returning an `ERROR:`
  that states the rule is fixed so the model does not retry with another spelling.
  `runs/` needed a distinction rather than a blanket rule: externalized `.txt` results
  stay readable because the model already saw them, while a run's `state.json` does not,
  because it holds the system prompt and every result of that run. Both halves are
  tested, and the externalization round trip was verified end to end afterwards;
- **root `read_file` at an explicit project root instead of `Path.cwd()`** (exposure 3),
  resolved once at startup from the CLI's working directory and passed in, so the
  boundary is a stated fact of the run rather than a property of the user's shell;
- argument-aware allowlists as hooks (#13), e.g. "send_email only to these domains";
- a `docs/threat-model.md` that states plainly what is trusted, what is not, and what this
  repo does *not* defend against — including that a stdio MCP server is unsandboxed code
  execution — so a fork knows what it is inheriting.

**Definition of done**:
- `read_file` refuses a file that is outside the project root but inside the directory
  the process was launched from, with a test that pins it by launching from elsewhere;
- `docs/threat-model.md` exists and names MCP child processes as the one unsandboxed
  code-execution path.

**Reference**: <https://www.sysdig.com/learn-cloud-native/prompt-injection> ·
<https://www.firecrawl.dev/blog/ai-agent-sandbox>

---

### 15. Declarative agent config (`agent.yaml`) — DONE (2026-09-03)

**Now**: model, MCP servers, tools, skills and every runtime ceiling are one flat list of
`cli.py` flags. Fine for one invocation; not for describing *an agent* — something you'd
want to diff, copy between projects, or hand to someone else.

**What shipped**: [`agent_config.py`](../src/teacup_agent/agent_config.py) — dataclasses
(`ModelProfile`, `ToolsConfig`, `RuntimeConfig`, `AgentConfig`), a `load(path)` that reads
YAML with `${VAR}` expansion (env-only; secrets never live in the file, same convention
as `mcp.json`), and `build_model()`/`resolve_run_dir()` factories. `cli.py --config
<path>` is a **parallel track, not a merge**: every other behavior flag is ignored once
it is set. `agent.example.yaml` is the committed template; `agent.yaml` is gitignored.
`trajectory.py --config --judge-profile` reuses the same registry for the LLM judge's
model instead of a second, silently-diverging default (it was `--model gpt-5-mini`,
hardcoded).

**A bug the tests caught before anyone hit it**: PyYAML parses a bare `off`/`on` as a
Python **bool**, not a string (YAML 1.1's "Norway problem"). Three fields here use
`"off"`/`"on"` as string sentinels (`runtime.plan`, `runtime.run_dir`, `skills.dir`), so
`plan: off` written unquoted parsed to `False`, and every `== "off"` check downstream
silently took the wrong branch — `resolve_run_dir(False)` crashed outright, but
`skills.dir: off` would have quietly loaded skills anyway. `_normalize_off_on()` maps the
boolean back to the matching sentinel before anything compares against it. Full writeup
and a regression test: `docs/design-notes.md` and `tests/test_agent_config.py`.

**A capability that came free**: reaching an OpenAI-compatible endpoint (`base_url`) that
was scoped as part of #16 turned out to need no change to `model.py` at all —
`OpenAIModel`/`ResponsesModel` already accept a pre-built `client`, so `build_model()`
just constructs one with a custom `base_url`. #16 below is now only the price-table
override and a native second wire protocol.

**Verification**:
```
uv run pytest                          # 171 passed
uv run python -m teacup_agent.evals    # 20/20 passed
uv run teacup-agent                    # offline demo unaffected, still instant
```

---

### 16. Multi-provider models: price overrides and a native second protocol

**Now**: `#15` already reaches any OpenAI-compatible endpoint (vLLM, Ollama, OpenRouter)
via `base_url`, for free. What is left is smaller than originally scoped:

- **Per-profile cost override.** `estimate_cost()` looks prices up in a name-keyed table
  (`model.PRICES`) with a fallback for anything unrecognized — accurate for OpenAI models,
  a guess for everything else. A profile-supplied `(price_input, price_cached,
  price_output)` should take priority over the table when present, so cost tracking stays
  meaningful for a local or third-party model.
- **A native Anthropic Messages API path** (a genuinely different request/response shape,
  not just a different URL) — added the same way `ResponsesModel` was added beside
  `OpenAIModel`: a new class behind the unchanged `Model` Protocol, no loop change.

**Definition of done**: a model profile with `price_input`/`price_cached`/`price_output`
changes `state.snapshot()`'s cost accounting; an `AnthropicModel` class round-trips a
tool call through the Messages API shape with a test pinning its `tool_result_item()`.

---

### 17. Agent2Agent (A2A) client: delegate to a remote agent — DONE (2026-09-03)

**Now**: `subagent.py`'s `delegate` tool hands a subtask to a child **in-process** loop.
There is no way to hand a subtask to a *different* agent process, possibly on a different
machine, possibly not running teacup-agent at all.

**Why A2A rather than inventing a wire format**: the Agent2Agent protocol reached v1.0 in
January 2026, is Linux-Foundation-governed, and has an official Python SDK (`a2a-sdk`,
built on Starlette/JSON-RPC/SSE). This repo already has the precedent for "depend on the
official SDK rather than hand-roll the protocol" — `mcp_tools.py` wraps the official `mcp`
package the same way.

**What shipped**: [`a2a/client.py`](../src/teacup_agent/a2a/client.py)'s `A2AHub` —
given a peer's URL and a bearer token resolved from `api_key_env` (`agent.yaml`'s
`a2a.peers`, given real shape here: `agent_config.A2APeer`/`A2AConfig`), resolves the
Agent Card, submits a task, consumes the response to a terminal state, and flattens it to
a plain string. Every failure path (connection error, a remote task ending in
`TASK_STATE_FAILED`/`REJECTED`/`CANCELED`/`INPUT_REQUIRED`/`AUTH_REQUIRED`) becomes an
`"ERROR: ..."` string, never a raised exception — the same discipline `tools.execute()`
and `mcp_tools.py` already hold. One tool, `delegate_a2a(peer, task)`, registered in
`cli.py`'s `_main_config()` only when `a2a.peers` is non-empty (the same "do not cost
prefix tokens when unused" pattern as `--subagents`/skills), `requires_approval=True` by
default — an outbound call to a third-party, possibly-billed agent is exactly the side
effect AGENTS.md rule 4 exists for. Full design rationale (where the hub lives and why,
the async/sync bridge, an API-surface note about `a2a-sdk` v1.1.2's real shape versus what
blog posts describe): `docs/design-notes.md`.

**No `loop.py` changes were needed.** The approval gate already reads
`Tool.requires_approval` generically, so registering the tool with that flag set was
everything this item needed from the loop — resolving the "where does the hub live"
question the original scoping left open in favor of the MCP shape (`cli.py`-owned),
not the subagent shape (`loop.run()`-owned).

**Definition of done**: [`tests/fixtures/demo_a2a_server.py`](../tests/fixtures/demo_a2a_server.py)
(a real `a2a-sdk` `AgentExecutor`, wired into a Starlette app object) receives and answers
a task sent by `delegate_a2a` — driven through `httpx.ASGITransport`, so the test makes
zero real socket calls, one step more hermetic than MCP's subprocess-based fixture (A2A
has no stdio transport to subprocess over).

**Verification**:
```
uv run pytest                          # 203 passed
uv run python -m teacup_agent.evals    # 20/20 passed
uv run teacup-agent                    # offline demo unaffected, still instant
```

---

### 18. Agent2Agent (A2A) server: expose this agent to other agents — DONE (2026-09-03)

**Now**: this agent can only be driven by a human running `cli.py`. #17 lets it call
*out* to another agent; nothing lets another agent call *in*.

**The tension worth stating plainly**: this is, structurally, a service layer — a
long-lived HTTP process accepting inbound tasks — and "Deliberately not doing" below has
said "no multi-tenancy, service layer or web UI" since this file began. The resolution,
at two levels: the `a2a-server` optional dependency group (Starlette + uvicorn) is never
installed by plain `uv sync`, only `uv sync --extra a2a-server`; and `teacup-agent-serve`
is a second, entirely separate console script — `uv run teacup-agent` is unaffected
whether or not the extra is even installed.

**What shipped**: [`a2a/card.py`](../src/teacup_agent/a2a/card.py) builds an Agent Card
from `agent.yaml`'s `a2a.card` plus `skills.py`'s catalog (one line per skill is exactly
the grain an `AgentSkill` wants — not the raw tool registry, which would be noisy and the
wrong level of detail); [`a2a/server.py`](../src/teacup_agent/a2a/server.py)'s
`TeacupAgentExecutor` wraps the existing `loop.run(goal=<incoming task text>, ...)`
unchanged, via `asyncio.to_thread()` since the handler is async and `loop.run()` is not.
Reuses `cli._make_approver` **unmodified** for the approval gate: its `"auto"` branch
already denies when there is no TTY, always true under `uvicorn`, so a served agent is
gated by AGENTS.md rule 4 with no new approval code. `cancel()` raises
`NotImplementedError`, matching `a2a-sdk`'s own reference examples for "not supported" —
`loop.run()` has no cooperative-cancel hook, so faking cancellation would be worse than
refusing it.

**A concurrency limit stated rather than silently shipped**: `tools_mod.REGISTRY` is
process-global, and `skills.enable()`/`subagent.enable()` mutate it with no locking,
assuming one `loop.run()` at a time. Two inbound tasks arriving concurrently on a server
whose `agent.yaml` turns on skills or subagents would race on that state.
`TeacupAgentExecutor` holds an `asyncio.Lock` around each run for exactly this reason —
correct, at the cost of one served task at a time; real per-run tool isolation is a larger
change, out of scope here.

**Definition of done**: a real end-to-end check with two actual OS processes over a real
loopback TCP port (a `ScriptedModel` injected so it cost nothing):
```
$ python a2a_server_manual_check.py &
Serving 'manual-check-agent' at http://127.0.0.1:9877 (Ctrl-C to stop)
INFO:     Uvicorn running on http://127.0.0.1:9877 (Press CTRL+C to quit)

$ python a2a_client_manual_check.py
resolved real card over real TCP: manual-check-agent - real two-process verification
answer over real TCP: manual check: 42
```
Plus `tests/test_a2a_server.py`, the hermetic automated version of the same check (real
`a2a-sdk` client and server, `httpx.ASGITransport`, zero real sockets) — skips cleanly
rather than failing when `a2a-server` is not installed, since plain `uv sync` (what a
plain-CLI user runs) does not install it; CI's own `uv sync --locked` was updated to add
`--extra a2a-server` so this suite is not silently skipped there too.

**Verification**:
```
uv run pytest                          # 196 passed
uv run python -m teacup_agent.evals    # 20/20 passed
uv run teacup-agent                    # offline demo unaffected, still instant
```

---

### 19. Self-recorded experience and lessons learned — DONE (2026-09-03)

**Now**: when a run goes well, or goes wrong and recovers, that knowledge lived only in
`runs/<timestamp>/state.json` until a human happened to read it. This file's own "Field
patches" section is the human-curated version of exactly this idea (symptom, root cause,
fix, general principle) — this item automates the *first draft* of that, without
replacing the human curation step.

**Trigger conditions**, computed for free from `trajectory.mechanical()` (no model call
unless one fires):
- **Experience** (success): `status == "done"`, not `salvaged`, zero `pending_todos`,
  zero `duplicate_tool_calls`, `not action_never_attempted` — deliberately strict, so a
  messy run that happened to finish is not written up as a model to imitate.
- **Lesson** (recovered error): `failed_tool_calls > 0` (an `ERROR:` appeared in the
  trace) **and** the run still ended `delivered=True` — proof the error was actually
  worked around, not just present.
- Neither firing costs nothing; both can fire in the same run.

**What shipped**: a new [`reflect.py`](../src/teacup_agent/reflect.py), shaped exactly
like `plan.py` — one extra model call, no tools, fed `trajectory.render_trajectory(state)`
(already built for the judge), asking for `{"experience": "...", "lesson": "..."}` (only
the keys that apply), each one sentence, explicitly told to generalize beyond the specific
query and never invent a mechanism the trace does not support. Any failure (bad JSON, an
exception) writes nothing — same "a broken planner must never stop the run" discipline as
`plan.py`. Wired into `loop.run()`'s inner `_loop()`, called once right before the final
`persist.save()` so the reflection call's own cost is captured in the persisted state; a
new `reflect: bool` param plus a `--reflect {auto,on,off}` flag, same `auto` =
on-for-`--live`/off-for-the-offline-demo convention as `--plan` (reuses the same
`_resolve_plan()` helper — it was already a generic auto/on/off resolver, not
plan-specific).

**Storage and trust**: a second list on `Memory` (`notes`, not `facts`) in the same
`memory.json` — inherits `NullMemory`'s no-op-during-evals safety for free, no new file.
`recall()` renders it as a **separately labeled, lower-trust block** ("unreviewed notes
... weigh accordingly"), never merged with the model's own `remember`-tool facts. The
intended workflow: a human periodically skims the notes, promotes the good ones into this
file's own "Field patches" section, and deletes the rest — the automated log is a feed for
that review step, not a replacement for it, so `REVIEW.md`'s "the reviewer is not the
author" holds even here.

**Risk stated plainly, not just mitigated**: this is a feedback loop — the agent grading
its own work and feeding the grade back into its own future context. A confabulated
"lesson" or a generous self-assessment of a mediocre run compounds quietly if nothing
catches it; the strict trigger conditions and the low-trust, human-reviewed framing above
are the actual mitigation, not a footnote.

**Verification**:
```
uv run pytest                          # includes tests/test_reflect.py
uv run python -m teacup_agent.evals    # loop health, scripted model, free
uv run teacup-agent                    # offline demo unaffected (reflect defaults off)
```

---

## Deliberately not doing

- **No agent framework** (LangGraph and friends). The value of this repo is that the
  80-line loop is **yours** and fits on one screen. Wrap it in a framework and the
  learning value drops to zero.
- **No multi-tenancy or web UI.** That is a different project. The one narrow exception
  is #18's optional A2A server: a second console script behind a separate install extra,
  never loaded by `uv run teacup-agent`, added because Agent2Agent interop needs an agent
  that can be called into, not because this became a service project.
- **No race for tool count.** Five tools demonstrate every mechanism; if you want more
  tools, go through MCP (#9).

---

## Priorities in one line

Want it **smarter** -> #1 (Responses API).
Want it to sustain **longer work** -> #4 (context compaction).
Want it **faster** -> #5 (parallel tools).
Want it **trustworthy** -> #7 (trajectory eval).
Want it to **do more** -> #9 (MCP).

---

## Field patches (from reviewing three real runs)

None of these were on the original roadmap; they came out of running real tasks. They
share a root cause: the loop was not wrong, the **model was missing the information it
needed to decide**.

### A. Tell the model what day it is — DONE

**Symptom** (run 1): asked to research Anthropic's last six months, the search returned
genuine 2026 news and the model judged it "wildly out of line with public information,
very likely false", refused to use it, and handed back a **request for permission**
instead of a briefing.

**Root cause**: no date in the system prompt. The model measured newer information
against training-era memory and systematically distrusted the search results — in a
fast-moving field that disables search entirely.

**Fix**: inject today's date, state explicitly that "search results take precedence
over your priors", and require **source grading** (primary > major media > SEO
aggregators) instead of blanket rejection. Later addition: **anchor query time
expressions to today too** — do not build queries on remembered years and events (in
run 2 it searched "funding 2025" and "Claude 3.5", and thereby missed a revenue report
it had found the run before).

### B. Tell the model how much is left — DONE

**Symptom** (run 1): with 6 turns and 97% of the budget remaining, it asked "may I run
2-3 more searches?" — and the CLI is single-shot, so **nobody could answer it**.

**Root cause**: `AgentState` had step and remaining_budget all along, but the model was
never told; nor did the system prompt say it was running unattended.

**Fix**: append a `[run status]` message each turn (at the end, so the prompt-caching
prefix stays intact), and state in the system prompt that "nobody will answer your
questions, do not ask for permission".

### C. Force a wrap-up when a ceiling is hit — DONE

**Symptom** (run 3): after A and B it became far more autonomous — too autonomous. It
spent all 8 turns searching (and searched well: anthropic.com/news, Bloomberg, NYT and
FT all found), hit `max_steps`, and printed "(no final answer)". **Ten searches paid
for, nothing to show.**

**Root cause**: the brake only stopped the car, it did not unload it. And the status
line showed a comfortable budget (91%) while the steps were nearly gone — the model
weighed the wrong one.

**Fix**, in two layers:
1. the final turn is handed an **empty tool list** (wording can be ignored, an empty
   list cannot), with sharper wording in the status line;
2. when a ceiling is actually hit, ask once more, again with no tools, and squeeze
   "conclusion + confidence + unverified items" out of what is there — `salvaged=True`
   only if something was actually rescued.

**The lesson**: all three point at the same general principle —
**an agent's failures are usually not in the control flow but in "the model does not
know its own situation"**. Having the state in `AgentState` is not the same as the
model knowing it; if you do not tell it, it can only guess.

### D. Time budget — DONE (2026-08-26)

**Motivation**: once search was real, the throttle interval, backoff retries and
network latency made wall-clock time grow a lot, while the loop had no idea how long it
had been running — 8% of the money spent and two minutes of a human's life gone. Money
measures model compute, time measures human waiting, and neither substitutes for the
other.

**Implementation**: `run(time_budget=...)` / `--deadline <seconds>`, **default 600
(10 minutes)**, 0 for unlimited, with a new `out_of_time` status that also goes through
the forced wrap-up from C. The `[run status]` line now puts **the tightest** of steps /
budget / time in front of the model. The `clock` parameter accepts a fake clock, so
this brake is reproducible in evals (`clock_values`) without sleeping.

**Known limitation**: time is only checked between turns, so a wedged tool call can
still overrun. That is what `--tool-timeout` (from #5) is for.

### E. Attempt the gated call; do not ask for permission in the answer — DONE (2026-08-26)

**Symptom**: asked to "research X, then email me the result", the agent researched
well and **never called `send_email` at all**. It wrote a draft into its final answer
and said "reply 'please send' to authorize me". More budget did not help: with the
address supplied and 14 turns available it still stopped at turn 6 with 97% of the
budget unspent. It was not running out of room; it never intended to make the call.

**Root cause**: the prompt offered an exit — "if denied, take another route **or state
in your final answer that this step is for the user to do**" — and the model took the
exit *pre-emptively*, as the safest-looking option. Which turned an interactive
question into a dead end, because the approval prompt was waiting one step away and
nobody reads the final answer before the run ends.

**Fix**: spell out the order, and say where authorization actually lives.

1. call the tool when the task asks for it;
2. only a *denied* call justifies another route, or saying the step is left to the user;
3. never re-send an identical denied call.

Plus one line: do not ask for authorization in the answer — the approval prompt is
where the user grants or refuses it.

**Verified live twice** (gpt-5-mini, `--approve deny` so nothing could be sent):
`send_email` is attempted, denied by policy, and the item is then marked blocked with
the reason. Before the fix, the tool was never called in any run.

### F. A checklist, so half a task cannot look finished — DONE (2026-08-26)

**Symptom**: the same run as E. Even with the prompt fixed, nothing in the system knew
that the request had two halves. `status: done` was reported for a task that was half
delivered, and neither the loop nor the metrics could tell.

**Root cause**: the same general principle as A/B/C, one level up — **the model did not
know it had missed something, because nobody was remembering.** A plan held in the
model's head cannot be recovered once it drifts out of attention.

**Implementation**: new [`plan.py`](../src/teacup_agent/plan.py) plus `AgentState.todo`.

- `decompose()` makes one model call at the start (no tools) and turns the goal into
  1-5 action items. A planner that fails returns an empty list, which degrades exactly
  to the old behaviour — a broken planner must never stop a run.
- Every turn's `[run status]` line carries the checklist with `[x]`/`[ ]` marks.
- The model ticks items off with the `update_todo` tool: `done`, or `blocked` with a
  reason. Blocked is settled — it stops being outstanding but keeps the reason.
- **Completion check**: if the model stops calling tools while items are open, the loop
  pushes back once with the open items and continues; the answer stands on the second
  attempt either way. Once, never a loop (an eval case pins that down).
- The forced wrap-up turn (C) also names unfinished items, so a run that ran out of
  resources admits what it never did.
- CLI: `--plan {auto,on,off}`, where auto means on for `--live` and off for the
  offline demo (which has nothing to plan). `snapshot()` reports
  `todo_done`.

**A bug this caught immediately**: `persist.load()` rebuilt `trace` into dataclasses but
left `todo` as raw dicts, so scoring a saved run crashed on attribute access. Nested
dataclasses do not survive `asdict()` on their own — now both are rebuilt, with a test.

**Metrics** (trajectory eval): `action_never_attempted` (the goal used an action verb
and no gated tool was ever called — a *denied* attempt does not count, since the model
did its part), `asks_without_trying` (asked for authorization in the answer without
attempting the call), and `pending_todos`.

**Left open**: the checklist is fixed at the start. Replanning mid-run, when the task
turns out to be different from what it looked like, is not implemented.
