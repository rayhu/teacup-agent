# Design notes

Why the agent behaves the way it does, one subsystem at a time. Most of these decisions
were paid for with a failed run; the run is described alongside the fix.

Start with [the README](../README.md) if you want to know what the project is. Start with
[the roadmap](roadmap.md) if you want to know what it still lacks.

---

## The three search_web modes

Web search goes through [`ddgs`](https://pypi.org/project/ddgs/) (DuckDuckGo) and
needs **no API key**; `uv sync` installs it. Switch with the `TEACUP_AGENT_SEARCH`
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
uv run teacup-agent --live "research OpenAI's funding and competition"    # really goes online
TEACUP_AGENT_SEARCH=offline uv run teacup-agent                           # force offline
```

**A design rule: returning "nothing found" always beats returning the wrong thing.**
An early version matched the offline corpus with `any()` (any keyword in the query
counted as a hit), so "OpenAI strategy" matched the "nvidia gpu strategy" entry and
the model nearly answered an OpenAI question with NVIDIA material. It uses `all()`
now, with a regression test watching it. For the same reason, a failed search in
`web` mode must return `ERROR:` rather than nothing — otherwise the model reads "the
tool is broken" as "this does not exist in the world".

---

## Two OpenAI backends

| | `--api responses` (default) | `--api chat` |
| --- | --- | --- |
| Tool definitions | flat `{"type":"function","name":...}` | nested `{"function":{...}}` |
| One turn of output | an `output` list (reasoning item + several function_calls) | a single assistant message |
| Call id | `call_id` | `id` |
| Result shape | `{"type":"function_call_output",...}` | a `role="tool"` message |
| Reasoning state | **preserved across tool calls** | discarded every turn |

All four differences are sealed inside the two classes in
[`model.py`](../src/teacup_agent/model.py); the control flow in
[`loop.py`](../src/teacup_agent/loop.py) did not move. The loop only needed two
generalizations: `state.messages.extend(reply.items)` (a turn can produce several
entries) and letting the backend's `tool_result_item()` decide the result shape.

Measured on the same task (gpt-5-mini): the responses context carries an extra
`reasoning` entry that goes back out with the next request; the chat side has no
equivalent. OpenAI's migration guide reports roughly +3% on SWE-bench with the same
prompt.

---

## Model routing: one agent, several models

An agent does not make one kind of model call. `loop.py`'s own turns are judgment —
decide what the task actually is, notice at step 20 that the approach is wrong, know
when to stop. Summarizing old context, writing a note in a fixed schema, or running a
short self-contained subtask are not: they are well-specified work with a checkable
result, and a harness (retrieval instead of recall, tool feedback instead of one-shot
correctness, retries instead of precision) closes most of the capability gap on exactly
that. Sending all six call sites to one model means either paying judgment prices for
clerical work, or accepting a clerk's judgment.

So `models.roles` in `agent.yaml` maps a **role** — a call site — to a model profile,
and [`routing.py`](../src/teacup_agent/routing.py) is the lookup. Omit the block and
every role runs on `models.default`, which is exactly what happened before this existed.

### Why roles, and not a classifier

The obvious design is "look at the task, pick a model". That is roadmap #20's Stage C,
and it is deliberately not here yet, because the honest version of it needs a
measurement first: route by measured task class on your own workload, not by vibes. A
role map needs no measurement to be safe — it is fixed for the run, it is declared in a
file you can diff, and a wrong choice shows up as a bill or a bad answer attached to a
named role, not as a mystery.

The reason to be careful is the shape of the failure. A small model that is out of its
depth does not stall or ask for help; it produces plausible, confidently wrong work and
carries on. That is why the routing question is never "how hard does this look" but:

1. is success programmatically checkable (tests, a schema, an exact string)?
2. is the spec complete — ambiguity is the expensive part;
3. how long is the horizon — per-step errors compound, and 0.95^30 is a coin flip;
4. how noisy is the context — long distractor-heavy context degrades small models much
   faster than their benchmark scores suggest.

`plan` is the role worth spending on: a tiny fraction of a run's tokens, and it produces
the checklist the loop then holds the model to for the rest of the run. `subagent` is
the textbook cheap one: short horizon, narrow spec, and its answer returns as a tool
result the parent can sanity-check. `compact` is deliberately **left to the default**
until it is measured — it has the largest input in the repo, so it is where the money
is, but a lossy summary loses the thread silently and unrecoverably, which is the exact
failure mode above. The forced wrap-up is not a knob at all: it runs on the main
context's own prefix, so routing it elsewhere would pay a cold prefix for one call, and
"salvage a conclusion from partial information" is judgment.

### What routing does to the prompt cache

Caches are per model, so the grouping key now hashes the model id along with the
context prefix — two profiles sharing this system prompt must not be handed the same
key. Only the `main` role gets a key: the other five each build their own short,
constant system prompt, well under the ~1024-token minimum a prefix needs to enter the
cache at all. That is also why routing them away costs nothing in cache terms — they
never shared the run's prefix — and the saving there is purely the price difference.

The router builds each profile **once per run** and shares the instances with the
routers it derives for subagents. A fresh instance per call would mean a fresh HTTP
client and a fresh (empty) cache key every turn, quietly cancelling the thing the stable
prefix was protecting.

Routing is fixed for the run, which is what makes all of this safe. A mid-run switch
would have two hard problems: reasoning items are model-specific (carrying them back
verbatim is the whole value of the Responses path, and handing one model's reasoning to
another is at best ignored), and the two APIs disagree on the tool-result shape, so a
run that crossed `chat` and `responses` would hold a message list in two shapes. Both
are Stage C's problem, and both are why an escalation there has to pass through a
context rewrite rather than just swapping the object.

### What the measurement said

`bench.py` ran these choices under real models (roadmap #20 Stage B, $0.77 across two
runs). The short version, because the defaults above are only as good as the evidence
for them:

- On a task with a complete spec and a checkable result, the small model was **5x
  cheaper at identical quality**. That is the axis routing works along.
- On an underspecified research task it was **12x cheaper and produced no answer at
  all** — while showing zero failed calls, zero duplicates and a `delivered` flag of
  yes. Every free mechanical signal said the run was fine; only the model judge saw the
  collapse. `mechanical()` is the column to trust, but it is not complete, and "did not
  actually answer the question" is outside it.
- **`compact` was left on the default and stays there.** Routing it to the small model
  is where the money is — five compaction calls cost $0.0255 on `gpt-5-mini`, roughly a
  fifth of what the same five would cost on `gpt-5`, which is most of a run — but the
  quality read was not interpretable at n=1, and the same small compactor produced both
  the worst and the best citation record in the same row.
- Single quality numbers from single runs are worth nothing here, and that is measured
  rather than assumed: two cells with **identical** models on every role that ran
  differed 2 against 7 on unsupported citations.

### Two things this exposed

`plan.decompose()` never charged its model call to the run. It went unnoticed while
every call was the same price; it becomes a lie in the budget the moment `plan` is the
role pointing at the expensive profile, so `decompose()` now takes the state and charges
like everything else.

`prompt_cache_key` was dead on the Chat Completions path. `OpenAIModel.complete()` has
always read `self.cache_key`, but only `ResponsesModel` had a `set_cache_key()`, so
`loop.py`'s `getattr(model, "set_cache_key", None)` silently skipped it. The
`getattr`-based optional-method style is what let it hide — nothing fails when the
method is missing, which is the point and also the hazard.

One wrinkle that predates routing and is still there: a subagent's own `loop.run()`
calls `set_cache_key()` with the child's system-prompt hash, overwriting the parent's
key on a shared instance for the rest of the parent run. It costs cache hits, never
correctness, and it only stops happening when parent and child resolve to different
profiles.

---

## MCP

Every tool in `tools.py` had to be written by hand. MCP is how you stop doing that:
point at a server and its tools appear in the registry.

### Setting it up

MCP is **off unless you configure it**, because connecting means starting third-party
processes and putting their tool schemas into the context prefix of every request. But
once a project has an `mcp.json`, that file's existence *is* the opt-in, which is the
convention every other MCP host uses, so it loads automatically from then on.

```bash
cp mcp.example.json mcp.json     # 1. start from the template
$EDITOR mcp.json                 # 2. keep the servers you want
uv run teacup-agent --live "read <url> and summarise it"   # 3. that is all
```

```
[mcp] using /path/to/mcp.json (pass --mcp off to skip it)
[mcp] fetch: 1 tool (0 gated) — fetch__fetch
[step 1] -> fetch__fetch({"url":"https://modelcontextprotocol.io/specification/versioning"})
```

`--mcp <path>` uses a different file; `--mcp off` disables it for one run. `mcp.json` is
gitignored, because the `env` block is where server credentials go — commit
`mcp.example.json` instead.

### Writing the config

```jsonc
{
  "servers": {
    "fetch": {
      "command": "uvx",                        // stdio server: a command we start
      "args": ["mcp-server-fetch"],
      "approve": "none"                        // "I trust this server"
    },
    "files": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
      "tools": ["read_text_file", "list_directory"]   // only these two
    },
    "remote": {
      "url": "https://example.com/mcp",        // or a Streamable HTTP server
      "env": {"API_TOKEN": "..."}
    }
  }
}
```

| Key | Meaning |
| --- | --- |
| `command` / `args` / `env` / `cwd` | a stdio server: the process to start |
| `url` | a Streamable HTTP server instead |
| `tools` | allowlist. **Use it.** Every schema you take sits in the prefix of every request |
| `approve` | `auto` (default: open only what the server marks read-only), `all`, `none` |
| `stderr` | `hide` (default) or `show` — legacy servers print a wall of validation errors when probed |
| `_`-prefixed keys | ignored, so you can write comments in JSON |

Servers live in the [MCP registry](https://github.com/modelcontextprotocol/servers) and in
the wider ecosystem; `uvx <server>` and `npx -y <package>` are the two usual ways to run
one. Start with one server and one allowlisted tool: a config that pulls in forty tools
makes every request slower, more expensive, and less focused.

**Which protocol**: the current revision, **2026-07-28**, the stateless one. It removed
the `initialize` handshake and the session id, so a tool call is now a single stateless
RPC and this integration is a fraction of what it would have cost a year ago. We use the
official Python SDK (v2), which also speaks the older revisions, so servers that have not
migrated still work — verified against `mcp-server-fetch`, which is still handshake-based.

Four details carry the weight:

| | |
| --- | --- |
| **Names are namespaced** | Two servers can each expose `search`, so tools land as `server__tool` (also sanitized: OpenAI function names allow only `[A-Za-z0-9_-]`, MCP allows dots) |
| **Errors keep the discipline** | MCP separates protocol errors from *tool execution errors* (`isError: true`) and says clients should hand the latter to the model to self-correct. That is what `execute()` already did, so both become `ERROR: ...` |
| **Approval is derived, and defaults to gated** | `annotations.read_only_hint` opens a tool; everything else needs approval — including a server that annotates nothing, which is the common case. The spec says annotations are untrusted unless the server is, so `"approve": "none"` is how you *state* trust rather than have us infer it |
| **Async lives in one place** | The SDK is async, our tools are sync functions in a thread pool, so one background event loop owns every session and nothing else in the codebase learns about asyncio |

Per-server config keys: `tools` (allowlist — every schema costs prefix tokens on every
request), `approve` (`auto` / `all` / `none`), `stderr` (`hide` / `show` — legacy servers
print a wall of validation errors when probed with `server/discover`).

A server that fails to connect prints why and the run continues without it.

**Not implemented**: resources, prompts, subscriptions, tasks, and everything the
2026-07-28 revision deprecated (roots, sampling, logging). A `resultType: "input_required"`
result, which is the multi-round-trip pattern, comes back as an `ERROR:` the model can
route around, since we do not do elicitation.

---

## Agent config (`agent.yaml`)

Every flag on `cli.py` describes one invocation. `agent.yaml` describes the *agent*:
which models it can reach and how, which MCP servers, which built-in tools and skills,
and the runtime ceilings — a file you can diff, copy between projects, and hand to
someone else instead of a wall of flags.

### It is a parallel track, not a merge

```bash
cp agent.example.yaml agent.yaml    # 1. start from the template
$EDITOR agent.yaml                  # 2. pick a model, MCP servers, ceilings
uv run teacup-agent --config agent.yaml "your goal"   # 3. that is all
```

Passing `--config` hands over control entirely: every other behavior flag (`--model`,
`--api`, `--mcp`, `--max-steps`, ...) is ignored rather than partially overridden. The
alternative — flags override individual YAML fields — means the effective config for any
given run is "whichever of ten sources won for this field," which is worse than two clear
modes. `goal`, `--quiet` and `--resume` stay CLI-only either way: a goal and a resume path
name one specific invocation, not a property of what the agent is.

### Secrets stay out of the file

A model profile names an environment variable (`api_key_env: OPENAI_API_KEY`), it never
holds a key. Any string value anywhere in the file may also embed `${VAR}`, expanded from
`os.environ` (which `.env` already populates). A missing variable is a load-time
`ValueError`, not a silently blank substitution — the same "fail loudly" instinct as
everywhere else here. `agent.yaml` is gitignored; `agent.example.yaml` is the template,
same pairing as `mcp.json`/`mcp.example.json`.

### Reaching more than one model

`models.profiles` names a model; `models.default` picks which one this run uses.
`provider: openai-compatible` plus `base_url` reaches anything that speaks the OpenAI
wire format — vLLM, Ollama, OpenRouter, any gateway — for free: `OpenAIModel`/
`ResponsesModel` already accept a pre-built `client`, so `agent_config.build_model()`
constructs one with a custom `base_url`/`api_key` instead of letting the class fall back
to its own `OPENAI_API_KEY`-from-env default. No change to either class was needed.

A native second wire protocol (Anthropic's own Messages API shape, not just a different
URL) is a distinct follow-up, added the same way `ResponsesModel` was added beside
`OpenAIModel` — see `docs/roadmap.md` #16.

### A YAML gotcha worth naming: the "Norway problem"

PyYAML (1.1 rules) parses a bare `off`/`on`/`yes`/`no` as a **boolean**, not a string. This
schema uses `"off"`/`"on"`/`"auto"` as string sentinels in four places (`runtime.plan`,
`runtime.reflect`, `runtime.run_dir`, `skills.dir`), so `plan: off` written unquoted parses
to the Python `False`, and a naive `== "off"` check would silently fall through to a
default instead.
`agent_config._normalize_off_on()` maps the boolean back to the matching sentinel string
before anything compares against it, so the field means the same thing quoted or not.
`tools.subagents.enabled` is a real boolean and deliberately does **not** go through this
— only the fields that use the words "on"/"off" as enum values do.

### What is in scope, and what is reserved

`models`, `mcp` (identical per-server shape to `mcp.json`, one level deeper), `tools`
(built-in tool exclusion, the subagent delegate tool), `skills` (which directory),
`runtime` (the scalar ceilings), `a2a.peers` and `a2a.card` are all live — the last two
landed as two independent items (#17 client, #18 server) that happened to give
`AgentConfig.a2a` the same real shape at the same time; see `docs/roadmap.md`.

---

## Agent2Agent (A2A) client

`subagent.py`'s `delegate` hands a subtask to a child **in-process** loop. `delegate_a2a`
is the same idea across a process (or machine) boundary, to an agent that may not even be
teacup-agent — built on the official `a2a-sdk` rather than hand-rolling
JSON-RPC/task-lifecycle/SSE, the same "depend on the official SDK" choice `mcp_tools.py`
already made for MCP.

### Setting it up

```yaml
a2a:
  peers:
    finance-agent:
      url: https://agents.example.com/finance
      api_key_env: FINANCE_AGENT_TOKEN   # optional; omit for no auth header
```

Only reachable through `agent.yaml` (`--config`) — there is no bare `--a2a` flag, the
same choice #15 made for `a2a.peers`' sibling blocks. Once configured, the model sees one
tool, `delegate_a2a(peer, task)`, regardless of how many peers are listed:

```
  [a2a] delegate_a2a available for peers: finance-agent
  [step 1] -> delegate_a2a({"peer": "finance-agent", "task": "..."})
```

`requires_approval=True` is not a default to reconsider: an outbound call to a
third-party, possibly-billed agent is exactly the side effect AGENTS.md rule 4 (deny by
default when nobody is watching) exists for.

### Where the hub lives, and why

`subagent.py` and `skills.py` register their tool(s) *inside* `loop.run()` because they
need nothing external. `mcp_tools.py`'s `McpHub` instead lives in `cli.py`, constructed
once per process invocation, because it owns a real external resource (child processes,
sessions) that doesn't belong to any single `loop.run()` call. An A2A peer connection is
the second kind — an `httpx` client and a resolved Agent Card, configured once from
`agent.yaml`, not per-turn — so `A2AHub` (`src/teacup_agent/a2a/client.py`) follows the
MCP shape: built in `cli.py`'s `_main_config()` next to `McpHub`, torn down in the same
`finally`. This means **`loop.py` needed no changes at all**: the approval gate already
reads `Tool.requires_approval` generically, so registering the tool with that flag set is
everything #17 needed from the loop.

Unlike MCP (one server -> N distinct tools, each with the server's own schema), every A2A
peer shares one tool shape, so there is no per-peer schema to fetch upfront. Connecting is
lazy: the first call to a given peer resolves its Agent Card and opens an
`httpx.AsyncClient`, cached for the rest of the run. Only each peer's `api_key_env` is
resolved eagerly, at `register()` time — a missing token fails at startup rather than
silently mid-run, the same discipline `agent_config.build_model()` already holds for
model profiles.

### The async/sync bridge

`a2a-sdk`'s client is async (`httpx`-based); `loop.py`'s tool-execution thread pool calls
tool functions synchronously. `A2AHub` solves this exactly the way `McpHub` already does:
one background thread runs one `asyncio` event loop for the hub's whole lifetime, and
`_run(coro, timeout)` bridges a synchronous call onto it with
`asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)`. Every failure —
connection error, a remote task that ends in `TASK_STATE_FAILED`/`REJECTED`/`CANCELED`, or
one that asks for interactive input (`TASK_STATE_INPUT_REQUIRED`/`AUTH_REQUIRED` — this
client does not do elicitation, same as MCP's `input_required`) — becomes an
`"ERROR: ..."` string, never a raised exception, so the model can self-correct.

### An API surface note

`a2a-sdk` (v1.1.2, confirmed against the installed package rather than blog posts, which
described an older, different shape) builds every message as a **protobuf** type
(`a2a.types.SendMessageRequest` is literally `a2a_pb2.SendMessageRequest`), with
`a2a.helpers` providing plain-Python constructors (`new_text_message`,
`get_stream_response_text`) so callers rarely touch protobuf directly.
`create_client(agent=<url string>, ...)` looks like the simplest entry point, but its
internal card resolution does not reliably go through the `httpx_client` passed in
`ClientConfig` — verified directly against the installed package, not assumed. `A2AHub`
resolves the card explicitly with `A2ACardResolver` first and passes the resolved
`AgentCard` to `create_client` instead, which is also what lets the test suite
(`tests/test_a2a_client.py`, `tests/fixtures/demo_a2a_server.py`) drive a real in-process
server through `httpx.ASGITransport` with zero real sockets.

---

## Agent2Agent (A2A) server

`teacup-agent-serve` is the flip side of `delegate_a2a` (#17): instead of this agent
calling *out* to a peer, another agent (or `delegate_a2a` itself) can call *in*. Built on
the same official `a2a-sdk` rather than hand-rolling JSON-RPC/task-lifecycle/SSE.

### The tension worth stating plainly

`docs/roadmap.md`'s "Deliberately not doing" list has said *"no multi-tenancy, service
layer or web UI"* since that file began, and a long-lived HTTP process accepting inbound
tasks is, structurally, a service layer. The resolution: make it a second, explicitly
opt-in surface at **two** levels, not one —

1. **Install time**: `a2a-sdk[http-server]` (Starlette + uvicorn) is gated behind a new
   `a2a-server` optional dependency group. `uv sync` (what a plain-CLI user runs) never
   installs it; only `uv sync --extra a2a-server` does.
2. **Invocation time**: a second console script, `teacup-agent-serve`, entirely separate
   from `teacup-agent`'s own `main()`. Running `uv run teacup-agent` is unaffected
   whether or not the extra is even installed.

### What it reuses, deliberately, rather than reinvents

- **The approval gate.** `TeacupAgentExecutor.execute()` calls the existing
  `cli._make_approver(cfg.runtime.approve, quiet=True)` unmodified. Its `"auto"` branch
  already denies whenever `sys.stdin.isatty()` is false — always true under `uvicorn` —
  so a served agent is gated by AGENTS.md rule 4 with **no new approval code**. A remote
  caller cannot trigger `send_email` (or anything else `requires_approval`) without this
  instance's own policy allowing it.
- **`loop.run()` itself, unchanged.** `execute()` reads the incoming message text and
  calls the same `loop.run(goal=..., model=..., ...)` every CLI invocation calls, via
  `asyncio.to_thread()` since the handler is async and `loop.run()` is not.
- **`skills.py`'s catalog, as the Agent Card's `skills` list** (`a2a/card.py`) — a
  one-line description per skill is exactly the grain an `AgentSkill` wants; enumerating
  every low-level tool name instead would be noisy and the wrong level of detail. No
  `skills/` directory configured gets one generic fallback skill rather than an empty
  list.

### What is honestly left undone

`TeacupAgentExecutor.cancel()` raises `NotImplementedError`, matching `a2a-sdk`'s own
reference examples for "not supported" — `loop.run()` has no cooperative-cancel hook
today, so faking cancellation would be worse than refusing it.

**Concurrency is deliberately serialized.** `tools_mod.REGISTRY` is process-global, and
`skills.enable()`/`subagent.enable()` mutate it with no locking, assuming one `loop.run()`
at a time. Two inbound tasks arriving concurrently on a server whose `agent.yaml` turns on
skills or subagents would race on that shared state. `TeacupAgentExecutor` holds an
`asyncio.Lock` around each `loop.run()` call for exactly this reason — correct, at the
cost of one served task at a time. Real per-run tool isolation (so concurrent runs do not
share global registry state at all) is a larger change, out of scope here.

### Verified for real, not just in-process

The automated suite (`tests/test_a2a_server.py`) drives the real `a2a-sdk` client and
server through `httpx.ASGITransport` — real protocol, zero sockets. Separately, a manual
check used two actual OS processes over a real loopback TCP port (a `ScriptedModel`
injected so it cost nothing):
```
$ python a2a_server_manual_check.py &
Serving 'manual-check-agent' at http://127.0.0.1:9877 (Ctrl-C to stop)
INFO:     Uvicorn running on http://127.0.0.1:9877 (Press CTRL+C to quit)

$ python a2a_client_manual_check.py
resolved real card over real TCP: manual-check-agent - real two-process verification
answer over real TCP: manual check: 42
```

---

## The checklist

Ask for "research X, **then email me the result**" and an agent will happily do the
first half. A real run did exactly that: excellent research, no email, `status: done`,
and it stopped at turn 6 of 14 with 97% of the budget unspent. It was never a resource
problem — **nothing was keeping track of the second half of the request**.

So the goal is decomposed once at the start into 1-5 action items — one extra model
call, controlled by `--plan {auto,on,off}`, where `auto` (the default) means on for
`--live` and off for the offline demo, which has nothing to plan:

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

---

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

---

## Two kinds of evaluation — do not conflate them

| | `evals.py` | `trajectory.py` |
| --- | --- | --- |
| Model | fake (scripted) | a real trajectory (`runs/*/state.json`) |
| Question | **is the machine broken?** message protocol, brakes, compaction cut points | **how well did this run go?** outcome, grounding, efficiency, honesty |
| Cost | zero | mechanical metrics free; the LLM judge costs per run |
| Verdict | every case must pass | relative scores, for comparing two versions |

```bash
uv run python -m teacup_agent.trajectory runs/20260826-xxxx           # mechanical, free
uv run python -m teacup_agent.trajectory runs/* --judge --out r.json  # add the LLM judge
```

**Mechanical metrics** (deterministic; read these first): steps, tool calls, failures,
**duplicate calls**, throttled, denied, compactions, elapsed, tokens, cache hits,
whether a conclusion was actually delivered, **whether it asked the user back**
(the very first real run failed exactly this way), citation count,
**`unsupported_citations`**, the deterministic detector for invented citations: links
that appear in the answer but in no tool result. Plus two signals for half-finished
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

---

## Self-recorded experience and lessons

A run's own knowledge of how it went used to live only in `runs/<timestamp>/state.json`,
read by a human if anyone got around to it. `docs/roadmap.md`'s own "Field patches"
section is the human-curated version of exactly this idea (symptom, root cause, fix,
general principle) — `reflect.py` automates the *first draft*.

### Trigger conditions, computed for free

Reusing `trajectory.mechanical()` (already free and deterministic — no model call
happens unless one of these fires):

| Note | Fires when |
| --- | --- |
| `experience` | `status == "done"`, not `salvaged`, zero `pending_todos`, zero `duplicate_tool_calls`, and the action asked for was actually attempted |
| `lesson` | at least one `ERROR:` appeared in the trace **and** the run still ended `delivered=True` — proof the failure was worked around, not just present |

Both conditions are deliberately strict. A run that finished but left something open,
duplicated a call, or never attempted the action it was asked for is not written up as a
model to imitate — the whole point of the eval metrics existing is to catch exactly that,
and this reuses them rather than re-deriving a looser version.

### The write itself

Shaped exactly like `plan.py`'s `decompose()`: one extra model call, no tools, fed
`trajectory.render_trajectory(state)` (already built for the judge), asked for
`{"experience": "...", "lesson": "..."}` with only the keys whose trigger fired, each one
sentence, told explicitly to generalize beyond the specific query and never invent a
mechanism the trajectory does not support. Any failure — bad JSON, a raised exception, the
model omitting a field whose trigger fired — writes nothing, the same "a broken planner
must never stop the run" discipline `plan.py` already has. `ScriptedModel` grew a matching
`reflection={"experience": ..., "lesson": ...}` constructor arg, so evals can script this
call the same way they already script the planner and the compaction summarizer.

Called once inside `_loop()`, right before the final `persist.save()` — so the
reflection call's own cost is captured in the state that gets written to disk, even
though `state.status` is already final by the time it runs.

### Storage is a second, lower-trust tier

`Memory` gained `notes` alongside `facts`, in the same `memory.json`: written by
`Memory.note(kind, text)`, never by the model's own `remember` tool. `recall()` renders
them as a **separately labeled block**, after the facts block, prefixed "unreviewed...
weigh accordingly" — a fact the model chose to remember mid-task and a note the harness
wrote about a run after it already ended are not the same kind of claim, and conflating
them would erase that distinction exactly where it matters most.

### The risk this does not pretend to have solved

This is a feedback loop: the agent grading its own work and feeding the grade back into
its own future context. A confabulated "lesson", or a generous self-assessment of a
mediocre run, compounds quietly if nothing ever looks at it. The strict trigger
conditions and the low-trust framing above are real mitigations, not decoration, but they
are not a substitute for the actual answer: a human periodically reads the accumulated
`notes`, promotes the good ones into this file's own "Field patches" section (reviewed,
durable, attributed to a real run), and deletes the rest. The automated log is a feed for
that review step. `REVIEW.md`'s "the reviewer is not the author" is not a rule that stops
applying just because the author is a harness instead of a human.

```bash
uv run teacup-agent --live --reflect on "..."     # force it (auto already does this for --live)
uv run teacup-agent --reflect off "..."            # skip it even with --live
```

---

## Persistence and resume

Every step writes the full state to `runs/<timestamp>/state.json` (temp file plus
rename, so no half-written files). **Saving every step rather than at the end is
deliberate** — save only at the end and the crash that most needed the data is the
one that leaves nothing.

```bash
uv run teacup-agent --max-steps 1 --run-dir runs/demo "..."   # hits the ceiling and stops
uv run teacup-agent --resume runs/demo --max-steps 3          # continues from turn 2
```

On resume, the command-line ceilings mean "**give it this much more**" (steps and
time already spent live in the state, so reusing them verbatim would hit the ceiling
again immediately). One easily-missed detail: **resume does not rebuild the system
message** — rebuilding changes the context prefix and voids every prompt-cache entry
earned so far. A test watches that byte for byte.

What gets saved is also the input to trajectory eval: full messages, every tool
call's arguments and result, the spend, the final state. Reviewing a run used to mean
guessing from the 120 characters the terminal printed.

---

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
showing 0% is normal: a prefix under ~1024 tokens never enters the cache at all.

---

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
goal and the last 8 entries are kept. The cut must not orphan anything: not a tool call
from its result, and not — in the Responses shape — a `function_call` from the
`reasoning` item the API requires with it. The second half of that rule was missing
until a live run 400ed on it (roadmap Field patch G); the suite was green because
nothing had ever compacted a Responses-shaped context. The summarizer prompt explicitly demands
"verified facts + source links + what was tried and failed". The last one matters
most, or the agent walks into the same dead end again after compaction.

**The dangerous part of compaction is not summary quality, it is the cut point.**
Split a tool call from its result and the next request fails with a 400. So
`safe_cut_points()` only cuts where nothing is dangling. The same scan as the
ordering invariant in evals. If there is no safe cut, it does not compact. One eval
case watches "the protocol survives compaction".

The decision uses the **real token count from the model** (`usage.input_tokens`),
falling back to a character estimate (CJK at 1.5 chars/token, everything else at 4)
only when there is none.

---

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

---

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
results, so the next request fails with a 400. The semantics here are "refused", not
"ignored". Reading that, the model converges on its own and squeezes ten queries into
the few that matter. `evals.py` pins this behaviour, and `state.snapshot()` reports
`tool_calls` (actually run) and `throttled` (held back) separately.

### A brake must do more than stop

Lesson from a real run: the model spent all 8 turns searching and produced not one
line of conclusion — ten searches paid for, nothing to show. So the brakes also have
to **unload the car**. Two mechanisms:

1. **No tools on the final turn**: when `state.step >= max_steps` the tool list goes
   out empty. Wording can be ignored, an empty list cannot. The model's only
   remaining option is to talk.
2. **A forced wrap-up turn**: when a ceiling is actually hit, ask once more, again
   with no tools, and require "conclusion + confidence + unverified items" from what
   is already there. `salvaged` is only set to true if something was actually
   rescued.

Each turn also appends a `[run status]` message telling the model which turn it is on
and how much budget is left: it cannot choose between digging and wrapping up
without knowing where it stands. That line is appended at the **end** rather than
written into the system prompt; otherwise the context prefix changes every turn and
prompt caching is void.

```bash
uv run teacup-agent --live --max-tool-calls 2 "research OpenAI's funding and competition"
```

---

## Subagents

Compaction summarises and therefore discards. Externalizing keeps the text but still
spends the excerpt. A subagent avoids the cost entirely: it reads five pages in a context
of its own and hands back three sentences, and the parent never pays for the pages
because it never sees them.

```bash
uv run teacup-agent --live --subagents "research X across several sources and summarise"
```

```
[step 1] -> delegate({"task": "read the spec page and summarise the versioning rules"})
    [sub1] -> fetch__fetch({"url": ...})
    [sub1] returned 412 chars to the parent
```

One parent step buys a whole child run. That is the trade: a step is cheap, a context
window is not.

**What the child gets**: a blank context (it cannot see the conversation, so the task
must be self-contained), every tool except `delegate`, a slice of the parent's *remaining*
budget (40% by default, read at call time so a nearly-broke parent cannot fund an
expensive child), its own step ceiling (`--subagent-steps`, default 4), and whatever wall
clock the parent has left.

**What comes back**: the answer only. Everything the child read stays in the child.

**What that costs, and how it is paid for.** The parent cannot audit a conclusion it
cannot see, so the child is not invisible, only out of context:

- the child's full trace and its own `state.json` are written to
  `runs/<timestamp>/sub01/`, so a review can open exactly what it read;
- its events are forwarded to the terminal with a `[sub1]` marker;
- every dollar and token it spends is charged to the parent, because they were the
  parent's to begin with.

**Guards.** A child cannot delegate further: `delegate` is left out of the tool list it is
given, and one level is enforced there rather than by a depth counter, because a recursion
here is a recursion that spends money. A child that ends without an answer becomes an
`ERROR:` result the parent can read and route around, the same as any other tool failure.

**Two implementation details worth knowing.** The tool is registered per run rather than
globally, since an unbound schema would sit in the context prefix of every request for a
capability the run cannot use. And `Tool.timeout` exists because of this feature: a child
run is not a page fetch, and the loop's 30-second default would kill it mid-thought.

### Measured, including the part that is not a win

Same task (read two spec pages, answer in three bullets), gpt-5-mini, MCP `fetch`:

| | flat run | with two subagents |
| --- | --- | --- |
| Raw characters entering the **parent** context from tools | 4,700 | **916** |
| Parent `context_tokens` at the end | 6,515 | **4,333** |
| Total input tokens (parent + children) | 28,866 | **38,465** |
| Cost | $0.0110 | **$0.0148** |

The mechanism does exactly what it claims: the children read 6,314 characters between
them and handed back 296 and 524 characters, so the parent's context ended a third
smaller. **And the run cost 35% more.** Isolation is not free. Each child carries its own
system prompt and re-reads material the parent would have had anyway, so total tokens go
up even as the parent's context goes down.

That trade only pays off when the parent would otherwise **carry** the bulk for the rest
of the run: context spent on turn 2 is context still being resent on turn 12. On a
two-page task that ends immediately afterwards, delegation is a net loss. On a long
investigation across ten sources, it is the difference between finishing and running out
of window. Use it for reading-heavy subtasks inside long runs, not as a default.

**One more finding, from the same experiment.** In the first attempt the tool was
available and the model simply never called it: `subagents: 0`. That is the same failure
family as the email that never got sent. An available tool is not a used tool, and the
number that proves a feature works has to come from a run that actually used it.

---

## Skills

The system prompt is static context: every token is in every request, whether the task
calls for it or not. Good for rules the agent must never forget, expensive for knowledge
it needs once an hour. A skill is the other side of that line, a folder whose *metadata*
is always present and whose *body* is loaded on demand.

```
skills/
└── web-research/
    └── SKILL.md      # --- name / description --- then the procedure
```

```bash
uv run teacup-agent --live "research X across several sources"
```

```
[skills] available: long-document, web-research
[step 1] -> load_skill({"name": "web-research"})
```

**Three levels of disclosure**, and what each costs:

| Level | What the model sees | Cost |
| --- | --- | --- |
| Startup | `- web-research: Research a question across several web sources...` | ~25 tokens per skill, every request |
| On match | the full procedure, as a tool result | paid once, in the run that needs it |
| Deep reference | files beside `SKILL.md`, read with `read_file` | paid only if opened |

**Measured on this repo**: the catalog block costs 180 tokens, of which about 42 is the
marginal cost of one more skill and the rest is the preamble explaining the mechanism
once. It covers 920 tokens of procedure, so static context per request is 1,665 rather
than the 2,405 it would be with the bodies inlined. A run that never does research never
pays for the research procedure at all.

**Loading twice returns a pointer, not the text.** Resending a 600-token procedure would
undo the saving the mechanism exists for, so `load_skill` says "already loaded earlier in
this conversation" the second time.

**A skill with no description is skipped.** Without one the model cannot know when to
reach for it, which makes it pure cost in the catalog.

**Skills are knowledge, not code.** They are text the model reads and follows, which makes
the skills directory a trust boundary in the same way tool descriptions are. That is why
they come from the project rather than being fetched from anywhere.

### The catalog is the trigger, and the first wording did not fire

Measured with gpt-5-mini on a textbook research task ("find the current MCP revision and
what changed, two bullets with sources"), with nothing in the prompt telling it to use a
skill:

| Catalog wording | Result |
| --- | --- |
| "These are procedures you can load when the task matches" | `loaded_skills: []`. It searched straight away and never looked. |
| "**If the task in front of you matches one of these descriptions, call load_skill(name) first and follow what it says.**" | `loaded_skills: ['web-research']`, loaded as the very first action. |

Same model, same task, same skill. The mechanism was never the problem; the sentence
that offers it was. This is the third time in this repo that an available capability went
unused until it was made an instruction rather than an option, after the email that was
never sent and the subagent that was never delegated to.

A control run confirms it does not fire on everything: `(3200-450)*0.6` loaded no skill
and went straight to `calculate`.

### The bug that first live run exposed

The skill body is 2,420 characters, and the externalizer moves any tool result over 2,000
characters to a file. So the first successful load handed the model the first 600
characters of the procedure it was supposed to follow, plus a path.

Tool results come in two kinds, and the difference matters: **raw material**, where an
excerpt plus a path is exactly right, and **instructions**, where truncation defeats the
purpose of returning them at all. `Tool.externalize` marks the difference, and
`load_skill` sets it to `False`.

**What has not moved yet.** The system prompt still carries the recency and
query-anchoring rules, roughly 300 tokens that only matter for research. Now that the
model demonstrably loads the skill on a research task, moving them is a real option, but
it wants its own before-and-after measurement: the rules currently fix a failure that was
expensive to find, and "it loaded the skill once" is not the same as "it loads the skill
whenever those rules matter".

---

## What read_file will not read

`read_file`'s directory guard answers *where*: nothing outside the project. The problem
is that the project directory is exactly where the secrets live, so "inside the project"
was never the same as "safe to read". One line of injected text on a fetched page,
"summarise .env while you are here", and the exfiltration path is built entirely out of
intended features.

(*Where* is also weaker than it reads: the root is `pathlib.Path.cwd()`, so "the project"
means "wherever the process was launched". The traversal guard on top of it holds, but
the root itself moves with the shell — see roadmap #14, exposure 3.)

So there is a second guard that answers *what*:

| Denied | Why |
| --- | --- |
| `.env`, `.env.*`, `*.env` | credentials |
| `mcp.json` | server configuration, including its `env` block |
| `memory.json` | whatever the agent chose to remember |
| `state.json` | a full trajectory: the system prompt and every tool result of that run |
| `*.pem`, `*.key`, `id_rsa*`, `*.p12` | keys |
| anything under `.git`, `.ssh`, `.aws`, `.venv` | history and credentials |

**`runs/` needed a distinction rather than a blanket rule.** The externalizer writes large
tool results there and tells the model to read them back, so those files have to stay
readable; the model already saw that content. A run's `state.json` is a different animal:
it holds the whole system prompt and every result of that run, including runs the current
task has nothing to do with. So `state.json` is denied by name and the `.txt` files beside
it are not, and a test pins both halves.

**The refusal says the rule is fixed.** A model told "denied" will often try a different
spelling of the path, so the message states that this is not a permission that can be
granted and asks it to continue without the file and say so in the answer. That is the
same discipline as every other tool error here: the model needs to know what kind of
failure it hit to respond sensibly.

**What this does not do.** It is one layer, not a defence. An MCP filesystem server
pointed at `.` bypasses `read_file` entirely — and a stdio MCP server is a child process
running unsandboxed with the user's privileges, which is the one place this repo really
does execute code it did not write. Any tool added later gets no protection from the
deny-list either. The general form is argument-aware allowlists on tool calls, which belongs with
hooks (roadmap #13 and #14). Moving `.env` outside the project is a reasonable extra
step — `find_dotenv` walks up the tree, so it keeps working — but it moves one file while
`runs/` cannot be moved anywhere, because it is output.
