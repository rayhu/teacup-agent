# mini-agent

A small AI agent with nothing stubbed out, built to turn one formula into code you can
run:

```
Agent = Model + State + Tools + Control Loop + Memory/Evals
```

Every part does real work, and each is kept to a few dozen lines so you can replace them
one at a time. The control loop is about 40 lines and fits on one screen. That is the
point of the project.

## Quick start

```bash
uv sync                      # create the environment (.venv + uv.lock)

uv run mini-agent            # offline demo: no API key, no cost, instant
uv run mini-agent "compute (3200-450)*0.6 and tell me what CUDA is"

uv run mini-agent --live "research NVIDIA's GPU strategy"   # real OpenAI call, needs .env

uv run python -m mini_agent.evals   # offline assertions about the control loop
uv run pytest                       # those cases plus unit tests
```

`--live` reads `OPENAI_API_KEY` from `.env`:

```bash
cp .env.example .env
```

## The loop

```python
while True:
    if out of steps, budget or time:      # four brakes, then a forced wrap-up
        break
    reply = model.complete(messages, tools)
    messages.extend(reply.items)          # trap 1: the call goes back before its results
    if not reply.tool_calls:              # trap 3: this is the only "done" signal
        break
    for call in reply.tool_calls:         # trap 2: every id gets exactly one result
        messages.append(result_of(call))
```

That is the whole idea. Everything else in the repo exists to keep this loop honest once
a real model meets a real network.

## Layout

```
src/mini_agent/
├── model.py       Model        the only part that thinks: Responses / Chat / scripted
├── state.py       State        goal, messages, steps, budget, status, tool trace
├── tools.py       Tools        function + JSON Schema + safe execution
├── memory.py      Memory       short term = messages; long term = memory.json
├── loop.py        Control Loop LLM -> tool call -> tool result -> LLM
├── evals.py       Evals        loop health, scripted model, free to run
├── context.py     context      externalize big results, compact history over the limit
├── plan.py        checklist    split the goal into items and hold the run to them
├── persist.py     persistence  save state every step, resume from it
├── mcp_tools.py   MCP          borrow tools from MCP servers
├── trajectory.py  scoring      grade a real run: mechanical metrics + an LLM judge
└── cli.py         entry point
tests/                  pytest: the eval cases plus unit tests
examples/               runnable demos, starting with approval_demo.py
mcp.example.json        template for MCP servers
CLAUDE.md               how to work in this repo
docs/design-notes.md    why each subsystem behaves the way it does
docs/roadmap.md         what is missing, and in what order to add it
NOTES.md                the original study notes this grew from
```

## The five parts

| Part | File | Today | When you outgrow it |
| --- | --- | --- | --- |
| Model | `model.py` | Responses API (default) and Chat Completions with cache-aware pricing, plus an offline scripted model | Claude, local models, multi-model routing |
| State | `state.py` | steps, budget, time, status machine, tool trace; saved every step and resumable | distributed or concurrent runs |
| Tools | `tools.py`, `mcp_tools.py` | six built-ins (search, calculate, read file, remember, checklist, send mail) plus anything an MCP server exposes | more servers |
| Control Loop | `loop.py` | four brakes, parallel tool execution, retries, a completion check | subagents, delegated planning |
| Memory | `memory.py` | JSON file with dedupe, keeping the last N facts | vector store, summarization, relevance recall |
| Evals | `evals.py`, `trajectory.py` | offline protocol cases plus real-trajectory scoring | fixed task suites, cross-version regression |

## What keeps it honest

Each of these earned its place by fixing a run that had gone wrong. The stories are in
[docs/design-notes.md](docs/design-notes.md).

- **Four brakes** (steps, budget, wall clock, tool calls per turn). Hitting one triggers a
  forced wrap-up, so a run never ends empty-handed.
- **A checklist** built from the goal, so a two-part request cannot report `done` with one
  part missing.
- **An approval gate** on tools with side effects, denied by default when no human is
  watching.
- **Context management**: large tool results go to disk and the context keeps an excerpt
  plus a path; older history is compacted only at points where no tool call is left
  dangling.
- **Two kinds of evaluation**: protocol cases with a scripted model (free, must stay
  green), and trajectory scoring of real runs, which asks whether it delivered, whether it
  invented citations, and whether it ever attempted the action it was asked for.
- **Persistence**: every step is written to `runs/<timestamp>/state.json`, and `--resume`
  continues from there.
- **MCP**: point at a server and its tools join the registry, namespaced and gated.

## The three traps in the control loop

All three are marked in the code and covered by `evals.py`:

1. **Order.** The assistant message carrying `tool_calls` must go back into `messages`
   before the tool results. Reversed, the next API call errors.
2. **Count.** One turn may contain several `tool_call`s, and every `tool_call_id` needs its
   own result message. Miss one and the next request is a 400.
3. **Termination.** There is no mysterious `check_completion()`. A run ends when the model
   stops requesting tools, or when a ceiling is hit.

There is also a fourth rule, less a trap than the whole design: when a tool fails, the
error text goes back **as the tool result** so the model can correct itself. That is not
defensive programming. It is the reason the loop exists.

## Where this sits today

The core is not dated. An agent in 2026 is still this loop. The engineering around it has
come a long way: Responses API, prompt caching, context management, parallel execution,
persistence and resume, an approval gate, trajectory scoring and MCP are all in place.
Subagents with isolated context and a serious search backend are not.

See [docs/roadmap.md](docs/roadmap.md) for what is missing, why it matters, and what order
to add it in.
