# Spec: AGENT.md — an agent described by one markdown file (v0.1)

**Status**: proposed, not implemented. This is the plan for roadmap item
[#15](roadmap.md), written before the diff so the design argument happens while it is
still cheap to lose.
**Why**: [intent.md](intent.md) — "an agent should be describable by a single markdown
file, and forking it should mean editing that file."
**Companion**: teacup-run's package format, which reserves `AGENT.md` at an agent's root.

---

## The problem

An agent in this repo is currently defined by an invocation:

```bash
uv run teacup-agent --live --model gpt-5-mini --budget 0.50 --max-steps 12 \
  --skills skills --plan on --subagents "research X"
```

That command is the agent. Nothing in the repository records it. A fork inherits the
defaults in `cli.py`, the defaults in `loop.run`, and whichever flags its author
remembered — three sources, none of them stating what *this* agent is supposed to be.
Change the model and you have edited nothing; you have typed something different.

What is wanted instead is one file, in the repository, that a fork edits:

```markdown
---
name: teacup/deep-research
description: Research a question across several sources and answer with graded citations.
framework: teacup-agent
model: gpt-5-mini
skills: [web-research, long-document]
budget: 0.50
max_steps: 12
plan: on
---

You are a research agent. Prefer primary sources, grade every source as it arrives, and
never present an aggregator's number as a filing's number.
```

Fork it, change two lines, and the diff says exactly what changed.

## The file

`AGENT.md` is **frontmatter plus a body**, the same shape `SKILL.md` already uses in this
repo and in teacup-run:

- **The frontmatter is the manifest** — machine-readable settings, one key per CLI flag.
- **The body is the instructions** — text the model receives, appended to the system
  prompt.

One file, because the thing a fork most wants to change (the model, the skills, the
budget, what the agent is told to do) should not be spread across a manifest, a prompt
file and a README.

### The accepted frontmatter subset

The frontmatter is a **flat YAML mapping**: `key: value` lines, plus inline lists in
square brackets. Nothing else.

```yaml
key: value            # scalar: string, number, true/false
key: [a, b, c]        # inline list of scalars
key: []               # empty list
# comment lines and blank lines are ignored
```

Not accepted, in v0.1: nested mappings, block lists (`- item` on its own line),
multi-line strings, anchors, references. Each is a **hard error naming the line**, not a
silent skip.

Two reasons for the restriction, and they pull the same way:

1. **No new dependency.** `skills.py` already parses frontmatter by hand rather than
   adding PyYAML to read two fields (roadmap #12), and this repo has four runtime
   dependencies it can explain. The flat subset is about thirty lines of parser.
2. **It is still valid YAML.** A flat mapping with inline lists is exactly what
   `yaml.safe_load` produces the same result for, so teacup-run — which already depends
   on PyYAML — reads the identical file with a real parser. The subset is a restriction
   on what an author may *write*, not a private dialect.

**A file that does not parse must never fall back to defaults.** This repo has already
been burned once by exactly that failure mode: `@AGENTS.md.` written inline imports
nothing, and the session then answers from an empty ruleset without saying so
(`docs/workflow.md`). A config that quietly loads nothing produces a run that looks
normal and is not the agent anybody described. Every failure below is fatal, before the
first model call, with a message naming the file and the line:

| Failure | Message says |
| --- | --- |
| `--agent` names a file that is not there | the path, and that no default was substituted |
| No `---` frontmatter block | that this file is a card, not a manifest |
| A line outside the accepted subset | the line number and what is accepted |
| An unknown key | the key, and the list of keys that exist |
| A value of the wrong type | the key, what was given, what was expected |
| A skill named that has no `SKILL.md` | the name, and the skills that are available |
| A tool named that is not in the registry | the name, and the tools that exist |
| A refused key (below) | why a file may not set it |

Unknown keys are an error rather than a warning because the alternative is a fork whose
`buget: 5.00` runs at four cents and reports success.

### The keys

Every key that has a CLI flag is named after it. That is the whole naming rule:
`--max-steps 12` is `max_steps: 12`, and a reader who knows one knows the other. The keys
with no flag are the ones that describe the agent's identity rather than a run.

| Key | Flag | Default today | Meaning |
| --- | --- | --- | --- |
| `name` | — | directory name | identity; printed at run start, used by teacup-run |
| `description` | — | empty | one line: what this agent is for |
| `version` | — | none | package version, for teacup-run |
| `framework` | — | `teacup-agent` | which runtime executes this package |
| `derived_from` | — | none | lineage: the agent this was forked from |
| `model` | `--model` | `gpt-5` | model used **when live**; see "capability" below |
| `api` | `--api` | `responses` | `responses` or `chat` |
| `skills` | `--skills` | `./skills` when it exists | restrict the catalog to these names; `[]` = none |
| `tools` | *(new)* | every built-in | restrict-list of tool names the model may see |
| `mcp` | `--mcp` | `./mcp.json` when it exists | MCP config path, or `off` |
| `budget` | `--budget` | `0.05` | spending ceiling, USD |
| `max_steps` | `--max-steps` | `8` | loop turns |
| `max_tool_calls` | `--max-tool-calls` | `3` | tool calls executed per turn |
| `deadline` | `--deadline` | `600` | wall-clock ceiling, seconds |
| `tool_timeout` | `--tool-timeout` | `30` | per tool call, seconds |
| `context_limit` | `--context-limit` | `30000` | compact the history above this |
| `plan` | `--plan` | `auto` | build the upfront checklist |
| `subagents` | `--subagents` | `false` | offer the `delegate` tool |
| `subagent_steps` | `--subagent-steps` | `4` | step ceiling for one child run |
| `memory` | `--memory` | `memory.json` | long-term memory file |
| `search` | `--search` | `auto` when live | `auto` / `web` / `offline`; see "capability" below |
| `instructions` | — | none | when present, *that file* is the prompt and the body is prose |
| `model_fallback` | — | none | model to fall back to; carried for teacup-run, unused here in v0.1 |
| `approve` | `--approve` | `auto` | **only `deny` is accepted from a file**; see "capability" below |

This table is the authority: a key that is not in it is a fatal unknown key, and a key
mentioned anywhere else in this document has a row here.

`tools:` is the only key with a behaviour and no flag behind it today. It maps onto the
`exclude_tools` parameter `loop.run` already takes: the file lists what the agent *may*
use, the loader subtracts that from the registry. A restrict-list, never an add-list — a file cannot
introduce a tool that does not exist in the code.

Two details the implementation has to get right, because both are silent when wrong. The
list is resolved after MCP servers have connected, so it governs their namespaced
`server__tool` names too. And `update_todo` is **exempt**: it is the checklist's
mechanism, not a capability, and `tools: [search_web, calculate]` must not quietly
disable the push-back that stops a half-finished run reporting `done` (design rule 5).
`load_skill` and `delegate` are not in the registry yet when the list is resolved; they
are governed by `skills:` and `subagents:`, which is where a reader would look for them.

**Not in the file, deliberately**: `--run-dir`, `--resume`, `--quiet`, `--live` and the
goal itself. Those describe *this run*, not this agent. An agent that hard-coded its own
run directory would fight every invocation of it.

### The body

The body — everything after the closing `---` — is appended to the system prompt as the
agent's instructions. Two rules, both inherited:

- **It never replaces the system prompt.** The prefix carries today's date, the run
  status, and the fact that nobody will answer a question the model asks (Field patches A
  and B in the roadmap: both were runs that failed for want of exactly that). A file that
  could drop them would reintroduce two bugs this repo has already paid for.
- **It is appended to the prefix, not spliced into it.** Per-turn notes go at the end so
  the cached prefix stays byte-identical (design rule 6). The agent body is static, so it
  joins the prefix once and never moves.

Where exactly: `loop.run` builds the system message as `SYSTEM_PROMPT.format(...)`, then
appends the recalled memory, then the skills catalog. The body goes **between the format
and the recall** — the two blocks after it vary between runs, and putting a static block
after a varying one means a changed memory file invalidates the cache for the agent's own
instructions as well.

If the frontmatter carries `instructions: <path>`, that file is the instructions and the
body is treated as human-facing prose. This is the escape hatch for teacup-run's existing
layout, where `AGENT.md` is a card for humans and `prompts/system.md` holds the prompt.
One rule, no magic headings: *if `instructions:` names a file, that file is the prompt;
otherwise the body is.*

## Discovery and precedence

`--agent <path>`, defaulting to `./AGENT.md` when that file exists, and `off` to skip it.
Same opt-in convention as `mcp.json` and `skills/`: the file's existence *is* the opt-in,
because a project that has one wants it, and one that does not should not need a flag to
say so.

Precedence, plainly:

```
CLI flag  >  AGENT.md  >  built-in default
```

No clamping, no floors, no asymmetry. The person at the terminal is the last word;
"the file may lower a budget but not raise it" is the kind of clever rule that produces a
run nobody can explain.

**The implementation trap this creates**: with `argparse` defaults as they are today
(`--model` defaults to `gpt-5`, `--budget` to `0.05`), the loader cannot tell "the user
asked for gpt-5" from "the user said nothing". Applied naively, every flag silently
overrides the file and `AGENT.md` becomes decorative — a name that lies about behaviour,
which is the failure this repo names in `AGENTS.md`. So the flags whose keys appear above
change to `default=None`, and the resolved value is computed once, in one place, as
`flag if flag is not None else file_value if present else default`. The defaults move
from `argparse` into that resolver, and `--help` keeps them in its text.

This is broad in `cli.py` and almost invisible everywhere else. `loop.run` gains
**exactly one parameter** — `instructions: str | None = None`, the body, appended where
the previous section says — and nothing else in it changes: every other key maps to a
parameter it already takes, `exclude_tools` included. The control loop itself does not
move, which is the strongest argument this item has.

## What a file may do to capability

An `AGENT.md` can arrive by fork, by clone, or one day by `from_pretrained`. Its body is
instructions the model will follow and its frontmatter decides what the run may reach, so
it sits on the same trust boundary as a skill or a tool description. One rule:

> **A file may restrict or request capability. It may never grant it.**

Which settles the cases that would otherwise need arguing:

- **`approve:` accepts only `deny`.** The approval policy for side-effecting tools comes
  from the human at the terminal (design rule 4). `approve: allow` in a file is refused
  with a message saying so, because a pulled artifact that unlocks its own send-email
  tool is precisely the attack.
- **The file cannot turn `--live` on.** `model:` names what to use *when* the run is
  live. Spending money stays a decision someone makes at the command line.
- **`search:` is honoured only when the run is live.** An offline run stays offline
  whatever the file says. Same class as `--live` — `search: web` in a root `AGENT.md`
  would otherwise make `uv run teacup-agent` with no key and no flag reach the network,
  which is the one thing the offline demo promises it does not do.
- **`mcp:` is honoured only when the named file already exists locally.** Connecting
  starts third-party processes; a file may point at a config, it may not cause one to be
  fetched or written.
- **`tools:` and `skills:` subtract, never add.** Both name things that must already
  exist in the code and on disk; a name that does not resolve is an error, not an
  install.

## Compatibility with teacup-run

The contract, stated so both sides can implement against it rather than against each
other's internals.

**What teacup-agent guarantees**: an `AGENT.md` written to this spec parses as a flat
YAML mapping with `yaml.safe_load`, and the keys mean what the table above says. That is
the whole surface. teacup-agent does not import teacup-run, does not know about hubs, and
runs the same whether one exists or not.

**What teacup-run has to do**, and why this is *not* in item #15's done criteria: the two
formats are not accidentally compatible. `AgentSpec.load` requires `name`, `version`, and
`model` **as a mapping** with `model.primary`; a flat `model: gpt-5-mini` raises
`ManifestError` today. So teacup-run needs a `from_markdown` adapter, roughly:

| `AGENT.md` (flat) | `agent.yaml` (nested) |
| --- | --- |
| `model` | `model.primary` |
| `model_fallback` | `model.fallback` |
| `budget` | `budget.default_usd` |
| `max_tool_calls` | `budget.max_tool_calls` |
| `deadline` | `budget.max_wall_clock_s` |
| `derived_from` | `lineage.derived_from` |
| `instructions` | `instructions` |
| `name`, `version`, `description`, `framework`, `tools`, `skills` | same names |

`framework: teacup-agent` is the join. teacup-run's manifest already carries that field
with a comment saying it exists so a package can declare a different runtime later; this
is that later. No parallel `runtime:` key is invented.

**Two things this spec does not claim.**

*This repository is not a teacup-run package.* teacup-run's format reserves `agent.yaml`,
`prompts/`, `skills/`, `tools.py`, `checks.py` and `evals/` at an agent's root, and this
repo has `skills/` plus `tools.py` and `evals.py` as *modules of the runtime*. An
`AGENT.md` at the root describes the default agent; making the root a pullable package
would be a restructure, and it is not part of this item.

*teacup-run's `goal:` and `checks:` are not adopted.* Deterministic predicates over a
run's output are teacup-run's completion mechanism; this repo's is the checklist plus the
model no longer asking for tools (design rule: "there is no mysterious
`check_completion()`"). Carrying both concepts in one format would blur which one ends a
run.

### AGENT.md next to AGENTS.md

The repo root will hold two files whose names differ by one letter and whose meanings are
unrelated:

| File | Read by | Says |
| --- | --- | --- |
| `AGENTS.md` | coding agents working **on** this repo | conventions, rules, how to verify |
| `AGENT.md` | teacup-agent, and teacup-run | what the agent this repo **runs** is |

The collision is real and cannot be renamed away: `AGENTS.md` is the cross-tool
convention Claude Code and Codex look for, and `AGENT.md` is the name teacup-run's format
already reserves. The mitigation is a first line in each pointing at the other, and this
table.

## What counts as done

1. `AGENT.md` at the repo root describing the current default agent, and a test asserting
   it **reproduces today's built-in defaults** — so the file cannot drift from the agent
   it claims to describe.
2. A loader module (`agent_md.py`, roughly 120 lines including errors) parsing the subset
   above. Not in `loop.py`.
3. `--agent <path|off>`, defaulting to `./AGENT.md` when present.
4. Unit tests: the accepted subset; each failure in the table producing an actionable
   message; precedence in all three directions (flag wins, file wins, default wins);
   refused capability keys.
5. **An eval**, not only parser tests. A scripted-model run configured from a fixture
   `AGENT.md` must demonstrably honour it — the declared ceilings appear in the state, a
   declared skill is in the catalog and an undeclared one is not, the body reaches the
   system prompt after the stable prefix. Parser tests prove the parser; only an eval
   proves the wiring landed.
6. A second example agent under `examples/`, differing from the default in model, skills
   and budget — the fork story, demonstrated rather than described.
7. `README.md` gains the file; `docs/design-notes.md` gains the section on why the
   frontmatter is flat and why the body is appended.
8. The three verification commands green, with output quoted. `uv run teacup-agent` with
   no arguments still runs the offline demo instantly, and adding an `AGENT.md` to the
   root must not change that.

## Out of scope for v0.1

Named so nobody has to discover it: no publishing or registry, no fetching an agent by
name, no lineage graph, no tools or checks defined *in* markdown (code stays Python), no
per-skill configuration, no nested frontmatter, no environment declarations beyond what
`.env` already does, and no adapter for another framework's agent.

## Open questions

- **The body's meaning differs between the two repos.** Here it is the prompt; in
  teacup-run's example, `AGENT.md` is a human card and the prompt is `prompts/system.md`.
  The `instructions:` key bridges it, but a package with a card-style body and no
  `instructions:` key would feed the card to the model. v0.1's answer is that teacup-agent
  refuses an `AGENT.md` with no frontmatter at all; whether teacup-run should converge on
  "body is the prompt" is a teacup-run decision, and the honest place to make it is when
  its `from_markdown` lands.
- **Does `tools:` here become the allowlist roadmap #14 wants?** It looks like the same
  mechanism seen from the configuration side. #14 should decide, not this item.
