# Threat model

What this repo trusts, what it does not, and what a fork inherits if it points this agent
at something new. `docs/roadmap.md` #14 is the design-decision record — why each of these
was worth doing and what it cost to find; this file is the current, standing statement,
kept short enough to actually read before forking.

## What is trusted, what is not

**Trusted**: the goal text the human gives it, the system prompt, local files under the
project root minus the deny-list below, and the human behind the approval gate.

**Not trusted, ever**, because the model reads it as ordinary text and nothing stops an
instruction hiding inside it: web search results, any MCP tool's description or the
content it returns, and — since `teacup-agent-serve` (#18) — the task text an inbound
Agent2Agent request carries. All four are exactly the "prompt injection" surface: text
that arrives from outside the human's own words and that the model cannot distinguish
from an instruction.

## The three original exposures

1. **`.env` exfiltration.** Before a deny-list existed, `read_file`'s only guard was
   "stay inside the project directory" — and the project directory is where the API keys
   live. Combined with `send_email` under `--approve allow`, or simply a model that quotes
   a secret into its answer, that was an exfiltration path built entirely from intended
   features. **Fixed**: `read_file`'s deny-list (`tools.py`'s `DENIED_FILES`/
   `DENIED_DIRS`) covers `.env*`, `mcp.json`, `memory.json`, `state.json`, key files, and
   `.git`/`.ssh`/`.aws`/`.venv`, returning an `ERROR:` that states the rule is fixed so the
   model does not retry with another spelling.
2. **MCP servers are third-party processes with our user's privileges.** The SDK passes a
   minimal environment (notably not `OPENAI_API_KEY`), but the child still has the
   filesystem and the network. **Accepted, not fixed**: a stdio MCP server is unsandboxed
   code execution, spawned by us, holding our user's privileges. This stays out of scope
   by cost — a Seatbelt or bubblewrap wrapper around the child command is real work for
   little payoff in a teaching repo — not because there is nothing to isolate. **If you
   fork this and point it at MCP servers you did not write, this is the sentence to
   re-read**: you are trusting that server's author the same way you trust code you run
   directly.
3. **The project-root boundary moved with the shell.** `read_file`'s root used to be
   `pathlib.Path.cwd()`, recomputed on every call — launched from `~`, everything under
   the home directory was "inside the project" except what the deny-list happened to
   catch. **Fixed**: the project root is now an explicit, settable fact of the run
   (`tools.set_project_root()`, wired from `cli.py`'s `--project-root`, defaulting to the
   launch directory unchanged), so the boundary is a stated property of the run rather
   than an accident of where the shell happened to be.

## The approval gate is the real boundary, not a sandbox

"Sandbox" bundles four properties that different tools buy separately:

| Property | The question it answers | Bought by |
| --- | --- | --- |
| Isolation | can it damage the host? | Seatbelt, bubblewrap, containers, microVMs |
| Reproducibility | does it see the same world every run? | an image plus a lockfile |
| Reversibility | can what it did be undone? | a git worktree, or a container over a *copied* tree |
| Credential and egress scope | which secrets and which network does the process hold? | none of the above |

Every exposure this repo actually defends against lives in the last row, which is why the
answer here is deny-lists and an approval gate rather than a container — a container
holding `OPENAI_API_KEY` with open egress is a perfectly good exfiltration channel. `Tool
.requires_approval`, denied by default when no human is watching (`AGENTS.md` rule 4), is
the real boundary: `send_email`, `delegate_a2a` and anything an MCP server does not mark
read-only all sit behind it.

## What #18 added, and how it is covered

`teacup-agent-serve` is a new inbound surface: another process can submit a task over the
network. It does not get a separate trust story — inbound tasks run through the exact same
`deny_all`-by-default approval gate as any local run (`cli._make_approver`, reused
unmodified). A remote caller cannot trigger `send_email` or `delegate_a2a` without this
instance's own approval policy allowing it. Concurrency is deliberately serialized (one
task at a time) because `tools_mod.REGISTRY` is process-global state shared with
`skills`/`subagent`, which is a correctness limit stated in `docs/design-notes.md`, not a
trust boundary.

## What this repo does not defend against

Stated plainly rather than silently assumed:

- **Compromise of the model provider itself.** If the API you call is compromised or
  malicious, nothing here helps.
- **Supply-chain compromise of this repo's own dependencies** (`openai`, `mcp`, `a2a-sdk`,
  `pyyaml`, and everything they pull in). Pinned versions and a lockfile slow this down;
  they do not stop it.
- **Network-level attacks** against the machine this runs on — this is an application
  boundary, not a network one.
- **Multi-tenant hosting.** `docs/roadmap.md`'s "Deliberately not doing" names one narrow,
  explicitly opt-in exception (`teacup-agent-serve`, behind a separate install extra and a
  separate console script); nothing here is designed to safely serve untrusted multiple
  tenants beyond that.
- **A code-execution tool.** None exists (`calculate` is a hand-written `ast` walker, not
  `eval`), and the rule stands: a code-execution tool must not be added until a sandbox
  exists to run it in.
