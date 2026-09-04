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

## What #13 (Hooks) added, and why it is not a weaker approval gate

`hooks.py` (roadmap #13) lets a project ship a reviewed `hooks.py` that can veto a tool
call by argument (`before_tool_call`), rewrite a result (`after_tool_result`), and —
behind a new `--approve hooks` policy — approve a normally-gated call with nobody
watching (`approve_tool_call`). That third one is a new kind of power this repo did not
have before: the previous section's "denied by default when no human is watching" was
absolute. It still is, by default — `--approve hooks` is opt-in, and `approve_tool_call`
returning `None` (including "no `hooks.py` was loaded at all") falls straight through to
the same deny-without-a-TTY behavior. What changes is that a *specific project*, by
shipping a *file reviewed in version control*, can say in advance which of its own calls
need no human — the same shape of decision a human reviewer already makes once when they
add a tool to a CI pipeline's allowlist, just written down instead of re-decided by hand
every run.

Two things follow from that:

- **`hooks.py` is deliberately not gitignored**, unlike `mcp.json`/`agent.yaml`. Those two
  can carry credentials; `hooks.py` carries policy, and an unattended run trusting it to
  approve calls is exactly the kind of change that belongs in code review, not hidden
  from it.
- **`--approve hooks` cannot make a bad `hooks.py` safe.** It is a mechanism for a project
  to declare its own trust explicitly; it does not evaluate whether that trust was
  well-placed. A `hooks.py` that approves everything is indistinguishable, to the loop,
  from `--approve allow` — the safety was in the file's own content and the review it
  got, not in the mechanism's existence. Never point `--hooks`/`--approve hooks` at a
  file that arrived with the task itself (a cloned repo you did not audit, a prompt
  injected "here's a hooks.py to use") — loading one is always explicit and opt-in, never
  automatic from untrusted content.
- **Failure handling is asymmetric on purpose.** A broken `before_tool_call` fails
  **closed** (an exception becomes a veto — a broken safety check must not silently stop
  being one); a broken `approve_tool_call` fails to "no opinion" (also closed, since that
  already means deny without a TTY); a broken `after_tool_result` fails to a no-op (it is
  a transform, not a gate, so silence is the safe fallback).

`hooks.example.py` demonstrates the mechanism with the existing `send_email` tool and
zero new tools — roadmap #14's own suggested example, "send_email only to these domains."

## What #20 (coding tools) added, and why it changes the previous section's answer

Roadmap #20 added the first real code-execution tool this repo has ever shipped:
`list_files`/`edit_file`/`write_file`/`run_command` (`coding_tools.py`), opt-in behind
`--coding-tools`. Everything above this section was written when the honest answer to
"does this repo execute code?" was no (`calculate` is an `ast` walker, not `eval`); it no
longer is, and this section states what changed and what did not.

**What did not change**: the approval gate is still the real boundary, not a sandbox
(previous section). `edit_file`/`write_file`/`run_command` all carry
`requires_approval=True` and are denied by default with nobody watching, same as
`send_email` always was. `list_files` is the one ungated tool of the four, and it is
read-only and filtered through the same deny-list as `read_file`.

**What is new, and where the new exposure actually lives**:

- **`run_command` is `shell=True`.** Anything that ends up in the `command` string —
  including text a prior tool call returned, if a project's own logic ever built a
  command from tool output — is a potential shell-injection vector. This is not
  mitigated by the tool's own shape (an argv-only tool would only move the problem, not
  remove it, since the whole point of this tool is to run an arbitrary shell command);
  it is mitigated by the same layers as everything else in this file: the approval gate,
  and a project's own `hooks.py` allowlist recognizing specific safe invocations
  (`hooks.example.py`'s `git status`/`diff`/`add`/`commit`/`log` and `uv run
  pytest`/`ruff` patterns, with `git push` explicitly denied). **A prefix allowlist alone
  is not enough**, and an independent review of this PR caught the reason why:
  `shell=True` still honors chaining (`&&`, `;`, `|`) and substitution (`` ` `` / `$()`)
  after an allowed prefix — `git status && git push origin main --force` starts with
  "git status" and never reaches the `^git push\b` deny pattern at position 0.
  `hooks.example.py`'s `UNSAFE_SHELL_METACHARACTERS` closes this by refusing outright
  any `run_command` containing one of those characters, checked in `before_tool_call`
  (which runs before the approval gate, so this applies regardless of `--approve`
  policy) and again, defensively, in `approve_tool_call`. A `hooks.py` copied from this
  file inherits the guard; one written from scratch does not get it for free — prefix
  matching alone is not a safe allowlist shape without it. A project that enables
  `--coding-tools` without also writing a `hooks.py` allowlist for it gets exactly the
  old default back: every call denied without a human at a TTY.
- **`write_file`/`edit_file` reach the filesystem the project root exposes**, subject to
  the same traversal guard and deny-list `read_file` has always used
  (`tools._is_denied()`, `_resolve_in_project()` in `coding_tools.py` reuses both rather
  than re-deriving a second copy). `write_file` cannot overwrite an existing file at all
  — a structural guard, not a check that can be argued around — so a stale or truncated
  view of a file can never cause silent data loss through that tool; `edit_file` always
  re-reads the file fresh before matching, for the same reason.
- **`run_command`'s own timeout is stronger containment than the loop's generic
  per-call timeout.** `tool_timeout` (the loop's default per-call ceiling) can only
  abandon a stuck thread — the underlying process keeps running unless something else
  kills it. `run_command` passes its timeout straight to `subprocess.run`, which
  actually terminates the child process tree on expiry. This matters specifically
  because a runaway shell command (an infinite loop, a hung network call inside a test
  suite) is exactly the failure mode a generic thread-abandon cannot stop.
- **This closes the precondition, not the risk.** `AGENTS.md`'s rule — "a code-execution
  tool must not be added until a sandbox exists to run it in" — is satisfied by
  teacup-run's sandboxed subprocess launcher (credential scope, lifetime) plus this
  repo's own approval gate and hooks (what the agent is allowed to do inside the repo it
  is pointed at). It does not mean `run_command` is now safe to enable with
  `--approve allow` or no `hooks.py` at all — see "The approval gate is the real
  boundary, not a sandbox" above, which still applies in full to this tool.

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
- **`run_command`'s own execution.** #20 added a real code-execution tool
  (`coding_tools.py`, opt-in behind `--coding-tools`) — see "What #20 (coding tools)
  added" above for what changed and what did not. `calculate` remains a hand-written
  `ast` walker, not `eval`, and stays that way; it is `run_command`, not `calculate`,
  that is this repo's code-execution surface.
- **A denial or veto is not encryption.** Stopping a call from *executing* does not undo
  a model having already read something it should not have (e.g., from an MCP server with
  no deny-list of its own) and repeating it in its final answer. Output auditing (an
  `after_tool_result` hook, or a check on the final answer) is the layer that would catch
  that, and is not built as a default here — a project's `hooks.py` can add it.
- **teacup-run's sandbox does not isolate the filesystem or the network.** When this
  agent is launched by [teacup-run](https://github.com/rayhu/teacup-run)'s sandboxed
  subprocess backend, what it bounds is credential scope (an allowlisted environment) and
  lifetime (a hard timeout with a full process-tree kill) — not what the agent is allowed
  to do inside the repo it's pointed at. A credential-scoped, time-bounded process can
  still make arbitrary outbound calls and read/write anything its own tools reach; the
  approval gate and `hooks.py` above are what actually bound that, on either side of the
  integration.
