# Intent

What this project is for, and what a change to it can fail against.

`README.md` says what teacup-agent *is*; `docs/roadmap.md` says what is missing. This
file says what the thing is **for**, which is what a fork needs in order to tell whether
its change is an improvement or merely a difference. It is stated once, not per change:
a roadmap item that cannot be argued from this file is probably an item this project
should not do.

---

## The one sentence

**A complete agent small enough to read in one sitting, and honest enough to fork.**

```
Agent = Model + State + Tools + Control Loop + Memory/Evals
```

Every part does real work; none is a stub. The control loop is about 40 lines because a
harness you cannot read is a harness you cannot trust, and trusting it is the whole
point.

## What it is for, in order

1. **To be read.** Someone who has used agents but never built one should be able to
   read `loop.py` and see the entire mechanism, including the three protocol traps that
   make real runs fail.
2. **To be forked.** The repo is MIT-licensed teaching material that expects to be
   copied. A fork changing the model, the skills and the ceilings should be editing
   *settings*, not surgery on Python.
3. **To be run.** Offline, instantly, for free, before anyone types an API key. Live,
   with a real model, on a real network, without changing the code.
4. **To be measured.** A fork that claims to be better should be able to show it — with
   `evals.py` for the loop's behaviour and `trajectory.py` for a real run — rather than
   saying "I changed the prompt and I think it works better".

Everything else is negotiable. These four are the product.

## Who it is for

One person who wants to understand agents by owning one, and the fork they make of it
afterwards. Not a platform team, not a twelve-person org, not production traffic —
`docs/roadmap.md` has a "Deliberately not doing" section that says so in more detail,
and the rule behind it is that a capability which makes the loop unreadable costs more
than it adds.

## What "forkable" has to mean

Forking is the metric this project actually cares about: **how often does someone take
this agent, improve it, and hand the improvement on?** That sets a bar the repo does not
yet clear.

Today a fork inherits an agent whose configuration is spread across three places: the
defaults in `cli.py`, the defaults in `loop.run`, and the flags the author happened to
remember from the README. Nothing in the repository states, in one place a human can
read and edit, *which model this agent uses, which skills it may load, what it is allowed
to spend, and what it has been told to do.* The answer exists only as an invocation
somebody typed once.

So the intent is:

> **An agent should be describable by a single markdown file** — the model it uses, the
> skills it may load, the ceilings it runs under, and the instructions it follows — and
> forking it should mean editing that file.

Data, not code. A fork that changes the model and adds a skill should be a diff a
reviewer can read in ten seconds, and a diff whose meaning does not depend on knowing
Python.

The format for that file is specified in [spec-agent-md.md](spec-agent-md.md) and
tracked as roadmap item #15.

## Where teacup-run fits

[teacup-run](https://github.com/rayhu/teacup-run) is the sibling project: a library and
registry for discovering, running, extending, evaluating and publishing agents —
`AutoAgent.from_pretrained("alice/deep-research")`, then fork, evaluate, publish. It is
the distribution layer.

The two are meant to pair, with a clean split:

| | teacup-agent | teacup-run |
| --- | --- | --- |
| Is | the runtime — a loop you can read | the ecosystem — pull, fork, eval, publish |
| Owns | how a turn is executed: brakes, protocol, approval, context, skills, MCP | how an agent is *named, shipped and compared* |
| Unit | a run | a package (a directory) |
| Answers | "what happens when this agent does something" | "where did this agent come from, and is my fork better" |

The seam between them is one file. teacup-run's package format already reserves
`AGENT.md` at the root of an agent directory and already carries a `framework:` field
whose stated purpose is to let a package declare a different runtime later. teacup-agent
declaring `framework: teacup-agent` and reading that same file is how the two meet:
**teacup-run says which agent, teacup-agent says what running it means.**

Two rules keep the pairing from becoming a dependency:

- **teacup-agent never imports teacup-run.** A fork must run with `uv run teacup-agent`
  and nothing else — no hub, no registry, no network. If the ecosystem disappears, the
  agent still runs.
- **The file is the contract, not the code.** teacup-agent implements the format;
  teacup-run reads and writes it. Neither reaches into the other's internals, and the
  format is specified in a document that outlives both implementations.

The reverse direction — teacup-run treating this repository as a pullable package — is a
teacup-run change and is deliberately not a criterion for anything landing here. See the
compatibility contract in [spec-agent-md.md](spec-agent-md.md).

## What a change can fail against

These are the criteria a fork owes this project. They are not style preferences; each one
has a mechanism behind it, and most were paid for by a run that went wrong (the stories
are in [design-notes.md](design-notes.md) and the "Field patches" section of
[roadmap.md](roadmap.md)).

1. **The loop still fits in one head.** If a feature makes `loop.py` harder to read, it
   belongs in a backend class, a module of its own, or the roadmap — not in the loop.
2. **The offline demo stays instant and free.** `uv run teacup-agent` with no key, no
   network, no writes into the repo. It is the first thing a new reader runs.
3. **The three verification commands stay green**, and their real output is quoted, not
   summarized. A part that was not verified is named, with the reason.
4. **Nothing is silently half-done — and nothing silently falls back.** A broken tool
   must not read as "this does not exist"; a config that fails to load must not read as
   "no config". Loud failure beats a plausible answer built on nothing.
5. **Deny by default when nobody is watching.** Side-effecting tools need a human. "No
   TTY, so allow it" is the most dangerous default there is.
6. **Names must not lie about behaviour.** A flag or a key that is inert in half the runs
   is worse than one that says what it does.
7. **Static context stays expensive on purpose.** Anything loaded every turn is paid for
   every turn; the default answer is dynamic context — a tool result, a skill body, an
   externalized file.
8. **No new required dependency without an argument in the diff.** Four runtime
   dependencies today. The fifth needs to be worth more than the explanation it costs.
9. **Every rule here is checkable.** If a criterion cannot be tested, evaluated, or at
   minimum shown in output, it is a preference and should be labelled as one.

## What failure looks like

Stated plainly, so it can be noticed early:

- Nobody forks it, because reading it is not the same as being able to change it.
- A fork exists but cannot say whether it is better, because nothing was measured.
- The loop grew a framework around it, and the 40 lines are now 400.
- The repo describes an agent that is different from the one it actually runs.

The last one is the failure this project is currently closest to, and item #15 exists to
close it.
