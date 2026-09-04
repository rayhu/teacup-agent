"""Routing — which model profile serves which role in one run.

An agent makes several *kinds* of model call, and they are not the same job. The
agent's own turns are judgment: decide what the task is, notice the approach is
wrong, know when to stop. Summarizing a transcript, writing a note in a fixed schema,
or running a short well-specified subtask are not — they are checkable work, and a
harness (retrieval instead of recall, tool feedback instead of one-shot correctness,
retries instead of precision) closes most of the capability gap on exactly that kind
of work. Sending all of them to one model means either paying judgment prices for
clerical work or accepting a clerk's judgment.

So: a **role** names a call site, `agent.yaml` maps roles to model profiles, and this
module is the lookup. There is no classifier here and no per-turn switching — that is
roadmap #21's Stage C, and it does not ship until Stage B has measured where the small
model actually breaks.

Two things this module exists to get right:

* **One model instance per profile, for the whole run.** A fresh instance per call
  would mean a fresh HTTP client and a fresh (empty) `cache_key` every turn, which
  quietly cancels the prompt-cache saving it was supposed to protect.
* **A bare `Model` still works everywhere.** `as_router()` wraps one into a router
  that answers every role with it, so `loop.run(model=...)`, every test and the
  offline demo are unchanged.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

# The call sites, and the whole set of them. Adding a role means adding a model call
# somewhere — if this tuple and the code disagree, the code is wrong. `judge` is the one
# a Router never serves: scoring a saved run happens in trajectory.py, outside any run,
# and builds that one model straight from the config.
ROLES = ("main", "plan", "compact", "reflect", "judge", "subagent")


class Router:
    """Maps a role to a model, building each profile at most once.

    `build` is a factory `profile name -> Model`; keeping it a callable is what keeps
    this module free of any knowledge of YAML, providers or API keys (that lives in
    `agent_config.build_router`) and makes it testable with already-built fakes.
    """

    def __init__(
        self,
        build: Callable[[str], Any],
        roles: Mapping[str, str] | None = None,
        default: str = "default",
        instances: dict[str, Any] | None = None,
    ):
        for role in roles or {}:
            if role not in ROLES:
                raise ValueError(f"unknown model role {role!r}; known roles: {list(ROLES)}")
        self._build = build
        self._roles = dict(roles or {})
        self._default = default
        # Shared with routers derived by child(), so a parent and its subagents reuse
        # one instance (and therefore one client and one cache key) per profile.
        self._instances: dict[str, Any] = {} if instances is None else instances

    def profile_for(self, role: str) -> str:
        """Which profile name serves this role. Unmapped roles fall back to the
        default profile, so a config that names no roles behaves exactly as before."""
        if role not in ROLES:
            raise ValueError(f"unknown model role {role!r}; known roles: {list(ROLES)}")
        return self._roles.get(role, self._default)

    def for_role(self, role: str) -> Any:
        name = self.profile_for(role)
        if name not in self._instances:
            self._instances[name] = self._build(name)
        return self._instances[name]

    def child(self, role: str) -> "Router":
        """A router for a child run (a subagent), whose `main` role is served by this
        router's `role` profile. Every other role keeps the parent's mapping, and the
        built instances are shared."""
        return Router(
            self._build,
            {**self._roles, "main": self.profile_for(role)},
            self._default,
            self._instances,
        )

    def profiles(self) -> dict[str, str]:
        """role -> profile for every role, for logging and run records."""
        return {role: self.profile_for(role) for role in ROLES}

    def is_split(self) -> bool:
        """True when more than one profile is in play — worth printing, not worth
        printing when there is nothing to say."""
        return len(set(self.profiles().values())) > 1


def single(model: Any, name: str | None = None) -> Router:
    """A router that answers every role with one already-built model.

    The profile name defaults to the model's own name when it has one, so the spend
    breakdown of a single-model run still reads as something rather than "default".
    """
    label = name or getattr(model, "model", None) or "default"
    return Router(lambda _: model, default=label)


def from_models(
    models: Mapping[str, Any], roles: Mapping[str, str] | None = None, default: str | None = None
) -> Router:
    """A router over already-built instances, keyed by profile name — what tests and
    evals want, and what `agent_config.build_router` would be if models were free."""
    if not models:
        raise ValueError("from_models() needs at least one model")
    chosen = default or next(iter(models))
    if chosen not in models:
        raise ValueError(f"default profile {chosen!r} is not one of {sorted(models)}")
    for role, profile in (roles or {}).items():
        if profile not in models:
            raise ValueError(f"role {role!r} names profile {profile!r}, which is not one of {sorted(models)}")
    return Router(lambda name: models[name], roles, chosen, instances=dict(models))


def as_router(model: Any) -> Router:
    """Accept either a Router or a bare Model. The whole point of the adapter: no
    existing caller of `loop.run(model=...)` has to learn about roles."""
    return model if isinstance(model, Router) else single(model)
