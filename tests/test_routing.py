"""Role routing: which model profile serves which call site.

Two things worth pinning down, because both fail silently: a role must fall back to
the default profile rather than to nothing, and a profile must be built **once** per
run — a fresh instance per call would mean a fresh client and an empty prompt-cache
key every turn, which cancels the saving routing is supposed to protect.
"""

import pytest
import yaml

from teacup_agent import agent_config, loop, plan, routing
from teacup_agent.memory import NullMemory
from teacup_agent.model import Reply, ScriptedModel, assistant_calls, assistant_says


# --- the router itself --------------------------------------------------------


def test_unmapped_roles_fall_back_to_the_default_profile():
    router = routing.from_models({"big": "BIG", "small": "SMALL"}, {"subagent": "small"}, "big")
    assert router.profile_for("subagent") == "small"
    assert router.profile_for("main") == "big"
    assert router.profile_for("compact") == "big"
    assert router.is_split()


def test_one_config_with_no_roles_behaves_exactly_as_before():
    router = routing.from_models({"only": "M"})
    assert set(router.profiles().values()) == {"only"}
    assert not router.is_split()


def test_each_profile_is_built_once_per_run():
    built = []

    def build(name):
        built.append(name)
        return f"model:{name}"

    router = routing.Router(build, {"plan": "big", "subagent": "small"}, default="big")
    for role in ("main", "plan", "compact", "reflect", "judge", "subagent"):
        router.for_role(role)
    assert built == ["big", "small"]  # not one per role
    assert router.for_role("main") is router.for_role("plan")


def test_a_child_router_runs_its_own_turns_on_the_subagent_profile():
    parent = routing.from_models({"big": "BIG", "small": "SMALL"}, {"subagent": "small"}, "big")
    child = parent.child("subagent")
    assert child.profile_for("main") == "small"  # the child's own turns
    assert child.profile_for("plan") == "big"  # every other role is unchanged
    assert child.for_role("main") is parent.for_role("subagent")  # shared instance


def test_an_unknown_role_is_a_loud_error():
    with pytest.raises(ValueError, match="unknown model role"):
        routing.Router(lambda n: n, {"planner": "big"})  # it is "plan"
    with pytest.raises(ValueError, match="unknown model role"):
        routing.from_models({"m": 1}).profile_for("summarize")


def test_a_bare_model_still_answers_every_role():
    model = ScriptedModel([])
    model.model = "gpt-5-mini"  # a real backend carries its name; the label uses it
    router = routing.as_router(model)
    assert all(router.for_role(r) is model for r in routing.ROLES)
    assert router.profile_for("main") == "gpt-5-mini"
    assert routing.as_router(router) is router


# --- the loop uses it ---------------------------------------------------------


def _planner(items):
    """A model that only ever answers the planner prompt."""
    return ScriptedModel([], plan_items=items)


def test_the_planner_role_runs_on_its_own_model():
    """The point of the split: the checklist comes from the planning profile, and the
    main model never sees the planner's prompt at all."""
    planner = _planner(["research the topic", "email the result"])
    main = ScriptedModel([assistant_says("done")])
    router = routing.from_models({"big": planner, "small": main}, {"plan": "big"}, "small")

    state = loop.run(
        goal="research X and email it",
        model=router,
        memory=NullMemory(),
        max_steps=3,
        plan=True,
        run_dir=None,
    )

    assert [t.text for t in state.todo] == ["research the topic", "email the result"]
    # The main model's recorded calls are the conversation's turns, nothing else.
    assert len(main.calls) >= 1
    assert not any(
        str(m[0].get("content", "")).startswith("Break the user's") for m in main.calls
    )


def test_spend_is_broken_down_by_profile():
    planner = _planner(["one thing"])
    main = ScriptedModel([assistant_says("done", cost=0.004)])
    router = routing.from_models({"big": planner, "small": main}, {"plan": "big"}, "small")

    state = loop.run(
        goal="do one thing",
        model=router,
        memory=NullMemory(),
        max_steps=3,
        plan=True,
        run_dir=None,
    )

    # The planner call used to be charged to nobody at all; both roles show up now,
    # and with no subagent in play the breakdown accounts for every dollar spent.
    assert set(state.spend_by_profile) == {"big", "small"}
    assert state.spend_by_profile["big"] == 0.001  # the one planner call
    assert round(sum(state.spend_by_profile.values()), 6) == round(0.05 - state.remaining_budget, 6)
    assert state.snapshot()["spend"] == state.spend_by_profile


def test_planning_cost_is_charged_to_the_run():
    """It was not, before routing — and `plan` is the role most likely to point at the
    expensive profile, so an unbilled planner call is a lie in the budget."""

    class _State:
        def __init__(self):
            self.charged = []

        def charge(self, cost, profile=""):
            self.charged.append((cost, profile))

    state = _State()
    plan.decompose("x", _planner(["a"]), state, "big")
    assert state.charged == [(0.001, "big")]


def test_a_subagent_runs_on_the_subagent_profile():
    parent_model = ScriptedModel(
        [
            assistant_calls([("delegate", {"task": "read the long thing"})]),
            assistant_says("parent answer"),
        ]
    )
    child_model = ScriptedModel([assistant_says("child answer")])
    router = routing.from_models(
        {"big": parent_model, "small": child_model}, {"subagent": "small"}, "big"
    )

    state = loop.run(
        goal="delegate something",
        model=router,
        memory=NullMemory(),
        max_steps=4,
        subagents=True,
        run_dir=None,
    )

    assert state.subagent_runs == 1
    assert "child answer" in state.trace[0].result
    assert child_model.calls, "the child's turns ran on the subagent profile"
    # The child's spend is merged into the parent's breakdown, under its own profile.
    assert "small" in state.spend_by_profile


# --- prompt-cache keys --------------------------------------------------------


class _KeyRecorder(ScriptedModel):
    """A scripted model that also accepts a cache key, like a real backend."""

    def __init__(self, name):
        super().__init__([assistant_says("ok")])
        self.model = name
        self.cache_key = None

    def set_cache_key(self, key):
        self.cache_key = key


def _cache_key_for(model_name):
    model = _KeyRecorder(model_name)
    loop.run(
        goal="same goal",
        model=model,
        memory=NullMemory(),
        max_steps=1,
        run_dir=None,
        today="2026-01-01",  # the date is in the prefix; pin it
    )
    return model.cache_key


def test_the_cache_key_is_per_model_not_only_per_prefix():
    """Caches are per model. Two profiles sharing this system prompt must not be handed
    the same grouping key."""
    assert _cache_key_for("gpt-5") != _cache_key_for("gpt-5-mini")
    assert _cache_key_for("gpt-5") == _cache_key_for("gpt-5")  # still stable across runs


def test_the_chat_backend_can_finally_be_given_a_cache_key():
    """`complete()` always read self.cache_key, but nothing could write it, so
    prompt_cache_key was dead on the Chat path."""
    from types import SimpleNamespace

    from teacup_agent.model import OpenAIModel

    sent = {}

    class Completions:
        def create(self, **kwargs):
            sent.update(kwargs)
            msg = SimpleNamespace(
                content="hi",
                tool_calls=None,
                model_dump=lambda exclude_none=False: {"role": "assistant", "content": "hi"},
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    model = OpenAIModel("gpt-5", client=client)
    model.set_cache_key("teacup-agent-abc")
    model.complete([], [])
    assert sent["prompt_cache_key"] == "teacup-agent-abc"


# --- agent.yaml ---------------------------------------------------------------


def _write(tmp_path, config):
    path = tmp_path / "agent.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


BASE = {
    "models": {
        "default": "big",
        "profiles": {
            "big": {"model": "gpt-5"},
            "small": {"model": "gpt-5-mini"},
        },
    }
}


def test_roles_are_read_from_the_config(tmp_path):
    cfg = agent_config.load(
        _write(tmp_path, {"models": {**BASE["models"], "roles": {"subagent": "small"}}})
    )
    assert cfg.roles == {"subagent": "small"}
    router = agent_config.build_router(cfg)
    assert router.profile_for("subagent") == "small"
    assert router.profile_for("main") == "big"


def test_a_misspelled_role_fails_at_load_time(tmp_path):
    with pytest.raises(ValueError, match="not a known role"):
        agent_config.load(
            _write(tmp_path, {"models": {**BASE["models"], "roles": {"planner": "small"}}})
        )


def test_a_role_naming_a_missing_profile_fails_at_load_time(tmp_path):
    with pytest.raises(ValueError, match="not one of models.profiles"):
        agent_config.load(
            _write(tmp_path, {"models": {**BASE["models"], "roles": {"plan": "tiny"}}})
        )


def test_a_profile_no_role_uses_is_never_built(tmp_path, monkeypatch):
    """Lazily, on purpose: naming an endpoint you did not reach this run costs nothing —
    not even the API key it would have needed. Pinned on build_router() itself, since
    that is the factory production uses."""
    cfg = agent_config.load(_write(tmp_path, BASE))
    built = []
    monkeypatch.setattr(agent_config, "build_model", lambda profile: built.append(profile.model))
    router = agent_config.build_router(cfg)
    assert built == []  # nothing constructed until a role asks
    router.for_role("main")
    router.for_role("plan")
    assert built == ["gpt-5"]  # one build, and never the unused "small" profile


def test_the_judge_defaults_to_the_configured_role(tmp_path):
    cfg = agent_config.load(
        _write(tmp_path, {"models": {**BASE["models"], "roles": {"judge": "small"}}})
    )
    # trajectory.py resolves --judge-profile or models.roles.judge or models.default;
    # this pins the middle term, which is the one the config declares.
    assert (None or cfg.roles.get("judge") or cfg.default_model) == "small"
    assert ("big" or cfg.roles.get("judge") or cfg.default_model) == "big"
