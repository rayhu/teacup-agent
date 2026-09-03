"""teacup-agent-serve — expose this agent to other agents over Agent2Agent (A2A).

This is the one explicit gesture that turns "a CLI you run once" into a long-lived
process: a separate console script, behind a separate install extra
(`uv sync --extra a2a-server`, since it needs Starlette + uvicorn), that `uv run
teacup-agent` never loads and never needs. Requires `agent.yaml` — there is no
bare-flag mode, matching #15's own framing that Agent2Agent config lives in the YAML.

Reuses `cli._make_approver` unmodified rather than writing a new approval policy: its
existing "auto" branch already denies when there is no TTY, which is always true under
`uvicorn`, so a served agent is gated by AGENTS.md rule 4 with no new code.

**Concurrency limitation, stated rather than silently shipped**: `tools_mod.REGISTRY`
is process-global, and `skills.enable()`/`subagent.enable()` mutate it with no locking,
assuming one `loop.run()` at a time. Two concurrent inbound tasks on a server whose
`agent.yaml` turns on skills or subagents would race on that global state. `_lock` below
serializes `execute()` calls to avoid that race — correct, at the cost of one task at a
time. Real per-run tool isolation is a larger change, out of scope here.
"""

from __future__ import annotations

import argparse
import asyncio

from a2a.helpers import new_task_from_user_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
import a2a.server.routes as routes
from starlette.applications import Starlette

from teacup_agent import agent_config, loop
from teacup_agent.a2a.card import build_agent_card
from teacup_agent.cli import _make_approver, _resolve_plan
from teacup_agent.memory import Memory
from teacup_agent.skills import discover as discover_skills


class TeacupAgentExecutor(AgentExecutor):
    """Bridges the A2A protocol to loop.run(). One instance per server process: the
    model and long-term memory are built once and shared across every served task."""

    def __init__(self, cfg: agent_config.AgentConfig) -> None:
        self._cfg = cfg
        self._model = agent_config.build_model(cfg.models[cfg.default_model])
        self._memory = Memory(cfg.runtime.memory)
        self._lock = asyncio.Lock()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task
        if task is None:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()

        goal = context.get_user_input()
        cfg = self._cfg
        # loop.run() is synchronous; this handler is async, so it runs in a thread
        # rather than blocking the server's event loop for the run's whole duration.
        async with self._lock:
            state = await asyncio.to_thread(
                loop.run,
                goal=goal,
                model=self._model,
                memory=self._memory,
                max_steps=cfg.runtime.max_steps,
                budget=cfg.runtime.budget,
                time_budget=cfg.runtime.deadline if cfg.runtime.deadline > 0 else None,
                tool_timeout=cfg.runtime.tool_timeout,
                context_limit=cfg.runtime.context_limit,
                run_dir=agent_config.resolve_run_dir(cfg.runtime.run_dir),
                approve=_make_approver(cfg.runtime.approve, quiet=True),
                max_tool_calls_per_step=cfg.runtime.max_tool_calls_per_step,
                plan=_resolve_plan(cfg.runtime.plan, live=True),
                reflect=_resolve_plan(cfg.runtime.reflect, live=True),
                skills=cfg.skills_dir,
                subagents=cfg.tools.subagents,
                subagent_max_steps=cfg.tools.subagent_max_steps,
                exclude_tools=cfg.tools.exclude or None,
            )

        if state.answer:
            await updater.add_artifact(parts=[new_text_part(state.answer)])
            await updater.complete()
        else:
            await updater.failed()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # a2a-sdk's own reference examples do the same for "not supported": loop.run()
        # has no cooperative-cancel hook today, so raising here is honest, not faked.
        raise NotImplementedError("cancel is not supported")


def build_app(cfg: agent_config.AgentConfig, url: str) -> Starlette:
    skills = discover_skills(cfg.skills_dir) if cfg.skills_dir else []
    card = build_agent_card(cfg.a2a.card, skills, url)
    handler = DefaultRequestHandler(
        agent_executor=TeacupAgentExecutor(cfg),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    app_routes = routes.create_agent_card_routes(card) + routes.create_jsonrpc_routes(
        handler, "/"
    )
    return Starlette(routes=app_routes)


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv()

    p = argparse.ArgumentParser(
        description="Serve this agent to other agents over the Agent2Agent protocol"
    )
    p.add_argument("--config", default="agent.yaml", help="agent.yaml to load")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9999)
    args = p.parse_args(argv)

    cfg = agent_config.load(args.config)
    url = f"http://{args.host}:{args.port}"
    app = build_app(cfg, url)

    import uvicorn

    print(f"Serving {cfg.a2a.card.get('name', 'teacup-agent')!r} at {url} (Ctrl-C to stop)")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
