"""A tiny in-process A2A server for tests — real a2a-sdk objects, real protocol, but
built as a Starlette app object rather than run with uvicorn, so tests drive it through
httpx.ASGITransport with zero real sockets (A2A has no stdio transport to subprocess
over the way tests/fixtures/demo_mcp_server.py does for MCP).

Behavior: echoes the task text back as the completed answer, unless the text contains
"fail", in which case the task ends in TASK_STATE_FAILED — one server, both paths.
"""

from __future__ import annotations

from a2a.helpers import new_task_from_user_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
import a2a.server.routes as routes
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from starlette.applications import Starlette


class EchoAgentExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task
        if task is None:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()

        query = context.get_user_input()
        if "fail" in query:
            await updater.failed()
            return
        await updater.add_artifact(parts=[new_text_part(f"echo: {query}")])
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancel is not supported")


def build_app() -> Starlette:
    card = AgentCard(
        name="demo-echo-agent",
        description="Echoes the task text back; fails if the text contains 'fail'.",
        version="0.1.0",
        supported_interfaces=[
            AgentInterface(url="http://test", protocol_binding="JSONRPC")
        ],
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[AgentSkill(id="echo", name="echo", description="Echoes the input.")],
    )
    handler = DefaultRequestHandler(
        agent_executor=EchoAgentExecutor(), task_store=InMemoryTaskStore(), agent_card=card
    )
    app_routes = routes.create_agent_card_routes(card) + routes.create_jsonrpc_routes(
        handler, "/"
    )
    return Starlette(routes=app_routes)
