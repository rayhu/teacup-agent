"""teacup-agent-serve, tested against a real a2a-sdk client over httpx.ASGITransport —
real protocol, zero real sockets. Needs the a2a-server extra (Starlette + uvicorn);
skips cleanly rather than failing when it is not installed, since `uv sync` alone (the
default a plain-CLI user runs) does not install it — see pyproject.toml.
"""

from __future__ import annotations

import pytest

starlette = pytest.importorskip("starlette", reason="needs `uv sync --extra a2a-server`")

import httpx

from teacup_agent import agent_config, model as model_mod
from teacup_agent.a2a.server import build_app

MINIMAL = """
models:
  default: main
  profiles:
    main:
      model: gpt-5
      api_key_env: FAKE_KEY
runtime:
  plan: off
  reflect: off
  run_dir: off
a2a:
  card:
    name: served-agent
    description: A test double.
    version: "0.1.0"
"""


def _cfg(tmp_path, monkeypatch, text=MINIMAL):
    monkeypatch.setenv("FAKE_KEY", "sk-test")
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "agent.yaml"
    path.write_text(text, encoding="utf-8")
    return agent_config.load(path)


async def _send(app, task_text: str) -> str:
    from a2a.client import A2ACardResolver, ClientConfig, create_client
    from a2a.helpers import get_stream_response_text, new_text_message
    from a2a.types import Role, SendMessageRequest

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as hc:
        card = await A2ACardResolver(httpx_client=hc, base_url="http://test").get_agent_card()
        client = await create_client(
            agent=card, client_config=ClientConfig(streaming=False, httpx_client=hc)
        )
        request = SendMessageRequest(message=new_text_message(task_text, role=Role.ROLE_USER))
        text = ""
        async for chunk in client.send_message(request):
            t = get_stream_response_text(chunk)
            if t:
                text = t
        await client.close()
        return text


def test_agent_card_reflects_the_configured_identity(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch)
    monkeypatch.setattr(
        agent_config, "build_model", lambda profile: model_mod.ScriptedModel(script=[])
    )
    app = build_app(cfg, url="http://test")

    async def get_card():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as hc:
            from a2a.client import A2ACardResolver

            return await A2ACardResolver(httpx_client=hc, base_url="http://test").get_agent_card()

    import asyncio

    card = asyncio.run(get_card())
    assert card.name == "served-agent"
    assert card.description == "A test double."


def test_a_served_task_gets_a_real_answer_back(tmp_path, monkeypatch):
    import asyncio

    cfg = _cfg(tmp_path, monkeypatch)
    scripted = model_mod.ScriptedModel(script=[model_mod.assistant_says("42")])
    monkeypatch.setattr(agent_config, "build_model", lambda profile: scripted)
    app = build_app(cfg, url="http://test")

    answer = asyncio.run(_send(app, "what is the answer?"))
    assert answer == "42"


def test_a_run_with_no_answer_fails_the_task(tmp_path, monkeypatch):
    """Defensive: loop.run() should always produce an answer (AGENTS.md rule 3), but
    the server must not silently report success on an empty one either."""
    import asyncio

    from teacup_agent.state import AgentState

    cfg = _cfg(tmp_path, monkeypatch)
    monkeypatch.setattr(agent_config, "build_model", lambda profile: model_mod.ScriptedModel([]))

    def fake_run(**kwargs):  # loop.run() itself is sync — asyncio.to_thread() calls it directly
        return AgentState(goal=kwargs["goal"], answer="", status="error")

    monkeypatch.setattr("teacup_agent.a2a.server.loop.run", fake_run)
    app = build_app(cfg, url="http://test")

    from a2a.types import TaskState

    async def send_and_get_state():
        from a2a.client import A2ACardResolver, ClientConfig, create_client
        from a2a.helpers import new_text_message
        from a2a.types import Role, SendMessageRequest

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as hc:
            card = await A2ACardResolver(
                httpx_client=hc, base_url="http://test"
            ).get_agent_card()
            client = await create_client(
                agent=card, client_config=ClientConfig(streaming=False, httpx_client=hc)
            )
            request = SendMessageRequest(
                message=new_text_message("do something", role=Role.ROLE_USER)
            )
            states = []
            async for chunk in client.send_message(request):
                if chunk.WhichOneof("payload") == "task":
                    states.append(chunk.task.status.state)
            await client.close()
            return states

    states = asyncio.run(send_and_get_state())
    assert TaskState.TASK_STATE_FAILED in states


def test_cancel_is_honestly_unsupported(tmp_path, monkeypatch):
    from teacup_agent.a2a.server import TeacupAgentExecutor

    cfg = _cfg(tmp_path, monkeypatch)
    monkeypatch.setattr(agent_config, "build_model", lambda profile: model_mod.ScriptedModel([]))
    executor = TeacupAgentExecutor(cfg)

    import asyncio

    with pytest.raises(NotImplementedError):
        asyncio.run(executor.cancel(None, None))
