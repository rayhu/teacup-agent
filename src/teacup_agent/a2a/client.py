"""A2A client — hand a task to a different agent process.

`subagent.py`'s `delegate` hands a subtask to a child **in-process** loop; this is the
same idea across a process (or machine) boundary, to an agent that may not even be
teacup-agent. Peers are configured in `agent.yaml`'s `a2a.peers` block (#15's schema,
given real shape by `agent_config.A2APeer`): a name, a URL, and an optional
`api_key_env` naming the bearer token's environment variable.

Why this lives in `cli.py` next to `McpHub`, not inside `loop.run()` next to
`subagent.py`/`skills.py`: a peer connection is an external resource (an `httpx` client,
a resolved Agent Card) configured once from `agent.yaml`, not per-turn — the same reason
`McpHub` is constructed once per process invocation rather than per `loop.run()` call.
Unlike MCP (one server -> N distinct tools, each with the server's own schema), every
peer shares one tool shape (`peer`, `task`), so there is no per-peer schema to fetch
upfront: connecting is lazy, and only each peer's `api_key_env` is resolved eagerly, at
`register()` time, so a missing token fails at startup rather than silently mid-run —
the same discipline `agent_config.build_model()` already holds for model profiles.

Why a background event loop + thread: `a2a-sdk`'s client is async (`httpx`-based), and
`loop.py`'s tool-execution thread pool calls tool functions synchronously — the exact
constraint `mcp_tools.py` already solved the same way. `_run()` below is the identical
bridge: `asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)`.
"""

from __future__ import annotations

import asyncio
import os
import threading
from typing import Any

import httpx

from teacup_agent import tools as tools_mod
from teacup_agent.agent_config import A2APeer

NAME = "delegate_a2a"
CALL_TIMEOUT = 120.0  # a remote agent's own run can take longer than a typical tool call

DESCRIPTION = (
    "Hand a task to a different agent - possibly on another machine, possibly not "
    "teacup-agent at all - configured as a peer in agent.yaml. Use this when the task "
    "needs a capability this agent does not have, but a known peer does."
)
PARAMETERS = {
    "type": "object",
    "properties": {
        "peer": {"type": "string", "description": "name of a configured peer"},
        "task": {"type": "string", "description": "what to ask the peer to do"},
    },
    "required": ["peer", "task"],
}

# States a peer's task can end in that are not success. TASK_STATE_INPUT_REQUIRED and
# TASK_STATE_AUTH_REQUIRED are the multi-round-trip pattern (the peer wants interactive
# input mid-task); this client does not do elicitation any more than mcp_tools.py does,
# so both become a self-correctable ERROR rather than hanging or crashing.
_FAILURE_STATE_NAMES = frozenset(
    {
        "TASK_STATE_FAILED",
        "TASK_STATE_REJECTED",
        "TASK_STATE_CANCELED",
        "TASK_STATE_INPUT_REQUIRED",
        "TASK_STATE_AUTH_REQUIRED",
    }
)


class A2AHub:
    """Owns the event loop, the resolved peer configs, and their live clients."""

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        # `transport` is a test-only escape hatch (httpx.ASGITransport, so tests can
        # drive a real in-process a2a-sdk server with zero real sockets); agent.yaml
        # has no way to set it.
        self._transport = transport
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._peers: dict[str, tuple[str, str | None]] = {}  # name -> (url, token)
        self._clients: dict[str, Any] = {}  # name -> a2a Client, built lazily
        self._httpx_clients: dict[str, httpx.AsyncClient] = {}

    def _run(self, coro: Any, timeout: float | None = None) -> Any:
        """Block the calling (sync) thread on a coroutine in the background loop."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    # -- connecting -----------------------------------------------------------

    def register(self, peers: dict[str, A2APeer]) -> None:
        """Resolve every peer's token now (fail fast); register the one shared tool."""
        resolved: dict[str, tuple[str, str | None]] = {}
        for name, peer in peers.items():
            token = None
            if peer.api_key_env:
                token = os.getenv(peer.api_key_env)
                if not token:
                    raise RuntimeError(
                        f"a2a peer {name!r} references api_key_env "
                        f"{peer.api_key_env!r}, but that variable is not set (check "
                        ".env)"
                    )
            resolved[name] = (peer.url, token)
        self._peers = resolved
        tools_mod.REGISTRY[NAME] = tools_mod.Tool(
            name=NAME,
            description=DESCRIPTION,
            parameters=PARAMETERS,
            fn=self._delegate,
            requires_approval=True,  # an outbound call to a third-party, possibly-billed
            timeout=CALL_TIMEOUT,    # agent is exactly AGENTS.md rule 4's case
        )

    async def _client_for(self, peer: str) -> Any:
        if peer in self._clients:
            return self._clients[peer]
        url, token = self._peers[peer]
        from a2a.client import A2ACardResolver, ClientConfig, create_client

        headers = {"Authorization": f"Bearer {token}"} if token else {}
        hc = httpx.AsyncClient(transport=self._transport, base_url=url, headers=headers)
        self._httpx_clients[peer] = hc
        # Resolve the card explicitly rather than passing the bare URL to create_client:
        # create_client(agent=<url>) does its own internal resolution that does not
        # reliably go through the httpx_client passed in ClientConfig (verified against
        # the installed a2a-sdk — resolving explicitly is also what lets the test suite
        # drive this through httpx.ASGITransport with zero real sockets).
        card = await A2ACardResolver(httpx_client=hc, base_url=url).get_agent_card()
        client = await create_client(
            agent=card, client_config=ClientConfig(streaming=False, httpx_client=hc)
        )
        self._clients[peer] = client
        return client

    async def _send(self, peer: str, task: str) -> str:
        from a2a.helpers import get_stream_response_text, new_text_message
        from a2a.types import Role, SendMessageRequest, TaskState

        client = await self._client_for(peer)
        request = SendMessageRequest(message=new_text_message(task, role=Role.ROLE_USER))
        last_text = ""
        async for chunk in client.send_message(request):
            text = get_stream_response_text(chunk)
            if text:
                last_text = text
            variant = chunk.WhichOneof("payload")
            state = None
            if variant == "task":
                state = chunk.task.status.state
            elif variant == "status_update":
                state = chunk.status_update.status.state
            if state is not None and TaskState.Name(state) in _FAILURE_STATE_NAMES:
                detail = f": {last_text}" if last_text else ""
                return f"ERROR: peer {peer!r} task ended in {TaskState.Name(state)}{detail}"
        return last_text or "(the peer returned no content)"

    # -- calling ----------------------------------------------------------------

    def _delegate(self, peer: str, task: str) -> str:
        if peer not in self._peers:
            return (
                f"ERROR: unknown peer {peer!r}. Configured peers: "
                f"{sorted(self._peers)}"
            )
        try:
            return self._run(self._send(peer, task), timeout=CALL_TIMEOUT)
        except Exception as e:  # connection errors, timeouts, anything the SDK raises
            return f"ERROR: delegate_a2a to {peer!r} failed: {type(e).__name__}: {e}"

    # -- teardown -----------------------------------------------------------

    def close(self) -> None:
        tools_mod.REGISTRY.pop(NAME, None)
        for client in self._clients.values():
            try:
                self._run(client.close(), timeout=10)
            except Exception:
                pass  # a peer that is already gone must not fail the run
        for hc in self._httpx_clients.values():
            try:
                self._run(hc.aclose(), timeout=10)
            except Exception:
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
