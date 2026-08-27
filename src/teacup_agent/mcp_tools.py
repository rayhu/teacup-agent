"""MCP — borrow other people's tools instead of writing them.

Every tool in tools.py had to be written by hand. MCP is the standard way to stop
doing that: connect to a server and its tools appear in the registry, ready for the
model to call. Filesystem, GitHub, browsers, databases, search — someone already
wrote them.

The integration is small because of what the 2026-07-28 revision of the spec removed:
there is no `initialize` handshake and no session id any more, so a tool call is one
stateless RPC. Two years of connection lifecycle management simply is not there.

Four details do the real work here:

1. **Names are namespaced.** Two servers may each expose `search`; the spec says
   clients aggregating servers must disambiguate. Tools land as `server__tool`,
   also sanitized because OpenAI function names allow only [A-Za-z0-9_-].
2. **Errors keep our discipline.** MCP separates protocol errors from *tool execution
   errors* (`isError: true`), and the spec says clients SHOULD hand the latter to the
   model so it can self-correct. That is precisely what tools.execute() already does,
   so both map onto an "ERROR: ..." string.
3. **Approval is derived, and defaults to gated.** `annotations.read_only_hint` opens
   a tool; anything else needs approval. The spec warns that annotations are untrusted
   unless the server is, so `"approve": "none"` in the config is the explicit way to
   say "I trust this server" rather than something we infer.
4. **Async lives in one place.** The SDK is async and our tools are sync functions
   called from a thread pool, so a single background event loop owns every session
   and the tool functions block on it. Nothing else in the codebase learns about
   asyncio.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from contextlib import AsyncExitStack
from typing import Any

from teacup_agent import tools as tools_mod

CALL_TIMEOUT = 60.0  # seconds a single MCP tool call may take
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]")


def _tool_name(server: str, tool: str) -> str:
    return _SAFE_NAME.sub("_", f"{server}__{tool}")


def _needs_approval(tool: Any, policy: str) -> bool:
    """Whether a tool from an MCP server has to pass the human gate.

    Default ("auto"): open only what the server explicitly marks read-only. A server
    that annotates nothing gets everything gated — the right incentive, and the right
    default for a gate that exists because nobody is watching.
    """
    if policy == "all":
        return True
    if policy == "none":  # an explicit statement of trust in this server
        return False
    annotations = getattr(tool, "annotations", None)
    return not bool(getattr(annotations, "read_only_hint", False))


def _result_text(result: Any) -> str:
    """Flatten a CallToolResult into the string our loop feeds back to the model."""
    if getattr(result, "result_type", "complete") == "input_required":
        # Multi round-trip requests: the server wants interactive input mid-call. We
        # do not implement elicitation, so say so as a tool error the model can route
        # around, rather than hanging or crashing.
        return (
            "ERROR: this tool asked for interactive input, which this agent does not "
            "support. Try a different tool, or supply the missing information as an "
            "argument."
        )

    parts = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
        elif getattr(block, "uri", None):  # resource_link
            parts.append(f"[resource] {block.uri}")
        else:
            parts.append(f"[{getattr(block, 'type', 'content')} omitted]")

    structured = getattr(result, "structured_content", None)
    if not parts and structured is not None:
        parts.append(json.dumps(structured, ensure_ascii=False))

    body = "\n".join(parts) or "(the tool returned no content)"
    return f"ERROR: {body}" if getattr(result, "is_error", False) else body


class McpHub:
    """Owns the event loop, the connections, and the tools they contributed."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._stack = AsyncExitStack()
        self._clients: dict[str, Any] = {}
        self._devnull: Any = None
        self.registered: list[str] = []

    def _run(self, coro, timeout: float | None = None):
        """Block the calling (sync) thread on a coroutine in the background loop."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    # -- connecting ---------------------------------------------------------

    def connect(self, name: str, spec: dict[str, Any]) -> list[str]:
        """Connect to one server and register its tools. Returns the names added.

        spec: {"url": ...} or {"command": ..., "args": [...], "env": {...}},
        plus optional "tools" (allowlist), "approve" ("auto" | "all" | "none") and
        "stderr" ("hide" | "show").
        """
        from mcp import Client, StdioServerParameters
        from mcp.client.stdio import stdio_client

        if url := spec.get("url"):
            target: Any = url
        else:
            params = StdioServerParameters(
                command=spec["command"],
                args=spec.get("args", []),
                env=spec.get("env") or None,
            )
            # A stdio server logs to its own stderr, which lands in the user's
            # terminal. Servers still on the pre-2026-07-28 protocol print a wall of
            # validation errors when our client probes with server/discover, so the
            # default is to hide it — with "stderr": "show" for when a server will
            # not start and you need to see why.
            if spec.get("stderr", "hide") == "show":
                target = params
            else:
                self._devnull = open(os.devnull, "w")
                target = stdio_client(params, errlog=self._devnull)

        try:
            client = self._run(self._stack.enter_async_context(Client(target)), timeout=60)
        except Exception as e:
            raise RuntimeError(
                f"could not connect to MCP server {name!r}: {type(e).__name__}: {e}. "
                'Set "stderr": "show" in the config to see the server\'s own output.'
            ) from e
        self._clients[name] = client

        listing = self._run(client.list_tools(), timeout=30)
        allow = set(spec.get("tools") or [])
        policy = spec.get("approve", "auto")

        added = []
        for tool in listing.tools:
            if allow and tool.name not in allow:
                continue  # every tool schema costs prefix tokens on every request
            added.append(self._register(name, tool, policy))
        self.registered.extend(added)
        return added

    def _register(self, server: str, tool: Any, policy: str) -> str:
        name = _tool_name(server, tool.name)
        remote = tool.name

        def call(**kwargs: Any) -> str:
            client = self._clients[server]
            result = self._run(client.call_tool(remote, kwargs), timeout=CALL_TIMEOUT)
            return _result_text(result)

        call.__name__ = name
        schema = tool.input_schema or {"type": "object"}
        description = (tool.description or tool.title or remote).strip()
        tools_mod.REGISTRY[name] = tools_mod.Tool(
            name=name,
            description=f"[{server}] {description}",
            parameters=schema,
            fn=call,
            requires_approval=_needs_approval(tool, policy),
        )
        return name

    # -- teardown -----------------------------------------------------------

    def close(self) -> None:
        for name in self.registered:
            tools_mod.REGISTRY.pop(name, None)
        self.registered.clear()
        try:
            self._run(self._stack.aclose(), timeout=15)
        except Exception:
            pass  # a server that died during shutdown must not fail the run
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        if self._devnull is not None:
            self._devnull.close()


def load_config(path: str) -> dict[str, dict[str, Any]]:
    """Read a config file shaped like the one every MCP host uses:

    {"servers": {"fetch": {"command": "uvx", "args": ["mcp-server-fetch"]}}}
    """
    data = json.loads(open(path, encoding="utf-8").read())
    servers = data.get("servers", data)
    # Config files get comments; JSON has none. Keys starting with _ are dropped so
    # "_comment" can be used the way everyone uses it anyway.
    return {
        name: {k: v for k, v in spec.items() if not k.startswith("_")}
        for name, spec in servers.items()
    }
