"""A tiny MCP server used by the tests — real protocol, no network, no npx.

It exposes three tools chosen to exercise the parts of the adapter that matter:
one annotated read-only, one annotated destructive, and one with no annotations at
all (the common case in the wild, and the one that must default to gated).
"""

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

server = MCPServer("demo")


@server.tool(
    description="Echo a string back.",
    annotations=ToolAnnotations(read_only_hint=True),
)
def echo(text: str) -> str:
    return f"echo: {text}"


@server.tool(
    description="Pretend to delete something.",
    annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True),
)
def wipe(target: str) -> str:
    return f"wiped {target}"


@server.tool(description="A tool with no annotations at all.")
def unannotated(value: str) -> str:
    return f"got {value}"


@server.tool(description="Always fails, as a tool execution error.")
def explode() -> str:
    raise ValueError("boom")


if __name__ == "__main__":
    server.run("stdio")
