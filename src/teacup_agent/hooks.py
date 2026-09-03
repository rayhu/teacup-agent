"""Hooks — project-local callbacks at two named points in the loop.

Every guardrail so far has been hardcoded in `loop.py`: the per-turn cap, the
approval gate, the forced wrap-up. Those are the loop's own rules, and they should
stay in the loop. A user's own rule — "never email outside this domain", "redact
anything that looks like a key" — should not require editing `loop.py` to add.

`hooks.py` in the project (opt-in by file, the same convention `mcp.json` and
`skills/` already use) may define either or both of:

    def before_tool_call(name: str, arguments: dict) -> str | None:
        ...  # return a reason to block the call; None means allowed

    def after_tool_result(name: str, arguments: dict, result: str) -> str | None:
        ...  # return a replacement result; None means unchanged

This module is the *framework's* loader, distinct from the project file it loads:
it turns those two optional functions into `veto()`/`rewrite()` calls the loop can
make unconditionally, whether or not a project defines either one.
"""

from __future__ import annotations

import importlib.util
import pathlib
from typing import Any, Callable

_before_tool_call: Callable[[str, dict[str, Any]], str | None] | None = None
_after_tool_result: Callable[[str, dict[str, Any], str], str | None] | None = None


def load(path: str | pathlib.Path) -> None:
    """Import the project's hooks.py and bind whichever functions it defines.

    Loaded under an internal module name (not the project's own "hooks") so it
    never collides with — or shadows — this module in sys.modules.
    """
    global _before_tool_call, _after_tool_result
    p = pathlib.Path(path)
    spec = importlib.util.spec_from_file_location("_teacup_agent_project_hooks", p)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load hooks from {p}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _before_tool_call = getattr(module, "before_tool_call", None)
    _after_tool_result = getattr(module, "after_tool_result", None)


def disable() -> None:
    global _before_tool_call, _after_tool_result
    _before_tool_call = None
    _after_tool_result = None


def veto(name: str, arguments: dict[str, Any]) -> str | None:
    """None means allowed. A returned string becomes the tool result, always
    prefixed ERROR: so the model reads it exactly like any other failed call —
    a hook author should not have to remember that convention themselves."""
    if _before_tool_call is None:
        return None
    reason = _before_tool_call(name, arguments)
    if reason is None:
        return None
    return reason if reason.startswith("ERROR:") else f"ERROR: {reason}"


def rewrite(name: str, arguments: dict[str, Any], result: str) -> str:
    """None from the hook means unchanged; anything else replaces the result."""
    if _after_tool_result is None:
        return result
    rewritten = _after_tool_result(name, arguments, result)
    return result if rewritten is None else rewritten
