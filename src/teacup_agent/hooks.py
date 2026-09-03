"""Hooks — project-declared callbacks at fixed points in the tool-call lifecycle.

Roadmap #13. Loaded from a project-local `hooks.py`, the same opt-in-by-file
convention as `mcp.json` and `skills/`: its mere presence is the opt-in, so a
project that never wrote one pays nothing (no import, no callback machinery)
for a capability it does not use.

Three hook points, each optional in the loaded module:

    def before_tool_call(call: ToolCall) -> str | None:
        '''Return a string to veto the call — it becomes the call's ERROR
        result, exactly the shape a denied or throttled call already gets.
        Return None to allow it through.'''

    def after_tool_result(call: ToolCall, result: str) -> str:
        '''Return the (possibly rewritten) result that reaches the model.'''

    def approve_tool_call(call: ToolCall, spec) -> bool | None:
        '''Return True or False to decide approval programmatically. Return
        None for "no opinion" — the run's approval policy decides as if this
        hook did not exist.'''

Why three hooks and not one: `before_tool_call` can only refuse, and
`after_tool_result` can only reshape something that already ran — neither can
turn a gated call into an approved one. `approve_tool_call` is the one hook
that can say **yes** on nobody's behalf, which is exactly why it is kept
separate, and why nothing consults it unless the run's `--approve` policy
explicitly asks (see `cli.py`'s `hooks` policy). Saying yes here is a trust
the *project being operated on* declared for itself by shipping this file —
see `docs/threat-model.md` for why that is not the same thing as weakening
"deny by default when nobody is watching".

Failure handling deliberately differs by hook, because the three are not
symmetric risks:
- `before_tool_call` fails **closed**: an exception becomes a veto. A broken
  safety check must not silently stop being a safety check.
- `approve_tool_call` fails to **no opinion** (None), which already means
  "deny without a TTY" through the normal approval fallback — also closed.
- `after_tool_result` fails to a **no-op** (the original, unrewritten
  result): it is a transform, not a gate, so the safe fallback is silence,
  the same "a broken planner must never stop the run" discipline `plan.py`
  and `reflect.py` already hold.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import Any, Callable

from teacup_agent.model import ToolCall

__all__ = ["load", "unload", "veto", "rewrite", "approve"]

_MODULE_NAME = "_teacup_agent_project_hooks"

_before_tool_call: Callable[[ToolCall], "str | None"] | None = None
_after_tool_result: Callable[[ToolCall, str], str] | None = None
_approve_tool_call: Callable[[ToolCall, Any], "bool | None"] | None = None


def load(path: str | pathlib.Path) -> bool:
    """Load a project-local hooks.py. Returns True if any hook was defined."""
    global _before_tool_call, _after_tool_result, _approve_tool_call
    p = pathlib.Path(path)
    if not p.is_file():
        return False

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, p)
    if spec is None or spec.loader is None:  # pragma: no cover - unreachable for real files
        return False
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)

    _before_tool_call = getattr(module, "before_tool_call", None)
    _after_tool_result = getattr(module, "after_tool_result", None)
    _approve_tool_call = getattr(module, "approve_tool_call", None)
    return any((_before_tool_call, _after_tool_result, _approve_tool_call))


def unload() -> None:
    """Clear loaded hooks. Mirrors skills.disable()/subagent.disable(): hooks are
    per-run state, not a global that outlives the process that loaded them."""
    global _before_tool_call, _after_tool_result, _approve_tool_call
    _before_tool_call = _after_tool_result = _approve_tool_call = None
    sys.modules.pop(_MODULE_NAME, None)


def veto(call: ToolCall) -> "str | None":
    """None = allowed. A string = the ERROR result the call gets instead of running."""
    if _before_tool_call is None:
        return None
    try:
        return _before_tool_call(call)
    except Exception as e:  # fail closed — see module docstring
        return f"ERROR: hooks.before_tool_call raised {type(e).__name__}: {e}"


def rewrite(call: ToolCall, result: str) -> str:
    """The result that reaches the model, after any project-declared rewrite."""
    if _after_tool_result is None:
        return result
    try:
        return _after_tool_result(call, result)
    except Exception:  # fail to a no-op — see module docstring
        return result


def approve(call: ToolCall, spec: Any) -> "bool | None":
    """True/False = a programmatic approval decision. None = no opinion."""
    if _approve_tool_call is None:
        return None
    try:
        return _approve_tool_call(call, spec)
    except Exception:  # fail to "no opinion" — see module docstring
        return None
