# Copy this to hooks.py to try it: uv run teacup-agent --hooks hooks.py --approve hooks
#   --coding-tools "..."
#
# hooks.py is opt-in by file, same convention as mcp.json and agent.yaml: its mere
# presence is what --hooks discovers by default, and nothing in this file runs unless
# something loads it. See docs/threat-model.md before using --approve hooks with a
# hooks.py you did not write yourself.
#
# Two examples in one file, both variations on roadmap #14's own suggested pattern
# ("send_email only to these domains"):
#
# 1. send_email, using the built-in tool, no new tools required.
# 2. run_command (coding_tools.py, --coding-tools): a fixed allowlist of prefixes —
#    read-only git inspection and the project's own test/lint commands are approved
#    automatically; `git push` is refused outright, however it is asked; anything
#    else falls through to the run's normal approval policy (a human, or deny
#    without a TTY) rather than being silently allowed or silently blocked.

import json
import re

ALLOWED_EMAIL_DOMAINS = ("example.com",)

# Order matters only in that denied is checked first (see before_tool_call) — a
# command matching both an allow and a deny pattern is refused, never approved.
ALLOWED_COMMAND_PATTERNS = (
    re.compile(r"^git (status|diff|add|commit|log)\b"),
    re.compile(r"^uv run (pytest|ruff)\b"),
)
DENIED_COMMAND_PATTERNS = (re.compile(r"^git push\b"),)


def _email_recipient_domain(call) -> str | None:
    try:
        to = json.loads(call.arguments).get("to", "")
    except (json.JSONDecodeError, AttributeError):
        return None
    return to.rsplit("@", 1)[-1].lower() if "@" in to else None


def _command_string(call) -> str | None:
    try:
        return json.loads(call.arguments).get("command", "").strip()
    except (json.JSONDecodeError, AttributeError):
        return None


def before_tool_call(call):
    """Return a string to veto the call; None to let it through to the approval gate."""
    if call.name == "send_email":
        domain = _email_recipient_domain(call)
        if domain not in ALLOWED_EMAIL_DOMAINS:
            return (
                f"ERROR: this project's hooks.py only allows send_email to "
                f"{', '.join(ALLOWED_EMAIL_DOMAINS)}. This is a fixed project rule, "
                "not a permission that can be granted by asking differently."
            )
        return None

    if call.name == "run_command":
        command = _command_string(call)
        if command is not None and any(p.match(command) for p in DENIED_COMMAND_PATTERNS):
            return (
                f"ERROR: this project's hooks.py never allows {command!r}, however "
                "it is asked. This is a fixed project rule, not a permission that "
                "can be granted."
            )
        return None

    return None


def approve_tool_call(call, spec):
    """Return True/False to decide approval with nobody watching; None for no opinion.

    Only reached for calls before_tool_call did not veto — so a denied run_command
    prefix is already gone by the time this runs. Approving here is what lets an
    unattended run (teacup-run's sandboxed subprocess, say) actually run the
    approved git/test commands instead of every one being denied for lack of a TTY.
    """
    if call.name == "send_email":
        if _email_recipient_domain(call) in ALLOWED_EMAIL_DOMAINS:
            return True
        return None

    if call.name == "run_command":
        command = _command_string(call)
        if command is not None and any(p.match(command) for p in ALLOWED_COMMAND_PATTERNS):
            return True
        return None

    return None  # no opinion on anything else — falls back to the run's own policy


def after_tool_result(call, result):
    """Return the (possibly rewritten) result. This example does not need to rewrite
    anything, so it returns the result unchanged — included to show the shape."""
    return result
