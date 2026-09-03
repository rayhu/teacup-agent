# Copy this to hooks.py to try it: uv run teacup-agent --hooks hooks.py --approve hooks "..."
#
# hooks.py is opt-in by file, same convention as mcp.json and agent.yaml: its mere
# presence is what --hooks discovers by default, and nothing in this file runs unless
# something loads it. See docs/threat-model.md before using --approve hooks with a
# hooks.py you did not write yourself.
#
# This one demonstrates the exact example roadmap #14 asked for: "send_email only to
# these domains" — using the built-in send_email tool, no new tools required. Adapt the
# allowlist and the tool name for whatever this repo's own coding tools need once they
# exist (roadmap: write_file/run_command, confined to specific paths/commands).

import json

ALLOWED_EMAIL_DOMAINS = ("example.com",)


def _email_recipient_domain(call) -> str | None:
    try:
        to = json.loads(call.arguments).get("to", "")
    except (json.JSONDecodeError, AttributeError):
        return None
    return to.rsplit("@", 1)[-1].lower() if "@" in to else None


def before_tool_call(call):
    """Return a string to veto the call; None to let it through to the approval gate."""
    if call.name != "send_email":
        return None
    domain = _email_recipient_domain(call)
    if domain not in ALLOWED_EMAIL_DOMAINS:
        return (
            f"ERROR: this project's hooks.py only allows send_email to "
            f"{', '.join(ALLOWED_EMAIL_DOMAINS)}. This is a fixed project rule, not a "
            "permission that can be granted by asking differently."
        )
    return None


def approve_tool_call(call, spec):
    """Return True/False to decide approval with nobody watching; None for no opinion.

    Only reached for calls before_tool_call did not veto — so by the time this runs,
    an out-of-allowlist send_email is already gone. Approving here is what lets an
    unattended run (teacup-run's sandboxed subprocess, say) actually send the mail
    instead of it being denied for lack of a TTY.
    """
    if call.name == "send_email" and _email_recipient_domain(call) in ALLOWED_EMAIL_DOMAINS:
        return True
    return None  # no opinion on anything else — falls back to the run's own policy


def after_tool_result(call, result):
    """Return the (possibly rewritten) result. This example does not need to rewrite
    anything, so it returns the result unchanged — included to show the shape."""
    return result
