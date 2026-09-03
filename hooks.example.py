# Copy this to hooks.py to try it: uv run teacup-agent --hooks hooks.py "..."
# (or --config agent.yaml with `hooks: path: hooks.py`).
#
# hooks.py is project-specific policy, executed on every tool call — gitignored, same
# as mcp.json and agent.yaml. Define either function below, or both; neither is
# required, and a run with no hooks.py behaves exactly as it does today.

ALLOWED_DOMAINS = {"example.com", "internal.example.org"}


def before_tool_call(name: str, arguments: dict) -> str | None:
    """Return a reason to block the call; None means allowed.

    This is the "block by argument, not just by tool name" case the roadmap calls
    out: send_email itself stays available, but only to an allowlisted domain.
    """
    if name == "send_email":
        recipient = arguments.get("to", "")
        domain = recipient.rpartition("@")[2].lower()
        if domain not in ALLOWED_DOMAINS:
            return f"recipient domain {domain!r} is not on the allowlist ({', '.join(sorted(ALLOWED_DOMAINS))})"
    return None


def after_tool_result(name: str, arguments: dict, result: str) -> str | None:
    """Return a replacement result; None means unchanged.

    A crude but real example: redact anything that looks like a bare API key before
    it ever reaches the model's context.
    """
    import re

    redacted = re.sub(r"\bsk-[A-Za-z0-9]{20,}\b", "[REDACTED]", result)
    return redacted if redacted != result else None
