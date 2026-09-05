# This repo's own hooks.py — not the example (see hooks.example.py for the
# annotated walkthrough of the mechanism). This file is what actually approves
# unattended tool calls under `--approve hooks`; read it before trusting it,
# and see docs/threat-model.md before changing what it approves.
#
# Two decisions this file makes, both explicit:
#
# 1. edit_file/write_file are approved unconditionally. Both already carry
#    their own structural guards (tools.py's deny-list and traversal guard,
#    edit_file's read-fresh/single-unambiguous-match requirement, write_file's
#    new-files-only restriction) — this file adds no further restriction on
#    *which* files, the same way a human reviewer trusts those guards rather
#    than re-deriving them. What actually keeps this safe is downstream, not
#    here: every coding task runs on its own disposable git worktree + branch
#    (teacup-run's coding_task.py), and nothing is pushed or merged without a
#    human reading the diff first.
# 2. run_command stays on a narrow, argument-aware allowlist — shell=True
#    means an approved verb could still smuggle a second command via chaining
#    or substitution, so UNSAFE_SHELL_METACHARACTERS (mirrored from
#    hooks.example.py, where an independent review found and fixed the
#    original gap) refuses those outright before the allowlist is even
#    consulted. `git push` is refused unconditionally, however it is asked.

import json
import re

ALLOWED_COMMAND_PATTERNS = (
    re.compile(r"^git (status|diff|add|commit|log)\b"),
    re.compile(r"^uv run (pytest|python -m teacup_agent\.evals)\b"),
)
DENIED_COMMAND_PATTERNS = (re.compile(r"^git push\b"),)
UNSAFE_SHELL_METACHARACTERS = re.compile(r"[;&|`$<>\n]")


def _command_string(call) -> str | None:
    try:
        return json.loads(call.arguments).get("command", "").strip()
    except (json.JSONDecodeError, AttributeError):
        return None


def before_tool_call(call):
    if call.name != "run_command":
        return None
    command = _command_string(call)
    if command is None:
        return None
    if UNSAFE_SHELL_METACHARACTERS.search(command):
        return (
            f"ERROR: this project's hooks.py only allows a single, literal "
            f"command — {command!r} contains a shell metacharacter (chaining, "
            "substitution, or redirection), which could smuggle a second, "
            "unvetted command past this allowlist. This is a fixed project "
            "rule, not a permission that can be granted."
        )
    if any(p.match(command) for p in DENIED_COMMAND_PATTERNS):
        return (
            f"ERROR: this project's hooks.py never allows {command!r}, however "
            "it is asked. This is a fixed project rule, not a permission that "
            "can be granted."
        )
    return None


def approve_tool_call(call, spec):
    if call.name in ("edit_file", "write_file"):
        return True

    if call.name == "run_command":
        command = _command_string(call)
        if command is None:
            return None
        if UNSAFE_SHELL_METACHARACTERS.search(command):
            return None  # before_tool_call already vetoes this — never approve it here too
        if any(p.match(command) for p in ALLOWED_COMMAND_PATTERNS):
            return True
        return None

    return None  # no opinion on anything else — falls back to the run's own policy


def after_tool_result(call, result):
    return result
