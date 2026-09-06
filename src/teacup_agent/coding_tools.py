"""Coding tools — the four capabilities that turn this from a research agent into
one that can also change and verify a repository: see what's there, edit or create a
file, and run a command.

Opt-in via `--coding-tools` (`loop.run(coding_tools=True)`), the same "do not cost
prefix tokens when unused, and do not exist as an attack surface when unused"
convention `--subagents`/`--skills`/`--mcp` already follow — and, per `AGENTS.md`,
exactly the precondition roadmap history named for this: "a code-execution tool must
not be added until a sandbox exists to run it in." A sandbox now exists
(teacup-run's sandboxed subprocess launcher), but only when this agent is launched
through it; the approval gate plus roadmap #13's hooks-based approval policy
(`--approve hooks`) are what make `write_file`/`run_command` safe to enable even
outside that sandbox, by keeping "deny by default when nobody is watching" the
default and letting a project opt a specific call into approval explicitly
(`docs/threat-model.md`).

Registered dynamically in `enable()`, the same shape `subagent.py`'s `delegate` tool
uses (`tools_mod.REGISTRY` mutated directly, not the `@tool` decorator every always-on
built-in tool uses) — these four must not exist in the registry at all unless
`--coding-tools` was passed.
"""

from __future__ import annotations

import os
import pathlib
import subprocess

from teacup_agent import tools as tools_mod

LIST_FILES = "list_files"
EDIT_FILE = "edit_file"
WRITE_FILE = "write_file"
RUN_COMMAND = "run_command"
_NAMES = (LIST_FILES, EDIT_FILE, WRITE_FILE, RUN_COMMAND)

_MAX_ECHO_LINES = 24  # cap on the post-edit echo; see _edited_region
_DEFAULT_COMMAND_TIMEOUT = 60.0
_MAX_COMMAND_TIMEOUT = 300.0


def enable() -> None:
    """Register the four coding tools for one run."""
    tools_mod.REGISTRY[LIST_FILES] = tools_mod.Tool(
        name=LIST_FILES,
        description=(
            "List files and directories under a path inside the project (top-level "
            "only unless recursive=true). Credentials, configuration, saved run "
            "states and .git/.venv never appear."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "directory, relative to the project root; default '.'",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "walk subdirectories too; default false",
                },
            },
        },
        fn=_list_files,
        requires_approval=False,
    )
    tools_mod.REGISTRY[EDIT_FILE] = tools_mod.Tool(
        name=EDIT_FILE,
        description=(
            "Replace one exact occurrence of old_string with new_string in an "
            "existing file. old_string may span multiple lines — this works the "
            "same way for a one-line change or a ten-line block, so a multi-line "
            "edit is never a reason to reach for run_command instead. Fails if "
            "old_string is not found, or is not unique — read the file first "
            "(read_file) and copy the exact current text verbatim, including "
            "whitespace and line breaks, rather than reconstructing it from memory; "
            "include enough surrounding context to make the match unambiguous. "
            "To insert a new line next to existing ones (a new field in a class, a "
            "new argument in a call), prefer matching only the single adjacent line "
            "already there — new_string is that same line plus the one being added — "
            "over trying to reproduce the whole surrounding block verbatim; a short, "
            "exact anchor is far more reliable than a long one and is exactly how "
            "this kind of insertion is normally done. If a call fails because "
            "old_string did not match, call read_file again (the file may not be "
            "what you expect), and if a longer match keeps failing, retry with a "
            "shorter one-line anchor instead of giving up — do not switch to "
            "run_command (sed, heredocs, `python -c`) to do the edit instead; that "
            "path is not part of this tool's approved workflow and is likely to be "
            "blocked outright. This has external side effects and cannot be "
            "trivially undone, so it requires human approval before it runs."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "path relative to the project root"},
                "old_string": {
                    "type": "string",
                    "description": "exact text to replace; must appear exactly once in the file",
                },
                "new_string": {"type": "string", "description": "replacement text"},
            },
            "required": ["path", "old_string", "new_string"],
        },
        fn=_edit_file,
        requires_approval=True,
    )
    tools_mod.REGISTRY[WRITE_FILE] = tools_mod.Tool(
        name=WRITE_FILE,
        description=(
            "Create a new file with the given content. Fails if the file already "
            "exists — use edit_file to change an existing one. This has external "
            "side effects and cannot be trivially undone, so it requires human "
            "approval before it runs."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "path relative to the project root"},
                "content": {"type": "string", "description": "full content of the new file"},
            },
            "required": ["path", "content"],
        },
        fn=_write_file,
        requires_approval=True,
    )
    tools_mod.REGISTRY[RUN_COMMAND] = tools_mod.Tool(
        name=RUN_COMMAND,
        description=(
            "Run a shell command inside the project directory and return its exit "
            "code, stdout and stderr. Can do anything a shell can (including run "
            "tests, git, or delete files), so it requires human approval before it "
            f"runs. Times out after {_DEFAULT_COMMAND_TIMEOUT:.0f}s by default "
            f"(max {_MAX_COMMAND_TIMEOUT:.0f}s); the process is actually terminated "
            "on timeout, not just abandoned. This is for running tests and git, not "
            "for reading or changing files — use list_files/read_file/edit_file/"
            "write_file for that instead. A project's own hooks.py commonly allows "
            "only a short, fixed list of exact commands here (e.g. specific git or "
            "test invocations) and refuses anything containing a shell "
            "metacharacter (chaining, substitution, redirection) outright, so a "
            "command built to work around another tool's limits is more likely to "
            "be denied than to succeed."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "the shell command to run"},
                "timeout": {
                    "type": "number",
                    "description": (
                        f"seconds, default {_DEFAULT_COMMAND_TIMEOUT:.0f}, "
                        f"max {_MAX_COMMAND_TIMEOUT:.0f}"
                    ),
                },
            },
            "required": ["command"],
        },
        fn=_run_command,
        requires_approval=True,
        # A few seconds beyond _MAX_COMMAND_TIMEOUT: subprocess.run's own timeout
        # (below) is what actually bounds this, and always fires first — this is
        # only a backstop in case it somehow doesn't.
        timeout=_MAX_COMMAND_TIMEOUT + 10,
    )


def disable() -> None:
    """Unregister the coding tools. Per-run state, not a global that outlives the
    process that enabled it — the same teardown skills.disable()/subagent.disable()
    already do."""
    for name in _NAMES:
        tools_mod.REGISTRY.pop(name, None)


# -- implementations ----------------------------------------------------------


def _resolve_in_project(path: str) -> tuple[pathlib.Path, pathlib.Path] | str:
    """Shared guard for edit_file/write_file: returns (root, target) or an ERROR
    string. Calls read_file's own traversal guard and deny-list (tools.py) rather
    than re-deriving a second copy of either — a second copy is exactly what this
    used to be, until review found it repeated the naive-string-prefix bug a shared
    helper (tools._resolve_project_path) now fixes in one place."""
    target = tools_mod._resolve_project_path(path)
    if isinstance(target, str):
        return target
    root = tools_mod._get_project_root()
    if tools_mod._is_denied(target.relative_to(root)):
        return (
            f"ERROR: {path} holds credentials or saved agent state and cannot be "
            "read or written by these tools. This is a fixed rule, not a permission "
            "that can be granted, so do not try a different spelling of the path."
        )
    return root, target


def _list_files(path: str = ".", recursive: bool = False) -> str:
    resolved = _resolve_in_project(path)
    if isinstance(resolved, str):
        return resolved
    root, target = resolved
    if not target.is_dir():
        return f"ERROR: no such directory: {path}"

    entries: list[str] = []
    if not recursive:
        for p in sorted(target.iterdir()):
            rel = p.relative_to(root)
            if tools_mod._is_denied(rel):
                continue
            entries.append(f"{rel}/" if p.is_dir() else str(rel))
    else:
        # os.walk with dirnames pruned in place, rather than glob-then-filter: a
        # denied directory like .venv can hold thousands of files, and there is no
        # reason to ever descend into it just to discard every result afterwards.
        for dirpath, dirnames, filenames in os.walk(target):
            dirnames.sort()
            dirnames[:] = [
                d
                for d in dirnames
                if not tools_mod._is_denied((pathlib.Path(dirpath) / d).relative_to(root))
            ]
            for name in sorted(filenames):
                rel = (pathlib.Path(dirpath) / name).relative_to(root)
                if not tools_mod._is_denied(rel):
                    entries.append(str(rel))
    return "\n".join(entries) if entries else f"(empty) {path}"


def _edit_file(path: str, old_string: str, new_string: str) -> str:
    resolved = _resolve_in_project(path)
    if isinstance(resolved, str):
        return resolved
    _root, target = resolved
    if not target.is_file():
        return f"ERROR: no such file: {path}. Use write_file to create a new file."

    # Read fresh rather than trusting the model's own (possibly stale or, before the
    # read_file fix, truncated) view of the file — old_string either matches the
    # real, current content or it does not.
    content = target.read_text(encoding="utf-8", errors="replace")
    count = content.count(old_string)
    if count == 0:
        return (
            f"ERROR: old_string was not found in {path}. Re-read the file with "
            "read_file and match its exact current content."
        )
    if count > 1:
        return (
            f"ERROR: old_string appears {count} times in {path}; it must match "
            "exactly one location. Include more surrounding context to disambiguate."
        )
    updated = content.replace(old_string, new_string, 1)
    broke = _newly_unparsable(target, content, updated)
    if broke is not None:
        # Leave the file exactly as it was. An edit that makes the file unparsable is
        # never the edit that was intended, and the model cannot see that it happened:
        # edit_file's reply used to be "replaced 1 occurrence" whether the result was
        # correct or wreckage. Observed live — a run inserted a keyword argument into a
        # call that already passed it, producing `f(subagent_max_steps=..., ...,
        # subagent_max_steps=...)`; that is a SyntaxError, so every module importing it
        # failed and the agent still reported the task done. Refusing here turns a
        # silently broken repo into one failed tool call the model gets to retry.
        return (
            f"ERROR: that edit was NOT applied — {path} is left unchanged, because "
            f"applying it would have made the file unparsable: {broke}. The edit was "
            "found and matched; what you asked to put there is the problem. Re-read "
            "the file and check whether new_string duplicates something already on "
            "the line below or above it."
        )
    target.write_text(updated, encoding="utf-8")
    at = content.index(old_string)
    return f"Edited {path}: replaced 1 occurrence.\n\n{_edited_region(updated, at, new_string)}"


def _newly_unparsable(target: pathlib.Path, before: str, after: str) -> str | None:
    """The syntax error `after` has and `before` did not, if any.

    Only Python, and only a *regression*: a file already broken when the model found
    it stays the model's to fix, and this must never block the edit that repairs it.
    """
    if target.suffix != ".py":
        return None
    if _compiles(after):
        return None
    if not _compiles(before):
        return None  # already broken before this edit — not ours to refuse
    try:
        compile(after, str(target), "exec")
    except (SyntaxError, ValueError) as exc:
        lineno = getattr(exc, "lineno", None)
        return f"{getattr(exc, 'msg', exc)}" + (f" (line {lineno})" if lineno else "")
    return None


def _compiles(source: str) -> bool:
    """`compile`, not `ast.parse` — deliberately. The duplicate-keyword-argument bug
    this guard exists to catch (`f(a=1, a=2)`) parses cleanly into an AST and is only
    rejected later, when the compiler walks it; `ast.parse` returns happily and the
    broken edit sails through. Verified both ways before relying on it."""
    try:
        compile(source, "<edit-check>", "exec")
        return True
    except (SyntaxError, ValueError):
        return False


def _edited_region(content: str, at: int, new_string: str, context: int = 3) -> str:
    """The edited lines plus a little around them, numbered.

    Returned on success because "replaced 1 occurrence" told the model nothing about
    what it actually wrote. Every wrong-indentation bug observed in a real run was
    invisible for exactly this reason: the model inserted a line at the wrong depth,
    got told the edit succeeded, and moved on. Showing the result next to its
    neighbours makes an indentation mistake visible in the same turn it is made.
    """
    lines = content.splitlines()
    # Derived from where the replacement actually happened, not by searching for the
    # new text: a one-line insertion is very often a copy of a line that also appears
    # earlier in the file, and searching would then show a confidently wrong region.
    idx = content[:at].count("\n")
    start = max(0, idx - context)
    end = min(len(lines), idx + len(new_string.splitlines()) + context)
    width = len(str(end))
    numbered = [f"{i + 1:>{width}} | {lines[i]}" for i in range(start, end)]
    # Cap it. This string is a tool result, and a result over 2000 characters gets
    # externalized to disk and replaced by an excerpt — an echo meant to make one
    # edit reviewable should never be big enough to trigger that machinery itself.
    if len(numbered) > _MAX_ECHO_LINES:
        head, tail = _MAX_ECHO_LINES // 2, _MAX_ECHO_LINES // 2
        numbered = numbered[:head] + [f"{'':>{width}} | ... {len(numbered) - head - tail} more lines ..."] + numbered[-tail:]
    shown = "\n".join(numbered)
    return (
        "The file now reads (check that what you added lines up with its "
        f"neighbours):\n{shown}"
    )


def _write_file(path: str, content: str) -> str:
    resolved = _resolve_in_project(path)
    if isinstance(resolved, str):
        return resolved
    _root, target = resolved
    if target.exists():
        return (
            f"ERROR: {path} already exists. Use edit_file to change it — write_file "
            "only creates new files, so a stale view of an existing file can never "
            "overwrite it through this tool."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} characters to {path}."


def _run_command(command: str, timeout: float | None = None) -> str:
    root = tools_mod._get_project_root()
    effective_timeout = min(timeout, _MAX_COMMAND_TIMEOUT) if timeout else _DEFAULT_COMMAND_TIMEOUT
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=effective_timeout,
        )
    except subprocess.TimeoutExpired:
        # subprocess.run's own timeout actually terminates the child process — unlike
        # the loop's generic per-call timeout (execute_calls, loop.py), which can only
        # abandon a stuck thread, this one really stops the command from running on.
        return (
            f"ERROR: command timed out after {effective_timeout:g}s and was "
            f"terminated: {command!r}"
        )
    output = proc.stdout
    if proc.stderr:
        output += f"\n[stderr]\n{proc.stderr}"
    return f"[exit {proc.returncode}]\n{output}"
