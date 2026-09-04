"""coding_tools.py: list_files/edit_file/write_file/run_command, opt-in behind
--coding-tools. These are the tools that turn this from a research agent into one
that can also change and verify a repository, so the deny-list/traversal-guard reuse
and the ambiguity/overwrite guards matter as much as the happy path.
"""

from __future__ import annotations

import json

import pytest

from teacup_agent import coding_tools, tools


@pytest.fixture(autouse=True)
def _project(tmp_path, monkeypatch):
    """Every test gets its own project root, and coding_tools enabled/disabled
    around it — the same per-test isolation test_hooks.py's _clean_hooks fixture
    uses, since REGISTRY and _project_root are both module-global state."""
    tools.set_project_root(tmp_path)
    coding_tools.enable()
    yield tmp_path
    coding_tools.disable()
    tools.set_project_root(None)


def _call(name: str, **kwargs) -> str:
    return tools.execute(name, json.dumps(kwargs))


# --- enable/disable ------------------------------------------------------------


def test_enable_registers_all_four_tools():
    for name in ("list_files", "edit_file", "write_file", "run_command"):
        assert name in tools.REGISTRY


def test_disable_unregisters_all_four_tools():
    coding_tools.disable()
    for name in ("list_files", "edit_file", "write_file", "run_command"):
        assert name not in tools.REGISTRY
    coding_tools.enable()  # so the fixture's own teardown has something to remove


def test_edit_file_and_write_file_require_approval_list_files_does_not():
    assert tools.REGISTRY["list_files"].requires_approval is False
    assert tools.REGISTRY["edit_file"].requires_approval is True
    assert tools.REGISTRY["write_file"].requires_approval is True
    assert tools.REGISTRY["run_command"].requires_approval is True


# --- list_files ------------------------------------------------------------------


def test_list_files_lists_top_level_entries(_project):
    (_project / "a.txt").write_text("a")
    (_project / "sub").mkdir()
    out = _call("list_files")
    assert "a.txt" in out and "sub/" in out


def test_list_files_on_project_root_itself_does_not_crash(_project):
    """Regression test: relative_to() returns a zero-part path when the target *is*
    the project root, and _is_denied() used to crash on parts[-1] with an empty
    list. list_files(".") is exactly this case and is the normal, expected call."""
    (_project / "notes.md").write_text("hi")
    out = _call("list_files", path=".")
    assert "notes.md" in out


def test_list_files_recursive_walks_subdirs(_project):
    (_project / "sub").mkdir()
    (_project / "sub" / "nested.txt").write_text("x")
    out = _call("list_files", recursive=True)
    assert "sub/nested.txt" in out


def test_list_files_hides_denied_entries(_project):
    (_project / ".git").mkdir()
    (_project / ".git" / "config").write_text("x")
    (_project / "mcp.json").write_text("{}")
    (_project / "visible.txt").write_text("x")
    out = _call("list_files", recursive=True)
    assert ".git" not in out
    assert "mcp.json" not in out
    assert "visible.txt" in out


def test_list_files_recursive_prunes_denied_dirs_without_descending(_project, monkeypatch):
    """os.walk should never even enter a denied directory, not just filter its
    contents afterward — verified by making .venv unreadable and confirming the
    walk still succeeds instead of raising on it."""
    venv = _project / ".venv"
    venv.mkdir()
    (venv / "lib").mkdir()
    out = _call("list_files", recursive=True)
    assert ".venv" not in out


def test_list_files_rejects_traversal_outside_project(_project):
    out = _call("list_files", path="../../../etc")
    assert out.startswith("ERROR:")


def test_list_files_rejects_a_sibling_directory_sharing_the_root_as_a_string_prefix(_project):
    """Regression test (same bug tests/test_tools.py pins for read_file): a sibling
    directory that merely shares the project root's characters as a string prefix —
    _project is .../pytest-.../test_x0, a sibling .../test_x0-secrets starts with that
    string — used to slip past the old `str(target).startswith(str(root))` guard.
    is_relative_to() (tools._resolve_project_path) does the real check now."""
    sibling = _project.parent / (_project.name + "-secrets")
    sibling.mkdir()
    (sibling / "creds.txt").write_text("hunter2", encoding="utf-8")
    out = _call("list_files", path=f"../{sibling.name}")
    assert out.startswith("ERROR:")


def test_list_files_errors_on_nonexistent_directory(_project):
    out = _call("list_files", path="does-not-exist")
    assert out.startswith("ERROR:")


def test_list_files_reports_empty_directory(_project):
    (_project / "empty").mkdir()
    out = _call("list_files", path="empty")
    assert "empty" in out.lower()


# --- edit_file ---------------------------------------------------------------------


def test_edit_file_replaces_unique_occurrence(_project):
    (_project / "f.py").write_text("x = 1\ny = 2\n")
    out = _call("edit_file", path="f.py", old_string="x = 1", new_string="x = 100")
    assert "replaced 1 occurrence" in out
    assert (_project / "f.py").read_text() == "x = 100\ny = 2\n"


def test_edit_file_errors_when_old_string_not_found(_project):
    (_project / "f.py").write_text("x = 1\n")
    out = _call("edit_file", path="f.py", old_string="z = 9", new_string="z = 10")
    assert out.startswith("ERROR:") and "not found" in out
    assert (_project / "f.py").read_text() == "x = 1\n"  # unchanged


def test_edit_file_errors_when_ambiguous(_project):
    (_project / "f.py").write_text("foo\nfoo\n")
    out = _call("edit_file", path="f.py", old_string="foo", new_string="bar")
    assert out.startswith("ERROR:") and "2 times" in out
    assert (_project / "f.py").read_text() == "foo\nfoo\n"  # unchanged


def test_edit_file_errors_on_nonexistent_file(_project):
    out = _call("edit_file", path="missing.py", old_string="a", new_string="b")
    assert out.startswith("ERROR:") and "write_file" in out


def test_edit_file_respects_deny_list(_project):
    (_project / ".env").write_text("SECRET=1")
    out = _call("edit_file", path=".env", old_string="SECRET=1", new_string="SECRET=2")
    assert out.startswith("ERROR:") and "not readable" not in out  # own wording, not read_file's
    assert (_project / ".env").read_text() == "SECRET=1"  # unchanged


def test_edit_file_rejects_traversal_outside_project(_project):
    out = _call("edit_file", path="../outside.txt", old_string="a", new_string="b")
    assert out.startswith("ERROR:")


def test_edit_file_reads_current_content_not_a_stale_view(_project):
    """The structural point of edit_file: it always reads the real file fresh, so
    there is no way for a caller's outdated idea of the content to cause a wrong
    replacement — old_string either matches what's actually on disk or it errors."""
    path = _project / "f.py"
    path.write_text("version 1")
    path.write_text("version 2")  # changed after any hypothetical earlier read
    out = _call("edit_file", path="f.py", old_string="version 1", new_string="version 3")
    assert out.startswith("ERROR:") and "not found" in out
    assert path.read_text() == "version 2"


# --- write_file ----------------------------------------------------------------------


def test_write_file_creates_new_file(_project):
    out = _call("write_file", path="new.txt", content="hello")
    assert "Wrote" in out
    assert (_project / "new.txt").read_text() == "hello"


def test_write_file_errors_if_file_exists(_project):
    (_project / "exists.txt").write_text("original")
    out = _call("write_file", path="exists.txt", content="overwritten")
    assert out.startswith("ERROR:") and "already exists" in out
    assert (_project / "exists.txt").read_text() == "original"  # never touched


def test_write_file_creates_parent_directories(_project):
    out = _call("write_file", path="a/b/c.txt", content="deep")
    assert "Wrote" in out
    assert (_project / "a" / "b" / "c.txt").read_text() == "deep"


def test_write_file_respects_deny_list(_project):
    out = _call("write_file", path="mcp.json", content='{"servers": {}}')
    assert out.startswith("ERROR:")
    assert not (_project / "mcp.json").exists()


def test_write_file_rejects_traversal_outside_project(_project):
    out = _call("write_file", path="../escape.txt", content="x")
    assert out.startswith("ERROR:")


# --- run_command ----------------------------------------------------------------------


def test_run_command_returns_exit_code_and_stdout(_project):
    out = _call("run_command", command="echo hello")
    assert out.startswith("[exit 0]")
    assert "hello" in out


def test_run_command_reports_nonzero_exit(_project):
    out = _call("run_command", command="exit 7")
    assert out.startswith("[exit 7]")


def test_run_command_captures_stderr(_project):
    out = _call("run_command", command="echo oops 1>&2")
    assert "[stderr]" in out and "oops" in out


def test_run_command_runs_inside_the_project_root(_project):
    (_project / "marker.txt").write_text("here")
    out = _call("run_command", command="cat marker.txt")
    assert "[exit 0]" in out and "here" in out


def test_run_command_timeout_actually_terminates_the_process(_project):
    out = _call("run_command", command="sleep 5", timeout=0.3)
    assert out.startswith("ERROR:") and "timed out" in out


def test_run_command_timeout_is_capped_at_the_maximum(_project, monkeypatch):
    """A model cannot ask for an unbounded run by requesting a huge timeout — it is
    clamped to _MAX_COMMAND_TIMEOUT before being handed to subprocess.run."""
    captured = {}
    import subprocess as subprocess_mod

    real_run = subprocess_mod.run

    def spy(*args, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        return real_run("echo x", shell=True, cwd=kwargs["cwd"], capture_output=True, text=True, timeout=5)

    monkeypatch.setattr(coding_tools.subprocess, "run", spy)
    _call("run_command", command="echo x", timeout=999999)
    assert captured["timeout"] == coding_tools._MAX_COMMAND_TIMEOUT
