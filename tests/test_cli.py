"""--json: the one stable, parseable contract an external caller relies on.

Everything else `cli.py` prints is a human log and free to change; this is not
(docs/integration.md). The test asserts on stdout being *exactly* one JSON line, not
just "parseable somewhere in the output" — a caller that splits stdout on newlines and
expects one line would break silently if that stopped being true.
"""

import json

from teacup_agent import cli


def test_json_flag_prints_exactly_one_json_object(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # keep run_dir/memory.json out of the repo
    exit_code = cli.main(["2+2", "--json", "--run-dir", "off"])
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.strip()]
    assert len(lines) == 1, f"expected exactly one line on stdout, got: {out!r}"

    result = json.loads(lines[0])
    assert result["status"] == "done"
    assert exit_code == 0
    assert result["exit_code"] == exit_code
    assert "NVIDIA" in result["answer"] or "340" in result["answer"]
    # every state.snapshot() key must be present — --json must not silently drop one
    assert {"goal", "step", "remaining_budget", "elapsed_s", "tool_calls"} <= result.keys()


def test_json_implies_quiet(tmp_path, monkeypatch, capsys):
    """Without --json, offline mode prints a multi-line human log to the same stdout."""
    monkeypatch.chdir(tmp_path)
    cli.main(["2+2", "--run-dir", "off"])
    human_lines = len(capsys.readouterr().out.splitlines())

    cli.main(["2+2", "--json", "--run-dir", "off"])
    json_lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]

    assert human_lines > 1
    assert len(json_lines) == 1
