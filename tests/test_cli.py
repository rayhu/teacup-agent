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


# --- the hosted-search money gate --------------------------------------------


def test_hosted_search_is_refused_without_live():
    """--live is this repo's money gate, and the offline demo really does call
    search_web — so this combination would bill a real API call from the path README
    and --help describe as key-less and instant."""
    assert "needs --live" in cli._search_refusal("hosted", live=False)


def test_hosted_search_is_allowed_with_live():
    assert cli._search_refusal("hosted", live=True) == ""


def test_every_free_mode_is_allowed_either_way():
    for mode in (None, "auto", "web", "offline"):
        assert cli._search_refusal(mode, live=False) == ""
        assert cli._search_refusal(mode, live=True) == ""


def test_the_refusal_honours_the_json_contract(capsys):
    """docs/integration.md promises exactly one JSON object on stdout and an exit_code
    field. A bare SystemExit would hand an external caller empty stdout to guess at."""
    code = cli.main(["x", "--search", "hosted", "--json"])
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)  # one object, parseable
    # integration.md: exit_code is 0 when status is "done" and 1 otherwise, and every
    # field through `throttled` is snapshot() unchanged — a caller reading
    # remaining_budget must not KeyError on a refusal.
    assert code == 1 and payload["exit_code"] == 1
    assert payload["status"] == "error" and "needs --live" in payload["answer"]
    for field in ("remaining_budget", "spend", "step", "max_steps", "throttled"):
        assert field in payload, f"the refusal payload dropped {field}"
