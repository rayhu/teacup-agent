import json

import pytest

from teacup_agent import tools


def test_specs_are_openai_shaped():
    specs = tools.specs()
    assert {s["function"]["name"] for s in specs} >= {"search_web", "calculate", "remember"}
    for s in specs:
        assert s["type"] == "function"
        assert s["function"]["parameters"]["type"] == "object"


def test_execute_parses_json_string_arguments():
    assert tools.execute("calculate", json.dumps({"expression": "(1200*0.85)/3"})) == "340.0"


def test_bad_json_becomes_error_result_not_exception():
    assert tools.execute("calculate", "{oops").startswith("ERROR:")


def test_wrong_argument_name_becomes_error_result():
    assert tools.execute("calculate", '{"expr": "1+1"}').startswith("ERROR:")


def test_unsafe_expression_is_rejected():
    assert tools.execute("calculate", '{"expression": "__import__(\'os\').system(\'ls\')"}').startswith("ERROR:")


def test_read_file_cannot_escape_project_dir():
    assert tools.execute("read_file", '{"path": "../../../etc/passwd"}').startswith("ERROR:")


# --- the three search_web modes, and "nothing beats the wrong thing" ---------


def test_offline_search_does_not_false_positive(monkeypatch):
    """Regression test: "OpenAI strategy" once matched the NVIDIA corpus entry
    because matching used any()."""
    monkeypatch.setenv("TEACUP_AGENT_SEARCH", "offline")
    out = tools.execute("search_web", '{"query": "OpenAI strategy 2026 roadmap"}')
    assert "NVIDIA" not in out
    assert "No results" in out


def test_offline_search_still_matches_full_key(monkeypatch):
    monkeypatch.setenv("TEACUP_AGENT_SEARCH", "offline")
    assert "NVIDIA" in tools.execute("search_web", '{"query": "what is the nvidia gpu strategy"}')


def test_auto_mode_falls_back_to_corpus_when_network_fails(monkeypatch):
    monkeypatch.setenv("TEACUP_AGENT_SEARCH", "auto")
    monkeypatch.setattr(tools, "_search_web_backend", lambda q, n: (_ for _ in ()).throw(OSError("no net")))
    out = tools.execute("search_web", '{"query": "cuda"}')
    assert out.startswith("[web search failed") and "CUDA" in out


def test_web_mode_reports_error_instead_of_pretending(monkeypatch):
    """In strict mode a failed search must be an ERROR, never something the model
    can read as "I looked and it does not exist"."""
    monkeypatch.setenv("TEACUP_AGENT_SEARCH", "web")
    monkeypatch.setattr(tools, "_search_web_backend", lambda q, n: (_ for _ in ()).throw(OSError("no net")))
    assert tools.execute("search_web", '{"query": "cuda"}').startswith("ERROR:")


def test_web_search_formats_results_with_links(monkeypatch):
    monkeypatch.setenv("TEACUP_AGENT_SEARCH", "web")
    monkeypatch.setattr(
        tools,
        "DDGS" if hasattr(tools, "DDGS") else "_search_web_backend",
        lambda q, n: f"1. Title\n   https://example.com\n   snippet (n={n})",
    )
    out = tools.execute("search_web", '{"query": "x", "max_results": 99}')
    assert "https://example.com" in out and "n=10" in out  # count clamped to 10


# --- search failures: backoff retries, never disguised as "nothing found" ----


class _FlakyDDGS:
    """Raise for the first `fail_times` calls, then return results."""

    calls = 0

    def __init__(self, fail_times=0, results=None):
        type(self).fail_times = fail_times
        type(self).results = results or [{"title": "T", "href": "https://e.com", "body": "B"}]

    def text(self, query, max_results=5):
        type(self).calls += 1
        if type(self).calls <= type(self).fail_times:
            raise RuntimeError("Ratelimit")
        return type(self).results


@pytest.fixture
def fake_ddgs(monkeypatch):
    import ddgs

    _FlakyDDGS.calls = 0
    monkeypatch.setattr(tools.time, "sleep", lambda *_: None)  # tests never wait
    monkeypatch.setattr(tools, "_last_search_at", 0.0)

    def install(fail_times=0):
        monkeypatch.setattr(ddgs, "DDGS", lambda: _FlakyDDGS(fail_times))
        return _FlakyDDGS

    return install


def test_search_retries_then_succeeds(monkeypatch, fake_ddgs):
    monkeypatch.setenv("TEACUP_AGENT_SEARCH", "web")
    flaky = fake_ddgs(fail_times=2)
    out = tools.execute("search_web", '{"query": "anything"}')
    assert "https://e.com" in out
    assert flaky.calls == 3  # two failures, then success on the third


def test_search_failure_is_not_disguised_as_no_results(monkeypatch, fake_ddgs):
    """For a question the corpus knows nothing about, a broken search must report
    ERROR rather than "no results"."""
    monkeypatch.setenv("TEACUP_AGENT_SEARCH", "auto")
    flaky = fake_ddgs(fail_times=99)
    out = tools.execute("search_web", '{"query": "Anthropic funding last six months"}')
    assert out.startswith("ERROR:") and "does **not** mean" in out
    assert "No results" not in out
    assert flaky.calls == tools._RETRIES  # all retries were used


# --- read_file's deny-list ----------------------------------------------------
#
# The directory guard answers "where"; the project directory is exactly where the
# secrets live. These answer "what".


@pytest.mark.parametrize(
    "path", [".env", ".env.local", "mcp.json", "memory.json", "server.pem", ".git/config"]
)
def test_credentials_and_config_are_not_readable(tmp_path, monkeypatch, path):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("SECRET=hunter2", encoding="utf-8")

    out = tools.execute("read_file", json.dumps({"path": path}))
    assert out.startswith("ERROR:") and "not readable" in out
    assert "hunter2" not in out


def test_the_refusal_tells_the_model_not_to_retry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("K=v", encoding="utf-8")
    out = tools.execute("read_file", '{"path": ".env"}')
    # A model that reads "denied" often tries a different spelling; say it is fixed.
    assert "not a permission that can be granted" in out


def test_a_saved_trajectory_is_denied_but_externalized_results_are_not(tmp_path, monkeypatch):
    """`runs/` needs a distinction, not a blanket rule: the externalizer writes large
    tool results there and tells the model to read them back."""
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "runs" / "20260826-000000"
    run.mkdir(parents=True)
    (run / "state.json").write_text('{"messages": "the whole system prompt"}', encoding="utf-8")
    (run / "step01_0_search_web.txt").write_text("page text the model already saw", encoding="utf-8")

    denied = tools.execute("read_file", '{"path": "runs/20260826-000000/state.json"}')
    allowed = tools.execute("read_file", '{"path": "runs/20260826-000000/step01_0_search_web.txt"}')
    assert denied.startswith("ERROR:") and "system prompt" not in denied
    assert allowed == "page text the model already saw"


def test_ordinary_project_files_still_read(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notes.md").write_text("readable", encoding="utf-8")
    assert tools.execute("read_file", '{"path": "notes.md"}') == "readable"
