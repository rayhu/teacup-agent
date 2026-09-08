import json
from types import SimpleNamespace

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


def test_read_file_no_longer_truncates_at_2000_chars(tmp_path, monkeypatch):
    """Regression test: read_file used to slice its result to [:2000], which meant a
    model editing from that view could clobber the part it never saw. The loop's own
    EXTERNALIZE_OVER mechanism is what should handle long results now, not a second,
    earlier cap here."""
    monkeypatch.chdir(tmp_path)
    long_content = "line\n" * 1000  # 5000 chars, well past the old cap
    (tmp_path / "big.txt").write_text(long_content, encoding="utf-8")
    assert tools.execute("read_file", '{"path": "big.txt"}') == long_content


def test_read_file_rejects_a_sibling_directory_sharing_the_root_as_a_string_prefix(tmp_path):
    """Regression test: the traversal guard used to be `str(target).startswith(str(root))`,
    which a sibling directory that merely shares the root's characters as a string prefix
    defeats — root=.../repo, target=.../repo-secrets/f.txt starts with str(root) even
    though it is a completely different directory. It only used to fail safe because the
    next line's target.relative_to(root) happened to raise, as an accident of ordering,
    not by design — and that raised ValueError, not the guard's own ERROR message.
    _resolve_project_path now uses is_relative_to(), the real component-wise check."""
    root = tmp_path / "repo"
    root.mkdir()
    tools.set_project_root(root)
    sibling = tmp_path / "repo-secrets"
    sibling.mkdir()
    (sibling / "creds.txt").write_text("hunter2", encoding="utf-8")
    try:
        out = tools.execute("read_file", '{"path": "../repo-secrets/creds.txt"}')
    finally:
        tools.set_project_root(None)
    assert out == "ERROR: only paths inside the current project directory are allowed"


# --- an explicit project root, independent of the launch directory (#14) -------------


def test_project_root_defaults_to_cwd_when_never_set(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notes.md").write_text("readable", encoding="utf-8")
    assert tools.execute("read_file", '{"path": "notes.md"}') == "readable"


def test_project_root_can_be_set_independent_of_the_launch_directory(tmp_path, monkeypatch):
    """The bug #14 fixes: before this, 'project root' and 'the directory the process
    launched from' were definitionally the same value, so a file outside the intended
    project but inside the launch directory was unreachable to test at all. Launch from
    one directory, set the root to a different one, and confirm the launch directory's
    own files are refused even though the traversal guard alone would have allowed
    them."""
    launch_dir = tmp_path / "launch"
    project_dir = tmp_path / "project"
    launch_dir.mkdir()
    project_dir.mkdir()
    (launch_dir / "secret.txt").write_text("should not be reachable", encoding="utf-8")
    (project_dir / "notes.md").write_text("readable", encoding="utf-8")

    monkeypatch.chdir(launch_dir)
    tools.set_project_root(project_dir.resolve())
    try:
        outside = tools.execute("read_file", '{"path": "secret.txt"}')
        inside = tools.execute("read_file", '{"path": "notes.md"}')
    finally:
        tools.set_project_root(None)  # must not leak into other tests

    assert outside.startswith("ERROR:") and "should not be reachable" not in outside
    assert inside == "readable"


# --- hosted search backend (#11) ---------------------------------------------


class _Ann(SimpleNamespace):
    pass


def _hosted_response(summary, citations, *, searched=True, usage=None, status="completed", n=1):
    """A Responses object shaped the way the hosted web_search tool returns one.

    `searched=False` models the case the docs are explicit about — "the model can
    choose to search the web or not" — where no web_search_call item is produced and
    the prose is recall, not retrieval.
    """
    anns = [_Ann(type="url_citation", url=u, title=t) for t, u in citations]
    items = []
    if searched:
        items.extend(SimpleNamespace(type="web_search_call", status=status) for _ in range(n))
    items.append(SimpleNamespace(type="message", content=[SimpleNamespace(annotations=anns)]))
    return SimpleNamespace(output_text=summary, output=items, usage=usage)


def _fake_openai(resp, monkeypatch):
    sent = {}

    class Responses:
        def create(self, **kwargs):
            sent.update(kwargs)
            return resp

    class FakeOpenAI:
        def __init__(self, *a, **k):
            sent["_client_kwargs"] = k
            self.responses = Responses()

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("TEACUP_AGENT_SEARCH", "hosted")
    tools.take_hosted_spend()  # zero the accumulator between tests
    return sent


def test_hosted_search_returns_sources_and_a_summary(monkeypatch):
    resp = _hosted_response(
        "NVIDIA sells GPUs.", [("NVIDIA Q3", "https://a.example"), ("Analysis", "https://b.example")]
    )
    sent = _fake_openai(resp, monkeypatch)

    out = tools.search_web("nvidia strategy")
    assert "https://a.example" in out and "https://b.example" in out
    assert "NVIDIA sells GPUs." in out
    assert sent["tools"] == [{"type": "web_search"}]
    assert sent["tool_choice"] == "required"  # forced, not offered


def test_prose_is_withheld_when_the_model_never_actually_searched(monkeypatch):
    """The whole point of this backend. A model that chose not to search answered from
    memory, and returning that as a search result is the failure #11 exists to remove.

    The earlier version of this test asserted only that no numbered source list
    appeared, which the un-cited prose passes — so the hallucinated URL went out to the
    model and the test stayed green. Assert on the URL itself.
    """
    resp = _hosted_response("See https://hallucinated.example for details.", [], searched=False)
    _fake_openai(resp, monkeypatch)

    out = tools.search_web("q")
    assert "hallucinated.example" not in out
    assert out.startswith("ERROR: the hosted search did not run a query")
    assert "does **not** mean the information does not exist" in out


def test_a_search_that_returns_nothing_citable_withholds_the_summary(monkeypatch):
    """A search ran but produced no citation: there is no way to tell grounded text
    from recalled text, so the paragraph does not go out as a result."""
    resp = _hosted_response("Acme raised $2B, see https://made-up.example", [], searched=True)
    _fake_openai(resp, monkeypatch)

    out = tools.search_web("acme funding")
    assert "made-up.example" not in out
    assert "no citable sources" in out
    assert not out.startswith("ERROR")  # the search worked; it just found nothing


def test_hosted_search_respects_max_results(monkeypatch):
    cites = [(f"t{i}", f"https://{i}.example") for i in range(10)]
    _fake_openai(_hosted_response("s", cites), monkeypatch)

    out = tools.search_web("q", max_results=3)
    assert out.count("https://") == 3


def test_hosted_search_failure_is_an_error_never_the_offline_corpus(monkeypatch):
    """A paid backend quietly answering from a local corpus is worse than saying it
    failed — the model cannot tell it is reading a fixture."""
    class Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("upstream 503")

    import openai

    monkeypatch.setattr(openai, "OpenAI", Boom)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("TEACUP_AGENT_SEARCH", "hosted")

    out = tools.search_web("q")
    assert out.startswith("ERROR: hosted search failed")
    assert "does **not** mean the information does not exist" in out


def test_hosted_search_without_a_key_says_it_is_configuration_not_weather(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("TEACUP_AGENT_SEARCH", "hosted")
    out = tools.search_web("q")
    assert "OPENAI_API_KEY" in out
    # a permanent config error must not be dressed up as a transient one, or the model
    # keeps retrying a mode that can never work
    assert "Retry later" not in out


def test_auto_never_reaches_for_the_paid_backend(monkeypatch):
    """Picking the backend that costs money should be a decision someone made, not
    one an unset environment variable made for them."""
    called = []
    monkeypatch.setattr(tools, "_search_hosted_backend", lambda *a: called.append(a) or "hosted")
    monkeypatch.setattr(tools, "_search_web_backend", lambda *a: "scraped")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("TEACUP_AGENT_SEARCH", raising=False)

    assert tools.search_web("q") == "scraped"
    assert called == []


def test_the_hosted_call_is_charged_so_the_budget_brake_can_see_it(monkeypatch):
    """This is the only tool in the repo that spends money. Left uncharged, a run can
    burn many times its stated ceiling while state.snapshot() reports it untouched."""
    usage = SimpleNamespace(input_tokens=1000, output_tokens=500)
    _fake_openai(_hosted_response("s", [("t", "https://a.example")], usage=usage), monkeypatch)

    tools.search_web("q")
    spent = tools.take_hosted_spend()
    assert spent >= tools._HOSTED_CALL_FEE  # the per-call fee, plus tokens
    assert tools.take_hosted_spend() == 0.0  # draining resets it


def test_the_search_model_is_configurable(monkeypatch):
    sent = _fake_openai(_hosted_response("s", [("t", "https://a.example")]), monkeypatch)
    monkeypatch.setenv("TEACUP_AGENT_SEARCH_MODEL", "gpt-5")
    tools.search_web("q")
    assert sent["model"] == "gpt-5"


def test_the_client_carries_a_timeout(monkeypatch):
    """The loop's 30s per-tool timeout cannot cancel a thread already inside an HTTP
    call: the request still completes and still bills while the model retries."""
    sent = _fake_openai(_hosted_response("s", [("t", "https://a.example")]), monkeypatch)
    tools.search_web("q")
    assert sent["_client_kwargs"]["timeout"] == tools._HOSTED_TIMEOUT


def test_a_failed_search_is_not_reported_as_nothing_to_find(monkeypatch):
    """web_search_call carries a status — completed, failed, incomplete. Asking only
    whether the item exists turns a broken search into "the information does not
    exist", which is the distinction this repo has been bitten by before."""
    _fake_openai(_hosted_response("", [], status="failed"), monkeypatch)

    out = tools.search_web("does it exist")
    assert out.startswith("ERROR: the hosted search did not complete")
    assert "does **not** mean the information does not exist" in out


def test_the_fee_counts_search_actions_not_api_calls(monkeypatch):
    """A reasoning model issues several searches in one response; OpenAI bills per
    search action."""
    _fake_openai(_hosted_response("s", [("t", "https://a.example")], n=3), monkeypatch)
    tools.search_web("q")
    assert tools.take_hosted_spend() == pytest.approx(3 * tools._HOSTED_CALL_FEE)


def test_a_response_that_never_searched_is_not_billed_a_search_fee(monkeypatch):
    _fake_openai(_hosted_response("from memory", [], searched=False), monkeypatch)
    tools.search_web("q")
    assert tools.take_hosted_spend() == 0.0


def test_spend_from_one_thread_is_invisible_to_another(monkeypatch):
    """Thread isolation *is* the billing contract, and it is what the two earlier
    designs lacked.

    `execute_calls` runs each tool in its own worker thread, and `delegate` starts a
    nested run inside one of them — so a turn issuing `search_web` and `delegate`
    together has the parent's search finishing while the child is live. With a module
    global the child's collection either wiped that fee or billed it to the child,
    cutting a child funded for several steps down to one. Neither a reset-at-run-start
    nor a take-and-give-back bracket fixed it; only a per-thread accumulator does.

    (An end-to-end version that forces the interleaving was attempted and dropped: the
    ordering could not be made deterministic, and a concurrency test that passes
    whatever the design is worse than none.)
    """
    import threading

    from teacup_agent import tools as tools_mod

    seen = {}
    added, main_has_read = threading.Event(), threading.Event()

    def other_thread():
        tools_mod._add_hosted_spend(0.05)  # a sibling run's search
        added.set()
        main_has_read.wait(timeout=5)  # hold it *pending* while this thread reads
        seen["other"] = tools_mod.take_hosted_spend()

    t = threading.Thread(target=other_thread)
    t.start()
    added.wait(timeout=5)
    seen["this"] = tools_mod.take_hosted_spend()  # sibling's 0.05 is still pending here
    main_has_read.set()
    t.join(timeout=5)

    assert seen["other"] == pytest.approx(0.05)  # the thread that spent it collects it
    assert seen["this"] == 0.0  # and nobody else can


def test_hosted_spend_reaches_the_budget_brake_through_the_loop(monkeypatch):
    """The accumulator only means something if the run charges it. Without the
    state.charge() in execute_calls a run could spend many times its ceiling while
    remaining_budget sat still."""
    from teacup_agent import loop, tools as tools_mod
    from teacup_agent.memory import NullMemory
    from teacup_agent.model import ScriptedModel, assistant_calls, assistant_says

    _fake_openai(_hosted_response("s", [("t", "https://a.example")]), monkeypatch)
    monkeypatch.setenv("TEACUP_AGENT_SEARCH", "hosted")

    events = []
    state = loop.run(
        "search for something",
        ScriptedModel([assistant_calls([("search_web", {"query": "q"})]), assistant_says("done")]),
        memory=NullMemory(),
        budget=0.05,
        run_dir=None,
        plan=False,
        on_event=lambda name, data: events.append((name, data)),
    )
    # the per-call fee landed against the budget, not just in the module global
    assert state.remaining_budget < 0.05 - tools_mod._HOSTED_CALL_FEE / 2
    assert tools_mod.take_hosted_spend() == 0.0  # fully drained by the loop
    assert any(e[0] == "tool_spend" for e in events)


def test_hosted_spend_can_stop_a_run_mid_flight(monkeypatch):
    """Asserts the brake itself, not just the arithmetic: a budget that affords one
    search and not two must stop the run before the step ceiling. Asserting only the
    final `remaining_budget` cannot tell "charged" from "charged in time to stop
    anything", and the brake is the reason the charge exists."""
    from teacup_agent import loop, tools as tools_mod
    from teacup_agent.memory import NullMemory
    from teacup_agent.model import ScriptedModel, assistant_calls

    _fake_openai(_hosted_response("s", [("t", "https://a.example")]), monkeypatch)
    monkeypatch.setenv("TEACUP_AGENT_SEARCH", "hosted")

    search = assistant_calls([("search_web", {"query": "q"})])
    state = loop.run(
        "search repeatedly",
        ScriptedModel([search] * 6),
        memory=NullMemory(),
        budget=tools_mod._HOSTED_CALL_FEE * 1.5,  # affords one search, not two
        max_steps=6,
        run_dir=None,
        plan=False,
    )
    assert state.status == "out_of_budget"
    assert state.step < 6  # stopped by the money, not by the step ceiling


def test_a_search_on_the_final_step_is_still_charged(monkeypatch):
    """A search on the last step there is. Billing now happens inside execute_calls, in
    the worker thread that made the call, so there is no "collected on a later
    iteration" path left to get wrong — this pins that."""
    from teacup_agent import loop, tools as tools_mod
    from teacup_agent.memory import NullMemory
    from teacup_agent.model import ScriptedModel, assistant_calls

    _fake_openai(_hosted_response("s", [("t", "https://a.example")]), monkeypatch)
    monkeypatch.setenv("TEACUP_AGENT_SEARCH", "hosted")

    state = loop.run(
        "search",
        ScriptedModel([assistant_calls([("search_web", {"query": "q"})])]),
        memory=NullMemory(),
        budget=0.05,
        max_steps=1,  # the search happens on the last step there is
        run_dir=None,
        plan=False,
    )
    assert state.remaining_budget < 0.05 - tools_mod._HOSTED_CALL_FEE / 2
    assert tools_mod.take_hosted_spend() == 0.0


def test_cached_input_tokens_are_billed_at_the_cached_rate(monkeypatch):
    """A cache hit charged at fresh rates overstates the cost of every repeated query."""
    from teacup_agent import tools as tools_mod

    details = SimpleNamespace(cached_tokens=1_000_000)
    usage = SimpleNamespace(input_tokens=1_000_000, output_tokens=0, input_tokens_details=details)
    _fake_openai(_hosted_response("s", [("t", "https://a.example")], usage=usage), monkeypatch)

    tools.search_web("q")
    spent = tools_mod.take_hosted_spend() - tools_mod._HOSTED_CALL_FEE
    # gpt-5-mini: 0.25 fresh vs 0.025 cached per 1M. Billing this as fresh would be 10x.
    assert spent == pytest.approx(0.025, rel=0.05)


def test_a_truncated_response_is_not_reported_as_nothing_found(monkeypatch):
    """The response itself can be cut short (status "incomplete", reason
    max_output_tokens) after the search completed but before the citations landed.
    Checking only the search item's status made that read as an absence — and the
    zero-citation wording explicitly tells the model to trust it."""
    resp = _hosted_response("partial", [], status="completed")
    resp.status = "incomplete"
    resp.incomplete_details = SimpleNamespace(reason="max_output_tokens")
    _fake_openai(resp, monkeypatch)

    out = tools.search_web("does acme have a 2026 filing")
    assert out.startswith("ERROR: the hosted search did not complete")
    assert "max_output_tokens" in out
    assert "nothing found" not in out


def test_an_unfinished_search_item_is_neither_billed_nor_counted_as_a_result():
    """web_search_call has a status. in_progress, searching, or absent all mean the
    search cannot be shown to have completed — so on a money path the safe reading is
    the one that does not charge, and does not claim a result either."""
    from teacup_agent import tools as tools_mod

    for status in ("in_progress", "searching", None):
        resp = SimpleNamespace(
            output=[SimpleNamespace(type="web_search_call", status=status)], output_text=""
        )
        done, broken = tools_mod._search_actions(resp)
        assert done == 0, f"{status!r} was counted as a completed search"
        assert broken, f"{status!r} was silently ignored"


def test_a_mixed_response_bills_its_completed_searches_and_still_reports_the_failure():
    from teacup_agent import tools as tools_mod

    resp = SimpleNamespace(
        output=[
            SimpleNamespace(type="web_search_call", status="completed"),
            SimpleNamespace(type="web_search_call", status="failed"),
        ],
        output_text="",
    )
    done, broken = tools_mod._search_actions(resp)
    assert done == 1  # the completed one is billable
    assert "failed" in broken  # and the failure is not masked by it


def test_permanent_hosted_failures_are_not_dressed_up_as_transient(monkeypatch):
    """A missing key raises _SearchNotConfigured; several other permanent failures do
    not, and were taking the "retry later" branch — which sends the model back to a mode
    that cannot work until a human edits something, until the step ceiling.

    These are the SDK's own classes, imported, not stand-ins: the fix matches on type
    *name*, so a test defining its own `class AuthenticationError` would only prove that
    string comparison works against a string the test chose. Importing them means a
    rename upstream fails here instead of silently reopening the retry loop.
    """
    import openai

    permanent = [
        openai.AuthenticationError,  # key rejected
        openai.PermissionDeniedError,  # not entitled
        openai.NotFoundError,  # TEACUP_AGENT_SEARCH_MODEL names a missing model
    ]
    for exc_type in permanent:
        def boom(*a, _t=exc_type, **k):
            raise _t.__new__(_t)  # these take a response kwarg; the type is the point

        monkeypatch.setattr(openai, "OpenAI", boom)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("TEACUP_AGENT_SEARCH", "hosted")

        out = tools.search_web("q")
        assert "not configured" in out, f"{exc_type.__name__} read as transient"
        assert "Retry later" not in out, f"{exc_type.__name__} told the model to retry"


def test_an_ambiguous_400_stays_retryable(monkeypatch):
    """BadRequestError can mean "this model cannot use web_search" — permanent — or it
    can mean the model wrote a query the API rejected, which rewording fixes. Every HTTP
    400 becomes this one class, so the two are indistinguishable by type. Calling it
    permanent would disable search for the rest of the run over one bad phrasing, and
    would drop the "reword the query" advice that is the actionable half."""
    import openai

    def boom(*a, **k):
        raise openai.BadRequestError.__new__(openai.BadRequestError)

    monkeypatch.setattr(openai, "OpenAI", boom)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("TEACUP_AGENT_SEARCH", "hosted")

    out = tools.search_web("q")
    assert "reword the query" in out
    assert "not configured" not in out


def test_a_broken_import_inside_the_backend_is_not_read_as_misconfiguration(monkeypatch):
    """openai is a hard dependency, so "not installed" is near-unreachable — but an
    ImportError raised by our own code inside this backend would then reach the model as
    "this is a setup problem" instead of as the bug it is."""
    import openai

    def boom(*a, **k):
        raise ImportError("cannot import name 'estimate_cost' from teacup_agent.model")

    monkeypatch.setattr(openai, "OpenAI", boom)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("TEACUP_AGENT_SEARCH", "hosted")

    out = tools.search_web("q")
    assert "not configured" not in out
    assert "ImportError" in out  # the real failure reaches the model, named


def test_spend_is_attributed_to_the_tool_that_incurred_it(monkeypatch):
    """Reverting `tool=calls[i].name` to a hardcoded "search_web" left the whole suite
    green, because search_web is the only tool that spends — so this registers a second
    one. The index matters too: `to_run` holds `(i, call)` pairs and skipped calls never
    enter it, so a denied call ahead of a spending one must not shift the attribution.
    """
    from teacup_agent import loop, tools as tools_mod
    from teacup_agent.memory import NullMemory
    from teacup_agent.model import ScriptedModel, assistant_calls, assistant_says

    def bill_something(amount: float = 0.02) -> str:
        tools_mod._add_hosted_spend(amount)
        return "billed"

    tools_mod.REGISTRY["bill_something"] = tools_mod.Tool(
        "bill_something", "spends money", {"type": "object", "properties": {}},
        bill_something, False, None, True,
    )
    try:
        events = []
        loop.run(
            "spend after a denied call",
            ScriptedModel(
                [
                    # send_email is gated and denied unattended, so index 0 never runs;
                    # the spending tool at index 1 is the one to attribute
                    assistant_calls(
                        [
                            ("send_email", {"to": "a@b.c", "subject": "s", "body": "b"}),
                            ("bill_something", {}),
                        ]
                    ),
                    assistant_says("done"),
                ]
            ),
            memory=NullMemory(),
            budget=0.5,
            run_dir=None,
            plan=False,
            on_event=lambda name, data: events.append((name, data)),
        )
    finally:
        tools_mod.REGISTRY.pop("bill_something", None)

    spend = [d for n, d in events if n == "tool_spend"]
    assert spend, "the tool's spend was never charged"
    assert all(d["tool"] == "bill_something" for d in spend), spend
    assert sum(d["cost"] for d in spend) == pytest.approx(0.02)
