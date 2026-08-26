import json

import pytest

from mini_agent import tools


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


# --- search_web 的三种模式与「宁缺毋滥」原则 --------------------------------


def test_offline_search_does_not_false_positive(monkeypatch):
    """回归测试：'OpenAI strategy' 曾因 any() 匹配命中 NVIDIA 那条语料。"""
    monkeypatch.setenv("MINI_AGENT_SEARCH", "offline")
    out = tools.execute("search_web", '{"query": "OpenAI strategy 2026 roadmap"}')
    assert "NVIDIA" not in out
    assert "没有找到" in out


def test_offline_search_still_matches_full_key(monkeypatch):
    monkeypatch.setenv("MINI_AGENT_SEARCH", "offline")
    assert "NVIDIA" in tools.execute("search_web", '{"query": "nvidia gpu strategy 是什么"}')


def test_auto_mode_falls_back_to_corpus_when_network_fails(monkeypatch):
    monkeypatch.setenv("MINI_AGENT_SEARCH", "auto")
    monkeypatch.setattr(tools, "_search_web_backend", lambda q, n: (_ for _ in ()).throw(OSError("no net")))
    out = tools.execute("search_web", '{"query": "cuda"}')
    assert out.startswith("[联网检索失败") and "CUDA" in out


def test_web_mode_reports_error_instead_of_pretending(monkeypatch):
    """严格模式下检索失败必须是 ERROR，不能让模型以为「查过了，不存在」。"""
    monkeypatch.setenv("MINI_AGENT_SEARCH", "web")
    monkeypatch.setattr(tools, "_search_web_backend", lambda q, n: (_ for _ in ()).throw(OSError("no net")))
    assert tools.execute("search_web", '{"query": "cuda"}').startswith("ERROR:")


def test_web_search_formats_results_with_links(monkeypatch):
    monkeypatch.setenv("MINI_AGENT_SEARCH", "web")
    monkeypatch.setattr(
        tools,
        "DDGS" if hasattr(tools, "DDGS") else "_search_web_backend",
        lambda q, n: f"1. 标题\n   https://example.com\n   摘要（n={n}）",
    )
    out = tools.execute("search_web", '{"query": "x", "max_results": 99}')
    assert "https://example.com" in out and "n=10" in out  # 条数被夹到上限 10


# --- 检索失败：退避重试，且不能伪装成「查无此事」 ---------------------------


class _FlakyDDGS:
    """前 fail_times 次抛异常，之后返回正常结果。"""

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
    monkeypatch.setattr(tools.time, "sleep", lambda *_: None)  # 测试不真的等
    monkeypatch.setattr(tools, "_last_search_at", 0.0)

    def install(fail_times=0):
        monkeypatch.setattr(ddgs, "DDGS", lambda: _FlakyDDGS(fail_times))
        return _FlakyDDGS

    return install


def test_search_retries_then_succeeds(monkeypatch, fake_ddgs):
    monkeypatch.setenv("MINI_AGENT_SEARCH", "web")
    flaky = fake_ddgs(fail_times=2)
    out = tools.execute("search_web", '{"query": "anything"}')
    assert "https://e.com" in out
    assert flaky.calls == 3  # 失败两次后第三次成功


def test_search_failure_is_not_disguised_as_no_results(monkeypatch, fake_ddgs):
    """本地语料没有的题目，检索挂了必须报 ERROR，不能说「没有找到」。"""
    monkeypatch.setenv("MINI_AGENT_SEARCH", "auto")
    flaky = fake_ddgs(fail_times=99)
    out = tools.execute("search_web", '{"query": "Anthropic 最近半年融资"}')
    assert out.startswith("ERROR:") and "不代表" in out
    assert "没有找到" not in out
    assert flaky.calls == tools._RETRIES  # 重试次数用满
