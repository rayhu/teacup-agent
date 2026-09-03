"""delegate_a2a, tested against a real a2a-sdk server over httpx.ASGITransport — real
protocol, zero real sockets. The fixture is tests/fixtures/demo_a2a_server.py.
"""

import pathlib
import sys

import httpx
import pytest

from teacup_agent import tools
from teacup_agent.a2a.client import A2AHub
from teacup_agent.agent_config import A2APeer

sys.path.insert(0, str(pathlib.Path(__file__).parent / "fixtures"))
from demo_a2a_server import build_app  # noqa: E402


def _hub(peers=None):
    transport = httpx.ASGITransport(app=build_app())
    h = A2AHub(transport=transport)
    h.register(peers or {"demo": A2APeer(url="http://test")})
    return h


@pytest.fixture
def hub():
    h = _hub()
    yield h
    h.close()


# --- registration ---------------------------------------------------------------


def test_delegate_a2a_lands_in_the_registry_gated(hub):
    assert "delegate_a2a" in tools.REGISTRY
    assert tools.REGISTRY["delegate_a2a"].requires_approval is True


def test_close_removes_the_tool():
    h = _hub()
    assert "delegate_a2a" in tools.REGISTRY
    h.close()
    assert "delegate_a2a" not in tools.REGISTRY


def test_register_fails_fast_on_a_missing_api_key_env(monkeypatch):
    monkeypatch.delenv("MISSING_A2A_TOKEN", raising=False)
    h = A2AHub()
    try:
        with pytest.raises(RuntimeError, match="MISSING_A2A_TOKEN"):
            h.register(
                {"demo": A2APeer(url="http://example.com", api_key_env="MISSING_A2A_TOKEN")}
            )
    finally:
        h.close()


# --- calling ----------------------------------------------------------------------


def test_a_successful_task_comes_back_as_the_answer(hub):
    result = tools.execute("delegate_a2a", '{"peer": "demo", "task": "hello there"}')
    assert result == "echo: hello there"


def test_a_failed_remote_task_becomes_an_ERROR_string(hub):
    result = tools.execute("delegate_a2a", '{"peer": "demo", "task": "please fail this"}')
    assert result.startswith("ERROR:") and "TASK_STATE_FAILED" in result


def test_unknown_peer_is_self_correctable(hub):
    result = tools.execute("delegate_a2a", '{"peer": "nope", "task": "x"}')
    assert result.startswith("ERROR:") and "nope" in result and "demo" in result


def test_a_second_call_to_the_same_peer_reuses_the_cached_client(hub):
    """Connecting is lazy and cached — this pins that a second call does not error out
    from trying to rebuild a client, and that both calls still work."""
    first = tools.execute("delegate_a2a", '{"peer": "demo", "task": "one"}')
    second = tools.execute("delegate_a2a", '{"peer": "demo", "task": "two"}')
    assert first == "echo: one"
    assert second == "echo: two"
