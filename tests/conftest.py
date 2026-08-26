"""单测一律走离线检索：确定性、零网络、跑得快。"""

import pytest


@pytest.fixture(autouse=True)
def _offline_search(monkeypatch):
    monkeypatch.setenv("MINI_AGENT_SEARCH", "offline")
