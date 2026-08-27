"""Unit tests always search offline: deterministic, network-free and fast."""

import pytest


@pytest.fixture(autouse=True)
def _offline_search(monkeypatch):
    monkeypatch.setenv("TEACUP_AGENT_SEARCH", "offline")
