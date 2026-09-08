"""Unit tests always search offline: deterministic, network-free and fast."""

import pytest


@pytest.fixture(autouse=True)
def _offline_search(monkeypatch):
    monkeypatch.setenv("TEACUP_AGENT_SEARCH", "offline")


@pytest.fixture(autouse=True)
def _drain_hosted_spend():
    """Leave no hosted-search spend behind on the main thread.

    In production the accumulator is drained by `_execute_and_bill` in the pool thread
    that spent it, so nothing accumulates on the main thread. A unit test calling
    `search_web` directly does accumulate there, and the leftover made the suite
    order-dependent — `pytest-randomly` turned up one failure that is invisible in file
    order. Measured across the suite: twelve tests leak without this.
    """
    from teacup_agent import tools

    tools.take_hosted_spend()
    yield
    tools.take_hosted_spend()
