"""
conftest.py — test-suite fixtures.

The agent loads .env via python-dotenv at import time, so OPENAI_API_KEY is
visible to test code. Without a guard, every log_session call would hit the
real API (slow, costs money, flakes on network). This autouse fixture
strips the key for every test so the suite is deterministic and offline.

If you ever DO want to test the live LLM path, mark the test and re-add the
key inside it via monkeypatch.setenv.
"""

import pytest


@pytest.fixture(autouse=True)
def _force_offline_llm(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
