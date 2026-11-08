"""Shared test fixtures."""

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_llm


@pytest.fixture
def make_client():
    """Return a function that builds a TestClient using a given fake LLM."""

    def _make(fake_llm):
        # Replace get_llm for the duration of the test.
        app.dependency_overrides[get_llm] = lambda: fake_llm
        return TestClient(app)

    yield _make
    # Always clean up so one test can't leak its fake into another.
    app.dependency_overrides.clear()