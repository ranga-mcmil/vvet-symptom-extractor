"""Tests for llm.py. We replace the Groq client with a fake, so no network or key."""

from types import SimpleNamespace

import pytest
from groq import GroqError

from app import llm
from app.errors import LLMError


def fake_client(content=None, error=None):
    """Build an object that quacks like groq.Groq() for our one call."""

    def create(**kwargs):
        if error:
            raise error
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    completions = SimpleNamespace(create=create)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def use_client(monkeypatch, client):
    # monkeypatch temporarily replaces the real client factory for one test.
    monkeypatch.setattr(llm, "_get_client", lambda: client)


def test_returns_parsed_json(monkeypatch):
    use_client(monkeypatch, fake_client(content='{"findings": []}'))
    assert llm.extract_raw("hello") == {"findings": []}


def test_invalid_json_raises(monkeypatch):
    use_client(monkeypatch, fake_client(content="not json"))
    with pytest.raises(LLMError):
        llm.extract_raw("hello")


def test_json_that_is_not_an_object_raises(monkeypatch):
    use_client(monkeypatch, fake_client(content="[1, 2]"))
    with pytest.raises(LLMError):
        llm.extract_raw("hello")


def test_provider_error_becomes_llm_error(monkeypatch):
    use_client(monkeypatch, fake_client(error=GroqError("boom")))
    with pytest.raises(LLMError):
        llm.extract_raw("hello")


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    llm._get_client.cache_clear()
    with pytest.raises(LLMError):
        llm.extract_raw("hello")