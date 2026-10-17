"""Tests for the OpenAI-compatible client used by the real-model confirmation study.

A confirmation run issues several thousand calls unattended against a paid endpoint, so the
behaviour that matters is not the happy path: it is that a rate limit does not lose the run,
that a repeated prompt is not paid for twice, and that the key never leaves the environment.
No test here contacts a network endpoint.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from ragsec.core.llm import OpenAICompatibleLLM, ResponseCache


@pytest.fixture
def client(tmp_path, monkeypatch) -> OpenAICompatibleLLM:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    return OpenAICompatibleLLM(
        model="test-model",
        cache_path=tmp_path / "cache.sqlite",
        backoff=1.0,
        max_retries=3,
    )


def _http_error(code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = {"Retry-After": retry_after} if retry_after else {}
    return urllib.error.HTTPError("u", code, "err", headers, None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def test_missing_key_fails_loudly_before_any_call(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAICompatibleLLM(model="test-model")


def test_key_is_read_from_a_configurable_variable(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("MY_PROVIDER_KEY", "abc")
    llm = OpenAICompatibleLLM(model="m", api_key_env="MY_PROVIDER_KEY")
    assert llm.api_key == "abc"


def test_cache_file_never_contains_the_key(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    path = tmp_path / "c.sqlite"
    llm = OpenAICompatibleLLM(model="m", cache_path=path)
    monkeypatch.setattr(llm, "_request", lambda messages: "answer")
    llm.generate("q", system="s")
    assert "sk-secret-value" not in path.read_bytes().decode("utf-8", "ignore")


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_repeated_prompt_is_served_from_cache(client, monkeypatch) -> None:
    calls = {"n": 0}

    def fake(messages):
        calls["n"] += 1
        return f"answer {calls['n']}"

    monkeypatch.setattr(client, "_request", fake)
    first = client.generate("same prompt", system="sys")
    second = client.generate("same prompt", system="sys")
    assert first == second == "answer 1"
    assert calls["n"] == 1, "the endpoint should be hit once, not twice"
    assert client.cache.hits == 1


def test_cache_key_separates_requests_that_could_differ(client, monkeypatch) -> None:
    monkeypatch.setattr(client, "_request", lambda messages: "x")
    base = ResponseCache.key("m", 0.0, "sys", "prompt")
    assert base != ResponseCache.key("other-model", 0.0, "sys", "prompt")
    assert base != ResponseCache.key("m", 0.7, "sys", "prompt")
    assert base != ResponseCache.key("m", 0.0, "other system", "prompt")
    assert base != ResponseCache.key("m", 0.0, "sys", "other prompt")


def test_cache_survives_a_new_client_over_the_same_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    path = tmp_path / "c.sqlite"

    first = OpenAICompatibleLLM(model="m", cache_path=path)
    monkeypatch.setattr(first, "_request", lambda messages: "remembered")
    first.generate("q")

    second = OpenAICompatibleLLM(model="m", cache_path=path)
    monkeypatch.setattr(
        second, "_request", lambda messages: pytest.fail("should not call the endpoint")
    )
    assert second.generate("q") == "remembered", "an interrupted run must resume for free"


# ---------------------------------------------------------------------------
# Retries
# ---------------------------------------------------------------------------


def test_rate_limit_is_retried_then_succeeds(client, monkeypatch) -> None:
    attempts = {"n": 0}

    def flaky(messages):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise _http_error(429)
        return "eventually"

    monkeypatch.setattr(client, "_request", flaky)
    monkeypatch.setattr("ragsec.core.llm.time.sleep", lambda s: None)
    assert client.generate("q") == "eventually"
    assert client.retries == 2


def test_transient_server_error_is_retried(client, monkeypatch) -> None:
    attempts = {"n": 0}

    def flaky(messages):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _http_error(503)
        return "ok"

    monkeypatch.setattr(client, "_request", flaky)
    monkeypatch.setattr("ragsec.core.llm.time.sleep", lambda s: None)
    assert client.generate("q") == "ok"


def test_client_error_is_not_retried(client, monkeypatch) -> None:
    attempts = {"n": 0}

    def bad_request(messages):
        attempts["n"] += 1
        raise _http_error(400)

    monkeypatch.setattr(client, "_request", bad_request)
    monkeypatch.setattr("ragsec.core.llm.time.sleep", lambda s: None)
    with pytest.raises(urllib.error.HTTPError):
        client.generate("q")
    assert attempts["n"] == 1, "a malformed request will not fix itself"


def test_retries_are_bounded(client, monkeypatch) -> None:
    attempts = {"n": 0}

    def always_limited(messages):
        attempts["n"] += 1
        raise _http_error(429)

    monkeypatch.setattr(client, "_request", always_limited)
    monkeypatch.setattr("ragsec.core.llm.time.sleep", lambda s: None)
    with pytest.raises(urllib.error.HTTPError):
        client.generate("q")
    assert attempts["n"] == client.max_retries + 1


def test_retry_after_header_is_honoured(client, monkeypatch) -> None:
    slept: list[float] = []
    attempts = {"n": 0}

    def flaky(messages):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _http_error(429, retry_after="7")
        return "ok"

    monkeypatch.setattr(client, "_request", flaky)
    monkeypatch.setattr("ragsec.core.llm.time.sleep", slept.append)
    client.generate("q")
    assert slept == [7.0], "the server's own backoff should win over ours"


def test_malformed_response_is_retried(client, monkeypatch) -> None:
    attempts = {"n": 0}

    def flaky(messages):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise json.JSONDecodeError("bad", "", 0)
        return "ok"

    monkeypatch.setattr(client, "_request", flaky)
    monkeypatch.setattr("ragsec.core.llm.time.sleep", lambda s: None)
    assert client.generate("q") == "ok"


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


def test_system_prompt_precedes_the_user_turn(client, monkeypatch) -> None:
    captured: list[list[dict[str, str]]] = []
    monkeypatch.setattr(client, "_request", lambda m: captured.append(m) or "ok")
    client.generate("the question", system="the system prompt")
    assert [m["role"] for m in captured[0]] == ["system", "user"]
    assert captured[0][0]["content"] == "the system prompt"
    assert captured[0][1]["content"] == "the question"


def test_call_counter_tracks_billable_calls_only(client, monkeypatch) -> None:
    monkeypatch.setattr(client, "_request", lambda m: "ok")
    client.generate("a")
    client.generate("a")
    client.generate("b")
    assert client.calls == 2, "a cache hit is not a billable call"
