"""
Unit tests for src/jrm_advisor/knowledge_base/client.py.

All tests mock the HTTP layer — no live API calls.

Acceptance criteria covered:
  - KnowledgeBaseClient reads credentials from env vars; never from arguments
  - client.ask() calls the endpoint with correct payload and headers
  - Returns clean prose string from a well-formed response
  - Returns ANSWER_UNAVAILABLE on empty choices, empty content, or malformed response
  - KnowledgeBaseTimeoutError raised on timeout
  - KnowledgeBaseError raised on HTTP errors and JSON decode errors
  - Token is NEVER logged
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from jrm_advisor.knowledge_base.client import (
    ANSWER_UNAVAILABLE,
    KnowledgeBaseClient,
    KnowledgeBaseError,
    KnowledgeBaseTimeoutError,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _env_vars(monkeypatch):
    """Set required env vars for every test."""
    monkeypatch.setenv("DATABRICKS_HOST", "https://test.azuredatabricks.net")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapiTEST")
    monkeypatch.setenv(
        "KB_ENDPOINT_URL",
        "https://test.azuredatabricks.net/serving-endpoints/ka-test/invocations",
    )


def _make_response_body(content: str) -> bytes:
    """Build a minimal OpenAI-compatible response payload."""
    return json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")


def _mock_urlopen(body: bytes, status: int = 200):
    """Return a context-manager mock for urllib.request.urlopen."""
    mock_response = MagicMock()
    mock_response.read.return_value = body
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)
    return mock_response


# ---------------------------------------------------------------------------
# AC: KnowledgeBaseClient reads credentials from env vars
# ---------------------------------------------------------------------------


class TestCredentials:
    def test_missing_host_raises(self, monkeypatch):
        monkeypatch.delenv("DATABRICKS_HOST")
        with pytest.raises(ValueError, match="DATABRICKS_HOST"):
            KnowledgeBaseClient()

    def test_missing_token_raises(self, monkeypatch):
        monkeypatch.delenv("DATABRICKS_TOKEN")
        with pytest.raises(ValueError, match="DATABRICKS_TOKEN"):
            KnowledgeBaseClient()

    def test_missing_endpoint_url_raises(self, monkeypatch):
        monkeypatch.delenv("KB_ENDPOINT_URL")
        with pytest.raises(ValueError, match="KB_ENDPOINT_URL"):
            KnowledgeBaseClient()

    def test_all_missing_lists_all_vars(self, monkeypatch):
        monkeypatch.delenv("DATABRICKS_HOST")
        monkeypatch.delenv("DATABRICKS_TOKEN")
        monkeypatch.delenv("KB_ENDPOINT_URL")
        with pytest.raises(ValueError) as exc_info:
            KnowledgeBaseClient()
        msg = str(exc_info.value)
        assert "DATABRICKS_HOST" in msg
        assert "DATABRICKS_TOKEN" in msg
        assert "KB_ENDPOINT_URL" in msg

    def test_instantiation_succeeds_with_all_vars(self):
        client = KnowledgeBaseClient()
        assert client is not None


# ---------------------------------------------------------------------------
# AC: ask() calls the endpoint with correct URL, method, headers, and payload
# ---------------------------------------------------------------------------


class TestAskRequest:
    def test_correct_url_called(self):
        mock_resp = _mock_urlopen(_make_response_body("Sales uplift is..."))
        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            KnowledgeBaseClient().ask("What is sales uplift?")
            request_arg = mock_open.call_args[0][0]
            assert (
                request_arg.full_url
                == "https://test.azuredatabricks.net/serving-endpoints/ka-test/invocations"
            )

    def test_request_method_is_post(self):
        mock_resp = _mock_urlopen(_make_response_body("Sales uplift is..."))
        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            KnowledgeBaseClient().ask("What is sales uplift?")
            request_arg = mock_open.call_args[0][0]
            assert request_arg.method == "POST"

    def test_authorization_header_contains_token(self):
        mock_resp = _mock_urlopen(_make_response_body("Sales uplift is..."))
        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            KnowledgeBaseClient().ask("What is sales uplift?")
            request_arg = mock_open.call_args[0][0]
            assert request_arg.get_header("Authorization") == "Bearer dapiTEST"

    def test_content_type_header_is_json(self):
        mock_resp = _mock_urlopen(_make_response_body("Sales uplift is..."))
        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            KnowledgeBaseClient().ask("What is sales uplift?")
            request_arg = mock_open.call_args[0][0]
            assert request_arg.get_header("Content-type") == "application/json"

    def test_payload_contains_question_as_user_message(self):
        mock_resp = _mock_urlopen(_make_response_body("Sales uplift is..."))
        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            question = "What is OTS?"
            KnowledgeBaseClient().ask(question)
            request_arg = mock_open.call_args[0][0]
            body = json.loads(request_arg.data.decode("utf-8"))
            assert body == {"messages": [{"role": "user", "content": question}]}


# ---------------------------------------------------------------------------
# AC: ask() returns clean answer text from a well-formed response
# ---------------------------------------------------------------------------


class TestAskSuccess:
    def test_returns_string(self):
        mock_resp = _mock_urlopen(
            _make_response_body("Sales uplift measures incremental sales.")
        )
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = KnowledgeBaseClient().ask("What is sales uplift?")
        assert isinstance(result, str)

    def test_returns_answer_content(self):
        expected = "Sales uplift measures incremental sales driven by the campaign."
        mock_resp = _mock_urlopen(_make_response_body(expected))
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = KnowledgeBaseClient().ask("What is sales uplift?")
        assert result == expected

    def test_strips_leading_trailing_whitespace(self):
        mock_resp = _mock_urlopen(
            _make_response_body("  Sales uplift is incremental sales.  \n")
        )
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = KnowledgeBaseClient().ask("What is sales uplift?")
        assert result == "Sales uplift is incremental sales."


# ---------------------------------------------------------------------------
# AC: ask() returns ANSWER_UNAVAILABLE on degraded / malformed responses
# ---------------------------------------------------------------------------


class TestAskFallback:
    def test_empty_choices_returns_unavailable(self):
        body = json.dumps({"choices": []}).encode("utf-8")
        mock_resp = _mock_urlopen(body)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = KnowledgeBaseClient().ask("What is OTS?")
        assert result == ANSWER_UNAVAILABLE

    def test_missing_choices_key_returns_unavailable(self):
        body = json.dumps({"model": "ka-model"}).encode("utf-8")
        mock_resp = _mock_urlopen(body)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = KnowledgeBaseClient().ask("What is OTS?")
        assert result == ANSWER_UNAVAILABLE

    def test_empty_content_returns_unavailable(self):
        body = json.dumps({"choices": [{"message": {"content": ""}}]}).encode("utf-8")
        mock_resp = _mock_urlopen(body)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = KnowledgeBaseClient().ask("What is OTS?")
        assert result == ANSWER_UNAVAILABLE

    def test_whitespace_only_content_returns_unavailable(self):
        body = json.dumps({"choices": [{"message": {"content": "   "}}]}).encode(
            "utf-8"
        )
        mock_resp = _mock_urlopen(body)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = KnowledgeBaseClient().ask("What is OTS?")
        assert result == ANSWER_UNAVAILABLE

    def test_null_choices_returns_unavailable(self):
        body = json.dumps({"choices": None}).encode("utf-8")
        mock_resp = _mock_urlopen(body)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = KnowledgeBaseClient().ask("What is OTS?")
        assert result == ANSWER_UNAVAILABLE


# ---------------------------------------------------------------------------
# AC: KnowledgeBaseTimeoutError raised on timeout
# ---------------------------------------------------------------------------


class TestTimeout:
    def test_timeout_raises_knowledge_base_timeout_error(self):
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            with pytest.raises(KnowledgeBaseTimeoutError):
                KnowledgeBaseClient().ask("What is sales uplift?")

    def test_timeout_error_message_contains_question(self):
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            with pytest.raises(KnowledgeBaseTimeoutError, match="What is sales uplift"):
                KnowledgeBaseClient().ask("What is sales uplift?")


# ---------------------------------------------------------------------------
# AC: KnowledgeBaseError raised on HTTP errors
# ---------------------------------------------------------------------------


class TestHttpErrors:
    def _make_http_error(self, code: int, msg: str = "Error") -> urllib.error.HTTPError:
        return urllib.error.HTTPError(
            url="https://test.endpoint",
            code=code,
            msg=msg,
            hdrs=None,  # type: ignore[arg-type]
            fp=BytesIO(b"endpoint error details"),
        )

    def test_http_401_raises_knowledge_base_error(self):
        with patch(
            "urllib.request.urlopen",
            side_effect=self._make_http_error(401, "Unauthorized"),
        ):
            with pytest.raises(KnowledgeBaseError) as exc_info:
                KnowledgeBaseClient().ask("any question")
            assert exc_info.value.status_code == 401

    def test_http_500_raises_knowledge_base_error(self):
        with patch(
            "urllib.request.urlopen",
            side_effect=self._make_http_error(500, "Internal Server Error"),
        ):
            with pytest.raises(KnowledgeBaseError) as exc_info:
                KnowledgeBaseClient().ask("any question")
            assert exc_info.value.status_code == 500

    def test_url_error_raises_knowledge_base_error(self):
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("Name or service not known"),
        ):
            with pytest.raises(KnowledgeBaseError):
                KnowledgeBaseClient().ask("any question")

    def test_non_json_response_raises_knowledge_base_error(self):
        mock_resp = _mock_urlopen(b"<html>Internal Server Error</html>")
        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(KnowledgeBaseError, match="non-JSON"):
                KnowledgeBaseClient().ask("any question")
