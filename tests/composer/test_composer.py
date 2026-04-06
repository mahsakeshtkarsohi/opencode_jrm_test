"""
Unit tests for src/jrm_advisor/composer/composer.py

All HTTP calls are mocked — no live endpoint calls.
"""

from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, patch


from jrm_advisor.composer.composer import (
    AnswerComposer,
    MAX_ROWS_IN_PROMPT,
    _fallback_compose,
)
from jrm_advisor.supervisor.agent import Intent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_composer(
    endpoint: str = "https://test.endpoint/invocations",
) -> AnswerComposer:
    return AnswerComposer(endpoint_url=endpoint, token="dapiTEST")


def _mock_response(content: str) -> MagicMock:
    """Build a fake urlopen response returning an OpenAI-compatible envelope."""
    body = json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")
    response = MagicMock()
    response.read.return_value = body
    response.__enter__ = lambda s: s
    response.__exit__ = MagicMock(return_value=False)
    return response


# ---------------------------------------------------------------------------
# _fallback_compose (pure function)
# ---------------------------------------------------------------------------


class TestFallbackCompose:
    def test_kb_only_returns_kb_answer(self):
        result = _fallback_compose(
            intent=Intent.KB_ONLY,
            kb_answer="Uplift is the incremental revenue.",
            genie_rows=None,
            genie_sql="",
        )
        assert "incremental revenue" in result

    def test_data_only_with_rows_returns_table(self):
        result = _fallback_compose(
            intent=Intent.DATA_ONLY,
            kb_answer=None,
            genie_rows=[{"year_week": "202501", "sales_uplift_percentage": "0.12"}],
            genie_sql="",
        )
        assert "202501" in result

    def test_data_only_no_rows_returns_no_data_message(self):
        result = _fallback_compose(
            intent=Intent.DATA_ONLY,
            kb_answer=None,
            genie_rows=None,
            genie_sql="",
        )
        assert "no data" in result.lower() or "no results" in result.lower()

    def test_hybrid_includes_kb_and_data(self):
        result = _fallback_compose(
            intent=Intent.HYBRID,
            kb_answer="Uplift is incremental.",
            genie_rows=[{"year_week": "202501", "sales_uplift_percentage": "0.12"}],
            genie_sql="",
        )
        assert "incremental" in result
        assert "202501" in result


# ---------------------------------------------------------------------------
# AnswerComposer — construction
# ---------------------------------------------------------------------------


class TestAnswerComposerConstruction:
    def test_no_endpoint_uses_fallback_only(self):
        composer = AnswerComposer(endpoint_url="", token="tok")
        result = composer.compose(
            question="What is uplift?",
            intent=Intent.KB_ONLY,
            kb_answer="Uplift is incremental.",
            genie_rows=None,
            genie_sql="",
        )
        assert "incremental" in result

    def test_no_token_uses_fallback_only(self):
        composer = AnswerComposer(endpoint_url="https://ep.example/inv", token="")
        result = composer.compose(
            question="What is uplift?",
            intent=Intent.KB_ONLY,
            kb_answer="Uplift is incremental.",
            genie_rows=None,
            genie_sql="",
        )
        assert "incremental" in result

    def test_reads_env_vars(self, monkeypatch):
        monkeypatch.setenv("COMPOSER_ENDPOINT", "https://ep.from.env/inv")
        monkeypatch.setenv("DATABRICKS_TOKEN", "envtoken")
        composer = AnswerComposer()
        assert composer._endpoint_url == "https://ep.from.env/inv"
        assert composer._token == "envtoken"


# ---------------------------------------------------------------------------
# AnswerComposer — successful LLM call
# ---------------------------------------------------------------------------


class TestAnswerComposerSuccess:
    def test_returns_llm_answer(self):
        composer = _make_composer()
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value = _mock_response(
                "Sales uplift for Heineken was strong in weeks 4–6."
            )
            result = composer.compose(
                question="How did Heineken perform?",
                intent=Intent.DATA_ONLY,
                kb_answer=None,
                genie_rows=[{"year_week": "202504", "actual_sales": "12000"}],
                genie_sql="SELECT ...",
            )
        assert "Heineken" in result or "weeks" in result

    def test_llm_answer_string_returned(self):
        composer = _make_composer()
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value = _mock_response("Specific business insight here.")
            result = composer.compose(
                question="Show results",
                intent=Intent.DATA_ONLY,
                kb_answer=None,
                genie_rows=[{"year_week": "202501", "actual_sales": "5000"}],
                genie_sql="",
            )
        assert isinstance(result, str)
        assert result  # non-empty

    def test_rows_capped_at_max_in_prompt(self):
        """Verify the user message only injects up to MAX_ROWS_IN_PROMPT rows."""
        composer = _make_composer()
        rows = [{"week": str(i), "sales": str(i * 100)} for i in range(50)]
        captured_payload: list[bytes] = []

        original_request = __import__("urllib.request", fromlist=["Request"]).Request

        def capture_request(url, data, headers, method):
            captured_payload.append(data)
            return original_request(url, data, headers, method)

        with patch("urllib.request.Request", side_effect=capture_request):
            with patch("urllib.request.urlopen") as mock_open:
                mock_open.return_value = _mock_response("Summary of 50 rows.")
                composer.compose(
                    question="Show all results",
                    intent=Intent.DATA_ONLY,
                    kb_answer=None,
                    genie_rows=rows,
                    genie_sql="",
                )

        assert captured_payload
        body = json.loads(captured_payload[0].decode("utf-8"))
        user_content = body["messages"][1]["content"]
        # The prompt should mention MAX_ROWS_IN_PROMPT, not 50
        assert f"{MAX_ROWS_IN_PROMPT} rows" in user_content
        # And should NOT contain week "21" (which would be beyond the cap)
        # The cap is 20 rows so week index 20 (value "20") should be absent
        # from the injected JSON rows (index 20 is the 21st item)
        injected = user_content
        # Rows 0..19 are present; row 20 is omitted
        assert '"20"' not in injected or "omitted" in injected


# ---------------------------------------------------------------------------
# AnswerComposer — fallback on LLM failure
# ---------------------------------------------------------------------------


class TestAnswerComposerFallback:
    def test_timeout_falls_back_to_rule_based(self):
        composer = _make_composer()
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            result = composer.compose(
                question="What is uplift?",
                intent=Intent.KB_ONLY,
                kb_answer="Uplift is incremental revenue.",
                genie_rows=None,
                genie_sql="",
            )
        assert "incremental revenue" in result

    def test_http_error_falls_back_to_rule_based(self):
        composer = _make_composer()
        err = urllib.error.HTTPError(
            url="https://ep",
            code=500,
            msg="Internal Error",
            hdrs=None,
            fp=BytesIO(b"server error"),  # type: ignore[arg-type]
        )
        with patch("urllib.request.urlopen", side_effect=err):
            result = composer.compose(
                question="What is uplift?",
                intent=Intent.KB_ONLY,
                kb_answer="Uplift is incremental revenue.",
                genie_rows=None,
                genie_sql="",
            )
        assert "incremental revenue" in result

    def test_url_error_falls_back_to_rule_based(self):
        composer = _make_composer()
        err = urllib.error.URLError("unreachable")
        with patch("urllib.request.urlopen", side_effect=err):
            result = composer.compose(
                question="What is uplift?",
                intent=Intent.KB_ONLY,
                kb_answer="Uplift is incremental revenue.",
                genie_rows=None,
                genie_sql="",
            )
        assert "incremental revenue" in result

    def test_empty_llm_response_falls_back(self):
        """Empty content from LLM → fallback to rule-based concatenation."""
        composer = _make_composer()
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value = _mock_response("")
            result = composer.compose(
                question="What is uplift?",
                intent=Intent.KB_ONLY,
                kb_answer="Uplift is incremental revenue.",
                genie_rows=None,
                genie_sql="",
            )
        # Empty LLM answer → _extract_answer returns "" → caller uses fallback
        assert result  # should not be empty


# ---------------------------------------------------------------------------
# AnswerComposer — prompt content
# ---------------------------------------------------------------------------


class TestAnswerComposerPrompt:
    def test_system_prompt_sent(self):
        composer = _make_composer()
        sent_payloads: list[dict] = []

        def capture(req, timeout=None):
            sent_payloads.append(json.loads(req.data.decode("utf-8")))
            return _mock_response("answer")

        with patch("urllib.request.urlopen", side_effect=capture):
            composer.compose(
                question="How did Heineken perform?",
                intent=Intent.DATA_ONLY,
                kb_answer=None,
                genie_rows=[{"year_week": "202501", "actual_sales": "5000"}],
                genie_sql="",
            )

        assert sent_payloads
        messages = sent_payloads[0]["messages"]
        assert messages[0]["role"] == "system"
        assert "JRM" in messages[0]["content"]

    def test_user_message_contains_question(self):
        composer = _make_composer()
        sent_payloads: list[dict] = []

        def capture(req, timeout=None):
            sent_payloads.append(json.loads(req.data.decode("utf-8")))
            return _mock_response("answer")

        with patch("urllib.request.urlopen", side_effect=capture):
            composer.compose(
                question="How did Heineken perform this week?",
                intent=Intent.DATA_ONLY,
                kb_answer=None,
                genie_rows=[{"year_week": "202501", "actual_sales": "5000"}],
                genie_sql="",
            )

        user_content = sent_payloads[0]["messages"][1]["content"]
        assert "Heineken" in user_content

    def test_kb_answer_included_in_prompt_when_present(self):
        composer = _make_composer()
        sent_payloads: list[dict] = []

        def capture(req, timeout=None):
            sent_payloads.append(json.loads(req.data.decode("utf-8")))
            return _mock_response("answer")

        with patch("urllib.request.urlopen", side_effect=capture):
            composer.compose(
                question="What is uplift and how did Heineken perform?",
                intent=Intent.HYBRID,
                kb_answer="Uplift is incremental revenue above baseline.",
                genie_rows=[{"year_week": "202501", "actual_sales": "5000"}],
                genie_sql="",
            )

        user_content = sent_payloads[0]["messages"][1]["content"]
        assert "incremental revenue" in user_content
