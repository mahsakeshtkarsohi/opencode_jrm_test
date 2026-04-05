"""
Unit tests for src/jrm_advisor/genie/client.py.

All tests mock the Databricks SDK — no live API calls.

Acceptance criteria from issue #003:
  - GenieClient reads env vars; never from arguments
  - client.ask() handles start → poll → retrieve lifecycle
  - GenieResult contains rows, sql, question
  - GenieTimeoutError on polling timeout
  - GenieError on API errors
  - SQL logged at DEBUG only (not INFO)
  - Unit tests mock HTTP layer — no live calls in CI
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from jrm_advisor.genie.client import (
    GenieClient,
    GenieError,
    GenieNoResultError,
    GenieResult,
    GenieTimeoutError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _env_vars(monkeypatch):
    """Set required env vars for every test."""
    monkeypatch.setenv("DATABRICKS_HOST", "https://test.azuredatabricks.net")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapiTEST")
    monkeypatch.setenv("GENIE_SPACE_ID", "test-space-id")


def _make_genie_message(
    *,
    sql: str = "SELECT year_week, sales_uplift_percentage FROM campaign",
    statement_id: str = "stmt-001",
    conversation_id: str = "conv-001",
    message_id: str = "msg-001",
    error=None,
    text_only: bool = False,
):
    """Build a mock GenieMessage with a query attachment."""
    from databricks.sdk.service.dashboards import (
        GenieAttachment,
        GenieMessage,
        GenieQueryAttachment,
        TextAttachment,
    )

    if text_only:
        attachment = GenieAttachment(
            attachment_id="att-001",
            text=TextAttachment(content="Coca-Cola campaign ran in Q1 2025."),
        )
    else:
        query_attachment = GenieQueryAttachment(
            query=sql,
            statement_id=statement_id,
        )
        attachment = GenieAttachment(
            attachment_id="att-001",
            query=query_attachment,
        )

    return GenieMessage(
        id=message_id,
        message_id=message_id,
        space_id="test-space-id",
        conversation_id=conversation_id,
        content="Show weekly sales uplift for the Coca-Cola campaign",
        attachments=[attachment],
        error=error,
    )


def _make_query_result_response(
    *,
    columns: list[str],
    rows: list[list[str]],
):
    """Build a mock GenieGetMessageQueryResultResponse with inline data."""
    from databricks.sdk.service import sql
    from databricks.sdk.service.dashboards import GenieGetMessageQueryResultResponse

    col_infos = [sql.ColumnInfo(name=c) for c in columns]
    schema = sql.ResultSchema(columns=col_infos)
    manifest = sql.ResultManifest(schema=schema)
    result_data = sql.ResultData(data_array=rows)
    statement_response = sql.StatementResponse(
        manifest=manifest,
        result=result_data,
        statement_id="stmt-001",
    )
    return GenieGetMessageQueryResultResponse(statement_response=statement_response)


@pytest.fixture()
def mock_ws():
    """Patch WorkspaceClient so no HTTP calls are made."""
    with patch("jrm_advisor.genie.client.WorkspaceClient") as MockWS:
        ws_instance = MagicMock()
        MockWS.return_value = ws_instance
        yield ws_instance


# ---------------------------------------------------------------------------
# AC: GenieClient reads credentials from env vars
# ---------------------------------------------------------------------------


class TestCredentials:
    def test_missing_host_raises(self, monkeypatch):
        monkeypatch.delenv("DATABRICKS_HOST")
        with pytest.raises(ValueError, match="DATABRICKS_HOST"):
            GenieClient()

    def test_missing_token_raises(self, monkeypatch):
        monkeypatch.delenv("DATABRICKS_TOKEN")
        with pytest.raises(ValueError, match="DATABRICKS_TOKEN"):
            GenieClient()

    def test_missing_space_id_raises(self, monkeypatch):
        monkeypatch.delenv("GENIE_SPACE_ID")
        with pytest.raises(ValueError, match="GENIE_SPACE_ID"):
            GenieClient()

    def test_all_missing_lists_all_vars(self, monkeypatch):
        monkeypatch.delenv("DATABRICKS_HOST")
        monkeypatch.delenv("DATABRICKS_TOKEN")
        monkeypatch.delenv("GENIE_SPACE_ID")
        with pytest.raises(ValueError) as exc_info:
            GenieClient()
        msg = str(exc_info.value)
        assert "DATABRICKS_HOST" in msg
        assert "DATABRICKS_TOKEN" in msg
        assert "GENIE_SPACE_ID" in msg

    def test_workspace_client_constructed_with_env_values(self, mock_ws):
        with patch("jrm_advisor.genie.client.WorkspaceClient") as MockWS:
            MockWS.return_value = MagicMock()
            GenieClient()
            MockWS.assert_called_once_with(
                host="https://test.azuredatabricks.net",
                token="dapiTEST",
            )


# ---------------------------------------------------------------------------
# AC: ask() returns GenieResult with rows, sql, question
# ---------------------------------------------------------------------------


class TestAskSuccess:
    def test_returns_genie_result(self, mock_ws):
        mock_ws.genie.start_conversation_and_wait.return_value = _make_genie_message()
        mock_ws.genie.get_message_query_result.return_value = (
            _make_query_result_response(
                columns=["year_week", "sales_uplift_percentage"],
                rows=[["202501", "0.12"], ["202502", "0.18"]],
            )
        )

        client = GenieClient()
        result = client.ask("Show weekly sales uplift for the Coca-Cola campaign")

        assert isinstance(result, GenieResult)

    def test_result_contains_question(self, mock_ws):
        mock_ws.genie.start_conversation_and_wait.return_value = _make_genie_message()
        mock_ws.genie.get_message_query_result.return_value = (
            _make_query_result_response(
                columns=["year_week", "sales_uplift_percentage"],
                rows=[["202501", "0.12"]],
            )
        )

        client = GenieClient()
        result = client.ask("Show weekly sales uplift for the Coca-Cola campaign")

        assert result.question == "Show weekly sales uplift for the Coca-Cola campaign"

    def test_result_contains_sql(self, mock_ws):
        expected_sql = "SELECT year_week, sales_uplift_percentage FROM campaign"
        mock_ws.genie.start_conversation_and_wait.return_value = _make_genie_message(
            sql=expected_sql
        )
        mock_ws.genie.get_message_query_result.return_value = (
            _make_query_result_response(
                columns=["year_week", "sales_uplift_percentage"],
                rows=[["202501", "0.12"]],
            )
        )

        client = GenieClient()
        result = client.ask("any question")

        assert result.sql == expected_sql

    def test_result_rows_mapped_to_column_names(self, mock_ws):
        mock_ws.genie.start_conversation_and_wait.return_value = _make_genie_message()
        mock_ws.genie.get_message_query_result.return_value = (
            _make_query_result_response(
                columns=["year_week", "sales_uplift_percentage"],
                rows=[["202501", "0.12"], ["202502", "0.18"]],
            )
        )

        client = GenieClient()
        result = client.ask("any question")

        assert len(result.rows) == 2
        assert result.rows[0] == {
            "year_week": "202501",
            "sales_uplift_percentage": "0.12",
        }
        assert result.rows[1] == {
            "year_week": "202502",
            "sales_uplift_percentage": "0.18",
        }

    def test_empty_data_array_returns_empty_rows(self, mock_ws):
        mock_ws.genie.start_conversation_and_wait.return_value = _make_genie_message()
        mock_ws.genie.get_message_query_result.return_value = (
            _make_query_result_response(
                columns=["year_week", "sales_uplift_percentage"],
                rows=[],
            )
        )

        client = GenieClient()
        result = client.ask("any question")

        assert result.rows == []

    def test_correct_space_id_passed_to_start_conversation(self, mock_ws):
        mock_ws.genie.start_conversation_and_wait.return_value = _make_genie_message()
        mock_ws.genie.get_message_query_result.return_value = (
            _make_query_result_response(
                columns=["year_week", "actual_sales"],
                rows=[["202501", "5000.0"]],
            )
        )

        client = GenieClient()
        client.ask("any question")

        mock_ws.genie.start_conversation_and_wait.assert_called_once_with(
            space_id="test-space-id",
            content="any question",
        )


# ---------------------------------------------------------------------------
# AC: GenieTimeoutError raised on timeout
# ---------------------------------------------------------------------------


class TestTimeout:
    def test_timeout_raises_genie_timeout_error(self, mock_ws):
        mock_ws.genie.start_conversation_and_wait.side_effect = TimeoutError(
            "timed out"
        )

        client = GenieClient()
        with pytest.raises(GenieTimeoutError):
            client.ask("any question")

    def test_timeout_error_message_contains_question(self, mock_ws):
        mock_ws.genie.start_conversation_and_wait.side_effect = TimeoutError(
            "timed out"
        )

        client = GenieClient()
        with pytest.raises(GenieTimeoutError, match="any question"):
            client.ask("any question")


# ---------------------------------------------------------------------------
# AC: GenieError raised on API error
# ---------------------------------------------------------------------------


class TestApiError:
    def test_api_exception_raises_genie_error(self, mock_ws):
        mock_ws.genie.start_conversation_and_wait.side_effect = RuntimeError(
            "500 Internal Server Error"
        )

        client = GenieClient()
        with pytest.raises(GenieError):
            client.ask("any question")

    def test_message_error_field_raises_genie_error(self, mock_ws):
        error_message = _make_genie_message()
        error_message.error = MagicMock()  # non-None error field

        mock_ws.genie.start_conversation_and_wait.return_value = error_message

        client = GenieClient()
        with pytest.raises(GenieError):
            client.ask("any question")

    def test_fetch_rows_exception_raises_genie_error(self, mock_ws):
        mock_ws.genie.start_conversation_and_wait.return_value = _make_genie_message()
        mock_ws.genie.get_message_query_result.side_effect = RuntimeError(
            "network error"
        )

        client = GenieClient()
        with pytest.raises(GenieError, match="stmt-001"):
            client.ask("any question")


# ---------------------------------------------------------------------------
# AC: GenieNoResultError when Genie returns text only (no data attachment)
# ---------------------------------------------------------------------------


class TestNoResult:
    def test_text_only_response_raises_genie_no_result_error(self, mock_ws):
        mock_ws.genie.start_conversation_and_wait.return_value = _make_genie_message(
            text_only=True
        )

        client = GenieClient()
        with pytest.raises(GenieNoResultError):
            client.ask("What is sales uplift?")

    def test_no_result_error_message_contains_question(self, mock_ws):
        mock_ws.genie.start_conversation_and_wait.return_value = _make_genie_message(
            text_only=True
        )

        client = GenieClient()
        with pytest.raises(GenieNoResultError, match="What is sales uplift"):
            client.ask("What is sales uplift?")

    def test_no_result_error_includes_genie_text_preview(self, mock_ws):
        mock_ws.genie.start_conversation_and_wait.return_value = _make_genie_message(
            text_only=True
        )

        client = GenieClient()
        with pytest.raises(GenieNoResultError, match="Coca-Cola campaign ran in Q1"):
            client.ask("any question")


# ---------------------------------------------------------------------------
# AC: SQL is not logged at INFO level (must stay at DEBUG)
# ---------------------------------------------------------------------------


class TestLogging:
    def test_sql_not_logged_at_info(self, mock_ws, caplog):
        mock_ws.genie.start_conversation_and_wait.return_value = _make_genie_message(
            sql="SELECT secret_column FROM sensitive_table"
        )
        mock_ws.genie.get_message_query_result.return_value = (
            _make_query_result_response(
                columns=["year_week", "sales_uplift_percentage"],
                rows=[["202501", "0.12"]],
            )
        )

        client = GenieClient()
        with caplog.at_level(logging.INFO, logger="jrm_advisor.genie.client"):
            client.ask("any question")

        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert not any("SELECT" in m for m in info_messages), (
            "SQL must not appear in INFO-level log messages"
        )

    def test_rows_not_logged_at_any_level(self, mock_ws, caplog):
        mock_ws.genie.start_conversation_and_wait.return_value = _make_genie_message()
        mock_ws.genie.get_message_query_result.return_value = (
            _make_query_result_response(
                columns=["year_week", "sales_uplift_percentage"],
                rows=[["202501", "0.99"]],
            )
        )

        client = GenieClient()
        with caplog.at_level(logging.DEBUG, logger="jrm_advisor.genie.client"):
            client.ask("any question")

        all_messages = " ".join(r.message for r in caplog.records)
        assert "0.99" not in all_messages, "Row values must never appear in log output"
