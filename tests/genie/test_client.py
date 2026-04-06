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
    """Build a mock GenieGetMessageQueryResultResponse.

    Returns schema/manifest only — row data is fetched via the chunk API
    (statement_execution.get_statement_result_chunk_n), not from this response.
    """
    from databricks.sdk.service import sql
    from databricks.sdk.service.dashboards import GenieGetMessageQueryResultResponse

    col_infos = [sql.ColumnInfo(name=c) for c in columns]
    schema = sql.ResultSchema(columns=col_infos)
    manifest = sql.ResultManifest(schema=schema, total_row_count=len(rows))
    statement_response = sql.StatementResponse(
        manifest=manifest,
        statement_id="stmt-001",
    )
    return GenieGetMessageQueryResultResponse(statement_response=statement_response)


def _make_chunk(rows: list[list[str]], next_chunk_index: int | None = None):
    """Build a mock ResultData chunk returned by get_statement_result_chunk_n."""
    from databricks.sdk.service import sql

    return sql.ResultData(data_array=rows, next_chunk_index=next_chunk_index)


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
        with pytest.raises(ValueError, match="authentication token"):
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
        assert "authentication token" in msg
        assert "GENIE_SPACE_ID" in msg

    def test_invalid_host_raises(self, monkeypatch):
        """SEC-4: a non-Databricks host URL is rejected at construction time."""
        monkeypatch.setenv("DATABRICKS_HOST", "https://evil.example.com")
        with patch("jrm_advisor.genie.client.WorkspaceClient"):
            with pytest.raises(
                ValueError, match="not a recognised Databricks workspace URL"
            ):
                GenieClient()

    def test_malformed_host_raises(self, monkeypatch):
        """SEC-4: a plainly malformed host is rejected."""
        monkeypatch.setenv("DATABRICKS_HOST", "http://insecure.azuredatabricks.net")
        with patch("jrm_advisor.genie.client.WorkspaceClient"):
            with pytest.raises(
                ValueError, match="not a recognised Databricks workspace URL"
            ):
                GenieClient()

    def test_workspace_client_constructed_with_env_values(self, mock_ws):
        with patch("jrm_advisor.genie.client.WorkspaceClient") as MockWS:
            MockWS.return_value = MagicMock()
            GenieClient()
            MockWS.assert_called_once_with(
                host="https://test.azuredatabricks.net",
                token="dapiTEST",
                auth_type="pat",
            )

    def test_workspace_client_uses_pat_auth_type(self, monkeypatch):
        """auth_type='pat' must be passed so that ambient DATABRICKS_CLIENT_ID /
        DATABRICKS_CLIENT_SECRET env vars injected by the Databricks App runtime
        do not trigger a 'more than one authorization method configured' error."""
        monkeypatch.setenv("DATABRICKS_CLIENT_ID", "fake-client-id")
        monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", "fake-client-secret")
        with patch("jrm_advisor.genie.client.WorkspaceClient") as MockWS:
            MockWS.return_value = MagicMock()
            GenieClient()
            assert MockWS.call_args.kwargs.get("auth_type") == "pat"


# ---------------------------------------------------------------------------
# AC: ask() returns GenieResult with rows, sql, question
# ---------------------------------------------------------------------------


class TestAskSuccess:
    def _setup(self, mock_ws, columns, rows):
        """Wire both the manifest response and the chunk data."""
        mock_ws.genie.start_conversation_and_wait.return_value = _make_genie_message()
        mock_ws.genie.get_message_query_result.return_value = (
            _make_query_result_response(columns=columns, rows=rows)
        )
        mock_ws.statement_execution.get_statement_result_chunk_n.return_value = (
            _make_chunk(rows)
        )

    def test_returns_genie_result(self, mock_ws):
        self._setup(
            mock_ws,
            columns=["year_week", "sales_uplift_percentage"],
            rows=[["202501", "0.12"], ["202502", "0.18"]],
        )
        result = GenieClient().ask(
            "Show weekly sales uplift for the Coca-Cola campaign"
        )
        assert isinstance(result, GenieResult)

    def test_result_contains_question(self, mock_ws):
        self._setup(
            mock_ws,
            columns=["year_week", "sales_uplift_percentage"],
            rows=[["202501", "0.12"]],
        )
        result = GenieClient().ask(
            "Show weekly sales uplift for the Coca-Cola campaign"
        )
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
        mock_ws.statement_execution.get_statement_result_chunk_n.return_value = (
            _make_chunk([["202501", "0.12"]])
        )
        result = GenieClient().ask("any question")
        assert result.sql == expected_sql

    def test_result_rows_mapped_to_column_names(self, mock_ws):
        self._setup(
            mock_ws,
            columns=["year_week", "sales_uplift_percentage"],
            rows=[["202501", "0.12"], ["202502", "0.18"]],
        )
        result = GenieClient().ask("any question")
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
        # total_row_count=0 means _fetch_rows returns early without calling chunk API
        result = GenieClient().ask("any question")
        assert result.rows == []

    def test_correct_space_id_passed_to_start_conversation(self, mock_ws):
        self._setup(
            mock_ws,
            columns=["year_week", "actual_sales"],
            rows=[["202501", "5000.0"]],
        )
        GenieClient().ask("any question")
        mock_ws.genie.start_conversation_and_wait.assert_called_once_with(
            space_id="test-space-id",
            content="any question",
        )

    def test_chunk_api_called_with_statement_id(self, mock_ws):
        self._setup(
            mock_ws,
            columns=["year_week", "sales_uplift_percentage"],
            rows=[["202501", "0.12"]],
        )
        GenieClient().ask("any question")
        mock_ws.statement_execution.get_statement_result_chunk_n.assert_called_once_with(
            statement_id="stmt-001",
            chunk_index=0,
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
        mock_ws.genie.get_message_query_result.return_value = (
            _make_query_result_response(
                columns=["year_week", "sales_uplift_percentage"],
                rows=[
                    ["202501", "0.12"]
                ],  # total_row_count=1 so chunk fetch is attempted
            )
        )
        mock_ws.statement_execution.get_statement_result_chunk_n.side_effect = (
            RuntimeError("network error")
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
        mock_ws.statement_execution.get_statement_result_chunk_n.return_value = (
            _make_chunk([["202501", "0.12"]])
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
        mock_ws.statement_execution.get_statement_result_chunk_n.return_value = (
            _make_chunk([["202501", "0.99"]])
        )

        client = GenieClient()
        with caplog.at_level(logging.DEBUG, logger="jrm_advisor.genie.client"):
            client.ask("any question")

        all_messages = " ".join(r.message for r in caplog.records)
        assert "0.99" not in all_messages, "Row values must never appear in log output"
