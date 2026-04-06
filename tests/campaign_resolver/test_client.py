"""
Unit tests for src/jrm_advisor/campaign_resolver/client.py

All tests mock the Databricks SDK — no live warehouse calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from jrm_advisor.campaign_resolver.client import (
    CampaignResolution,
    CampaignResolverClient,
    CampaignResolverError,
    CampaignResolverTimeoutError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(monkeypatch) -> CampaignResolverClient:
    """Return a CampaignResolverClient with env vars and a mocked WorkspaceClient."""
    monkeypatch.setenv("DATABRICKS_HOST", "https://test.azuredatabricks.net")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapiTEST")
    monkeypatch.setenv("DATABRICKS_SQL_WAREHOUSE_ID", "warehouse-abc")
    monkeypatch.setenv("CAMPAIGN_RESOLVER_THRESHOLD", "0.7")
    monkeypatch.setenv("CAMPAIGN_RESOLVER_AMBIGUITY_GAP", "0.05")

    with patch("jrm_advisor.campaign_resolver.client.WorkspaceClient"):
        client = CampaignResolverClient()
    return client


def _make_statement_response(
    rows: list[list], columns: list[str], state: str = "SUCCEEDED"
):
    """Build a minimal fake StatementResponse."""
    from databricks.sdk.service.sql import StatementState

    state_enum = getattr(StatementState, state)

    col_mocks = []
    for c in columns:
        m = MagicMock()
        m.name = c
        col_mocks.append(m)
    schema_mock = MagicMock()
    schema_mock.columns = col_mocks

    manifest_mock = MagicMock()
    manifest_mock.schema = schema_mock

    result_mock = MagicMock()
    result_mock.data_array = rows

    status_mock = MagicMock()
    status_mock.state = state_enum
    status_mock.error = None

    response = MagicMock()
    response.manifest = manifest_mock
    response.result = result_mock
    response.status = status_mock
    return response


def _rows_for_matches(matches: list[tuple[str, float]]) -> tuple[list[list], list[str]]:
    """Build (rows, columns) for a list of (name, score) tuples."""
    columns = ["page_content", "metadata"]
    rows = [
        [name, {"score": str(score), "Brand": "TestBrand"}] for name, score in matches
    ]
    return rows, columns


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


class TestCredentials:
    def test_missing_host_raises(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_TOKEN", "tok")
        monkeypatch.setenv("DATABRICKS_SQL_WAREHOUSE_ID", "wh")
        monkeypatch.delenv("DATABRICKS_HOST", raising=False)
        with pytest.raises(ValueError, match="DATABRICKS_HOST"):
            CampaignResolverClient()

    def test_missing_token_raises(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_HOST", "https://x.net")
        monkeypatch.setenv("DATABRICKS_SQL_WAREHOUSE_ID", "wh")
        monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
        with pytest.raises(ValueError, match="DATABRICKS_TOKEN"):
            CampaignResolverClient()

    def test_missing_warehouse_id_raises(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_HOST", "https://x.net")
        monkeypatch.setenv("DATABRICKS_TOKEN", "tok")
        monkeypatch.delenv("DATABRICKS_SQL_WAREHOUSE_ID", raising=False)
        with pytest.raises(ValueError, match="DATABRICKS_SQL_WAREHOUSE_ID"):
            CampaignResolverClient()

    def test_all_vars_set_succeeds(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_HOST", "https://x.net")
        monkeypatch.setenv("DATABRICKS_TOKEN", "tok")
        monkeypatch.setenv("DATABRICKS_SQL_WAREHOUSE_ID", "wh")
        with patch("jrm_advisor.campaign_resolver.client.WorkspaceClient"):
            client = CampaignResolverClient()
        assert client is not None


# ---------------------------------------------------------------------------
# Successful resolution
# ---------------------------------------------------------------------------


class TestResolveSuccess:
    def test_returns_campaign_resolution(self, monkeypatch):
        client = _make_client(monkeypatch)
        rows, cols = _rows_for_matches([("Heineken Pilsner W4 2024", 0.91)])
        response = _make_statement_response(rows, cols)
        client._ws.statement_execution.execute_statement.return_value = response

        result = client.resolve("Show Heineken results")
        assert isinstance(result, CampaignResolution)

    def test_match_name_correct(self, monkeypatch):
        client = _make_client(monkeypatch)
        rows, cols = _rows_for_matches([("Heineken Pilsner W4 2024", 0.91)])
        client._ws.statement_execution.execute_statement.return_value = (
            _make_statement_response(rows, cols)
        )
        result = client.resolve("Show Heineken results")
        assert result.match is not None
        assert result.match.name == "Heineken Pilsner W4 2024"

    def test_match_score_correct(self, monkeypatch):
        client = _make_client(monkeypatch)
        rows, cols = _rows_for_matches([("Heineken Pilsner W4 2024", 0.91)])
        client._ws.statement_execution.execute_statement.return_value = (
            _make_statement_response(rows, cols)
        )
        result = client.resolve("Show Heineken results")
        assert result.match.score == pytest.approx(0.91)

    def test_not_ambiguous_when_gap_is_large(self, monkeypatch):
        client = _make_client(monkeypatch)
        rows, cols = _rows_for_matches(
            [("Heineken Pilsner W4 2024", 0.91), ("Heineken 0.0 W3 2024", 0.74)]
        )
        client._ws.statement_execution.execute_statement.return_value = (
            _make_statement_response(rows, cols)
        )
        result = client.resolve("Show Heineken results")
        assert result.is_ambiguous is False
        assert result.match is not None

    def test_raw_query_stored(self, monkeypatch):
        client = _make_client(monkeypatch)
        rows, cols = _rows_for_matches([("Heineken W4 2024", 0.88)])
        client._ws.statement_execution.execute_statement.return_value = (
            _make_statement_response(rows, cols)
        )
        result = client.resolve("Show me Heineken")
        assert result.raw_query == "Show me Heineken"

    def test_candidates_returned(self, monkeypatch):
        client = _make_client(monkeypatch)
        rows, cols = _rows_for_matches(
            [("Campaign A", 0.92), ("Campaign B", 0.81), ("Campaign C", 0.73)]
        )
        client._ws.statement_execution.execute_statement.return_value = (
            _make_statement_response(rows, cols)
        )
        result = client.resolve("something")
        assert len(result.candidates) == 3


# ---------------------------------------------------------------------------
# Threshold and ambiguity
# ---------------------------------------------------------------------------


class TestThresholdAndAmbiguity:
    def test_no_match_when_score_below_threshold(self, monkeypatch):
        client = _make_client(monkeypatch)
        rows, cols = _rows_for_matches([("Some Campaign", 0.65)])
        client._ws.statement_execution.execute_statement.return_value = (
            _make_statement_response(rows, cols)
        )
        result = client.resolve("something obscure")
        assert result.match is None
        assert result.candidates == []

    def test_ambiguous_when_gap_is_small(self, monkeypatch):
        client = _make_client(monkeypatch)
        rows, cols = _rows_for_matches(
            [("Heineken Pilsner W4 2024", 0.88), ("Heineken 0.0 W4 2024", 0.84)]
        )
        client._ws.statement_execution.execute_statement.return_value = (
            _make_statement_response(rows, cols)
        )
        result = client.resolve("Show Heineken results")
        assert result.is_ambiguous is True
        assert result.match is None  # no winner chosen when ambiguous
        assert len(result.candidates) == 2

    def test_match_returned_when_second_below_threshold(self, monkeypatch):
        client = _make_client(monkeypatch)
        # First above threshold, second below — not ambiguous by gap but gap check only
        # runs on above-threshold candidates. Second at 0.60 is filtered out.
        rows, cols = _rows_for_matches(
            [("Heineken Pilsner W4 2024", 0.88), ("Unrelated Campaign", 0.60)]
        )
        client._ws.statement_execution.execute_statement.return_value = (
            _make_statement_response(rows, cols)
        )
        result = client.resolve("Show Heineken results")
        assert result.is_ambiguous is False
        assert result.match is not None
        assert result.match.name == "Heineken Pilsner W4 2024"

    def test_empty_rows_returns_no_match(self, monkeypatch):
        client = _make_client(monkeypatch)
        rows, cols = [], ["page_content", "metadata"]
        response = _make_statement_response(rows, cols)
        response.result.data_array = []
        client._ws.statement_execution.execute_statement.return_value = response
        result = client.resolve("something")
        assert result.match is None


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_sdk_exception_raises_resolver_error(self, monkeypatch):
        client = _make_client(monkeypatch)
        client._ws.statement_execution.execute_statement.side_effect = RuntimeError(
            "network error"
        )
        with pytest.raises(CampaignResolverError):
            client.resolve("any query")

    def test_failed_statement_raises_resolver_error(self, monkeypatch):

        client = _make_client(monkeypatch)
        rows, cols = [], ["page_content", "metadata"]
        response = _make_statement_response(rows, cols, state="FAILED")
        response.status.error = MagicMock(message="statement error")
        client._ws.statement_execution.execute_statement.return_value = response
        with pytest.raises(CampaignResolverError, match="statement failed"):
            client.resolve("any query")

    def test_timeout_statement_raises_timeout_error(self, monkeypatch):
        # The SDK has no TIMEDOUT state. When wait_timeout expires the SDK returns
        # the statement in a non-terminal state (PENDING or RUNNING).
        client = _make_client(monkeypatch)
        rows, cols = [], ["page_content", "metadata"]
        response = _make_statement_response(rows, cols, state="PENDING")
        client._ws.statement_execution.execute_statement.return_value = response
        with pytest.raises(CampaignResolverTimeoutError):
            client.resolve("any query")

    def test_score_not_in_final_answer(self, monkeypatch):
        """The score field must be internal — not reachable via match.name."""
        client = _make_client(monkeypatch)
        rows, cols = _rows_for_matches([("Heineken Pilsner W4 2024", 0.91)])
        client._ws.statement_execution.execute_statement.return_value = (
            _make_statement_response(rows, cols)
        )
        result = client.resolve("Show Heineken results")
        assert result.match is not None
        assert "0.91" not in result.match.name
        assert "score" not in result.match.name.lower()
