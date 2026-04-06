"""
Tests for jrm_advisor.app.backend_mock — deterministic mock backend.

No Databricks connection required.
"""

from __future__ import annotations

from jrm_advisor.app.backend import AppResponse
from jrm_advisor.app.backend_mock import MockBackend


class TestMockBackendReturnType:
    def test_returns_app_response(self):
        b = MockBackend()
        result = b.ask("what is sales uplift")
        assert isinstance(result, AppResponse)

    def test_unknown_question_returns_default(self):
        b = MockBackend()
        result = b.ask("completely unknown xyzzy question")
        assert result.text  # non-empty
        assert isinstance(result, AppResponse)


class TestMockBackendExactMatch:
    def test_kb_question_no_visualization(self):
        b = MockBackend()
        result = b.ask("what is sales uplift")
        assert result.visualization is None
        assert result.needs_clarification is False
        assert "uplift" in result.text.lower() or "incremental" in result.text.lower()

    def test_out_of_scope_question(self):
        b = MockBackend()
        result = b.ask("ropo analysis")
        assert "not" in result.text.lower()
        assert result.needs_clarification is False

    def test_clarification_question(self):
        b = MockBackend()
        result = b.ask("show results for heineken")
        assert result.needs_clarification is True
        assert result.visualization is None
        assert result.genie_rows == []


class TestMockBackendDataResponses:
    def test_heineken_line_chart_spec(self):
        b = MockBackend()
        result = b.ask("show weekly sales for heineken")
        assert result.visualization is not None
        assert result.visualization["should_visualize"] is True
        assert result.visualization["chart_type"] == "line"
        assert result.visualization["y_field"] == "actual_sales"

    def test_heineken_has_four_rows(self):
        b = MockBackend()
        result = b.ask("show weekly sales for heineken")
        assert len(result.genie_rows) == 4

    def test_heineken_resolved_campaign_set(self):
        b = MockBackend()
        result = b.ask("show weekly sales for heineken")
        assert result.resolved_campaign_name == "Heineken Pilsner W4 2024"

    def test_cocacola_bar_chart_spec(self):
        b = MockBackend()
        result = b.ask("show uplift for coca-cola")
        assert result.visualization is not None
        assert result.visualization["chart_type"] == "bar"
        assert result.visualization["y_field"] == "sales_uplift_percentage"
        assert result.visualization["scale"] == "x100"

    def test_cocacola_has_sql(self):
        b = MockBackend()
        result = b.ask("show uplift for coca-cola")
        assert result.genie_sql  # non-empty


class TestMockBackendCaseInsensitive:
    def test_uppercase_question(self):
        b = MockBackend()
        result = b.ask("WHAT IS SALES UPLIFT")
        assert isinstance(result, AppResponse)
        # Should match via lowercase key
        assert "incremental" in result.text.lower() or "uplift" in result.text.lower()

    def test_trailing_punctuation_stripped(self):
        b = MockBackend()
        result = b.ask("what is sales uplift?")
        assert "incremental" in result.text.lower() or "uplift" in result.text.lower()
