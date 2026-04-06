"""
Unit tests for src/jrm_advisor/supervisor/agent.py.

All sub-components (KB, Genie) are mocked — no live API calls.

Acceptance criteria covered:
  - classify_intent routes correctly for KB, data, hybrid, and out-of-scope questions
  - wants_visualization detects chart keywords
  - SupervisorAgent.answer() calls only the correct sub-components per intent
  - KB-only questions: returns KB answer, no Genie call
  - Data-only questions: returns Genie data, no KB call
  - Hybrid questions: calls both, combines answer
  - Visualization spec attached when chart keywords present AND rows available
  - Visualization spec NOT attached when no rows
  - Visualization spec NOT attached when no chart keyword
  - Out-of-scope questions: returns refusal, no sub-component calls
  - KB timeout/error: returns graceful fallback, does not raise
  - Genie timeout: returns graceful fallback, does not raise
  - Genie no-result: returns graceful fallback, does not raise
  - Genie error: returns graceful fallback, does not raise
  - Empty question defaults to KB_ONLY intent
  - SupervisorResponse fields: text, visualization, genie_rows, genie_sql
"""

from __future__ import annotations

from unittest.mock import MagicMock

from jrm_advisor.genie.client import (
    GenieError,
    GenieNoResultError,
    GenieResult,
    GenieTimeoutError,
)
from jrm_advisor.knowledge_base.client import (
    ANSWER_UNAVAILABLE,
    KnowledgeBaseError,
    KnowledgeBaseTimeoutError,
)
from jrm_advisor.supervisor.agent import (
    Intent,
    SupervisorAgent,
    SupervisorResponse,
    _compose_answer,
    classify_intent,
    wants_visualization,
)
from jrm_advisor.visualization.spec import VisualizationSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_genie_result(
    rows: list[dict] | None = None,
    sql: str = "SELECT year_week, sales_uplift_percentage FROM campaign",
    question: str = "test question",
) -> GenieResult:
    return GenieResult(
        question=question,
        sql=sql,
        rows=rows if rows is not None else [],
    )


def _make_agent(
    kb_answer: str = "Sales uplift is incremental sales.",
    genie_rows: list[dict] | None = None,
    genie_side_effect=None,
    kb_side_effect=None,
) -> tuple[SupervisorAgent, MagicMock, MagicMock]:
    """Build a SupervisorAgent with mocked sub-components."""
    mock_kb = MagicMock()
    mock_genie = MagicMock()

    if kb_side_effect:
        mock_kb.ask.side_effect = kb_side_effect
    else:
        mock_kb.ask.return_value = kb_answer

    if genie_side_effect:
        mock_genie.ask.side_effect = genie_side_effect
    else:
        mock_genie.ask.return_value = _make_genie_result(rows=genie_rows or [])

    agent = SupervisorAgent(kb_client=mock_kb, genie_client=mock_genie)
    return agent, mock_kb, mock_genie


# ---------------------------------------------------------------------------
# classify_intent
# ---------------------------------------------------------------------------


class TestClassifyIntent:
    def test_definition_question_is_kb_only(self):
        # "sales uplift" hits both KB and data keywords → hybrid; use a pure KB phrase
        assert classify_intent("What is the baseline methodology?") == Intent.KB_ONLY

    def test_terminology_question_is_kb_only(self):
        assert classify_intent("Define OTS for me") == Intent.KB_ONLY

    def test_methodology_question_is_kb_only(self):
        assert classify_intent("How is the baseline calculated?") == Intent.KB_ONLY

    def test_kpi_question_is_kb_only(self):
        assert classify_intent("Explain the KPI methodology") == Intent.KB_ONLY

    def test_sales_performance_is_data_only(self):
        assert (
            classify_intent("Show weekly sales for the Coca-Cola campaign")
            == Intent.DATA_ONLY
        )

    def test_campaign_results_is_data_only(self):
        assert (
            classify_intent("Give me the campaign results for Q1") == Intent.DATA_ONLY
        )

    def test_uplift_performance_is_hybrid(self):
        # "uplift" hits KB_KEYWORDS, "campaign" + "performance" hit DATA_KEYWORDS
        result = classify_intent(
            "What is uplift and how did my campaign perform this period?"
        )
        assert result == Intent.HYBRID

    def test_explain_and_show_is_hybrid(self):
        result = classify_intent(
            "Explain sales uplift and show me the weekly trend for Coca-Cola"
        )
        assert result == Intent.HYBRID

    def test_store_level_roi_is_out_of_scope(self):
        assert (
            classify_intent("Show me store-level ROI for the campaign")
            == Intent.OUT_OF_SCOPE
        )

    def test_ropo_is_out_of_scope(self):
        assert classify_intent("Can you do ROPO analysis?") == Intent.OUT_OF_SCOPE

    def test_digital_channel_is_out_of_scope(self):
        assert (
            classify_intent("How did the digital campaign perform?")
            == Intent.OUT_OF_SCOPE
        )

    def test_ooh_is_out_of_scope(self):
        assert classify_intent("Show OOH campaign results") == Intent.OUT_OF_SCOPE

    def test_forecast_is_out_of_scope(self):
        assert classify_intent("Forecast the pre-launch uplift") == Intent.OUT_OF_SCOPE

    def test_empty_question_defaults_to_kb_only(self):
        assert classify_intent("") == Intent.KB_ONLY

    def test_case_insensitive(self):
        assert classify_intent("WHAT IS THE BASELINE METHODOLOGY?") == Intent.KB_ONLY

    def test_wap_is_out_of_scope(self):
        assert classify_intent("What is the WAP for this store?") == Intent.OUT_OF_SCOPE


# ---------------------------------------------------------------------------
# wants_visualization
# ---------------------------------------------------------------------------


class TestWantsVisualization:
    def test_chart_keyword_returns_true(self):
        assert wants_visualization("Show me a chart of weekly sales") is True

    def test_graph_keyword_returns_true(self):
        assert wants_visualization("Can you graph the uplift?") is True

    def test_trend_keyword_returns_true(self):
        assert wants_visualization("Show the trend over time") is True

    def test_visualize_keyword_returns_true(self):
        assert wants_visualization("Please visualize the results") is True

    def test_no_viz_keyword_returns_false(self):
        assert wants_visualization("Show weekly sales for the campaign") is False

    def test_definition_question_returns_false(self):
        assert wants_visualization("What is sales uplift?") is False


# ---------------------------------------------------------------------------
# SupervisorAgent — intent routing (sub-component call verification)
# ---------------------------------------------------------------------------


class TestRoutingKbOnly:
    def test_kb_called_for_definition_question(self):
        agent, mock_kb, mock_genie = _make_agent()
        agent.answer("What is the baseline methodology?")
        mock_kb.ask.assert_called_once()

    def test_genie_not_called_for_definition_question(self):
        agent, mock_kb, mock_genie = _make_agent()
        agent.answer("What is the baseline methodology?")
        mock_genie.ask.assert_not_called()

    def test_response_contains_kb_answer(self):
        agent, _, _ = _make_agent(kb_answer="Sales uplift is incremental revenue.")
        response = agent.answer("What is the baseline methodology?")
        assert "incremental revenue" in response.text

    def test_response_is_supervisor_response(self):
        agent, _, _ = _make_agent()
        response = agent.answer("What is the baseline methodology?")
        assert isinstance(response, SupervisorResponse)

    def test_no_visualization_for_kb_only(self):
        agent, _, _ = _make_agent()
        response = agent.answer("What is the baseline methodology?")
        assert response.visualization is None


class TestRoutingDataOnly:
    def test_genie_called_for_data_question(self):
        agent, mock_kb, mock_genie = _make_agent(
            genie_rows=[{"year_week": "202501", "sales_uplift_percentage": "0.12"}]
        )
        agent.answer("Show weekly sales for the Coca-Cola campaign")
        mock_genie.ask.assert_called_once()

    def test_kb_not_called_for_data_question(self):
        agent, mock_kb, mock_genie = _make_agent()
        agent.answer("Show weekly sales for the Coca-Cola campaign")
        mock_kb.ask.assert_not_called()

    def test_response_contains_row_data(self):
        agent, _, _ = _make_agent(
            genie_rows=[{"year_week": "202501", "sales_uplift_percentage": "0.12"}]
        )
        response = agent.answer("Show weekly sales for the Coca-Cola campaign")
        assert "202501" in response.text

    def test_genie_rows_stored_in_response(self):
        rows = [{"year_week": "202501", "actual_sales": "5000"}]
        agent, _, _ = _make_agent(genie_rows=rows)
        response = agent.answer("Show weekly sales for the Coca-Cola campaign")
        assert response.genie_rows == rows

    def test_genie_sql_stored_in_response(self):
        mock_kb = MagicMock()
        mock_genie = MagicMock()
        expected_sql = "SELECT year_week, actual_sales FROM campaign"
        mock_genie.ask.return_value = _make_genie_result(
            rows=[{"year_week": "202501", "actual_sales": "5000"}],
            sql=expected_sql,
        )
        agent = SupervisorAgent(kb_client=mock_kb, genie_client=mock_genie)
        response = agent.answer("Show weekly sales for the Coca-Cola campaign")
        assert response.genie_sql == expected_sql


class TestRoutingHybrid:
    def test_both_components_called_for_hybrid_question(self):
        agent, mock_kb, mock_genie = _make_agent(
            genie_rows=[{"year_week": "202501", "sales_uplift_percentage": "0.12"}]
        )
        agent.answer("What is uplift and how did my campaign perform?")
        mock_kb.ask.assert_called_once()
        mock_genie.ask.assert_called_once()

    def test_response_contains_both_kb_and_data(self):
        agent, _, _ = _make_agent(
            kb_answer="Sales uplift is incremental revenue.",
            genie_rows=[{"year_week": "202501", "sales_uplift_percentage": "0.12"}],
        )
        response = agent.answer("What is uplift and how did my campaign perform?")
        assert "incremental revenue" in response.text
        assert "202501" in response.text


# ---------------------------------------------------------------------------
# SupervisorAgent — visualization gating
# ---------------------------------------------------------------------------


class TestVisualizationGating:
    def test_viz_attached_when_rows_and_chart_keyword(self):
        agent, _, _ = _make_agent(
            genie_rows=[{"year_week": "202501", "sales_uplift_percentage": "0.12"}]
        )
        response = agent.answer(
            "Show me a chart of weekly sales uplift for the campaign"
        )
        assert response.visualization is not None

    def test_viz_is_visualization_spec_when_matched(self):
        agent, _, _ = _make_agent(
            genie_rows=[{"year_week": "202501", "sales_uplift_percentage": "0.12"}]
        )
        response = agent.answer(
            "Show me a chart of weekly sales uplift for the campaign"
        )
        assert isinstance(response.visualization, VisualizationSpec)

    def test_viz_not_attached_when_no_rows(self):
        agent, _, _ = _make_agent(genie_rows=[])
        response = agent.answer("Show me a chart of weekly sales for the campaign")
        assert response.visualization is None

    def test_viz_not_attached_without_chart_keyword(self):
        agent, _, _ = _make_agent(
            genie_rows=[{"year_week": "202501", "sales_uplift_percentage": "0.12"}]
        )
        # "show" alone is not a viz keyword
        response = agent.answer("Show weekly sales uplift for the Coca-Cola campaign")
        assert response.visualization is None

    def test_viz_not_attached_for_kb_only_with_chart_keyword(self):
        # KB-only path never has rows → no viz
        agent, _, _ = _make_agent()
        response = agent.answer("What is uplift? Show me a chart")
        # Intent will be KB_ONLY (no data keywords strong enough) — no rows
        assert response.visualization is None or (
            isinstance(response.visualization, dict)
            and not response.visualization.get("should_visualize", True)
        )


# ---------------------------------------------------------------------------
# SupervisorAgent — out-of-scope handling
# ---------------------------------------------------------------------------


class TestOutOfScope:
    def test_out_of_scope_returns_refusal(self):
        agent, mock_kb, mock_genie = _make_agent()
        response = agent.answer("Can you do ROPO analysis?")
        assert "not" in response.text.lower()

    def test_out_of_scope_does_not_call_kb(self):
        agent, mock_kb, mock_genie = _make_agent()
        agent.answer("Can you do ROPO analysis?")
        mock_kb.ask.assert_not_called()

    def test_out_of_scope_does_not_call_genie(self):
        agent, mock_kb, mock_genie = _make_agent()
        agent.answer("Can you do ROPO analysis?")
        mock_genie.ask.assert_not_called()

    def test_store_level_roi_refusal_text(self):
        agent, _, _ = _make_agent()
        response = agent.answer("Show me the store-level ROI")
        assert (
            "store-level ROI" in response.text or "not yet supported" in response.text
        )

    def test_digital_channel_refusal_text(self):
        agent, _, _ = _make_agent()
        response = agent.answer("How did the digital campaign perform?")
        assert (
            "in-store" in response.text.lower()
            or "out of scope" in response.text.lower()
        )

    def test_out_of_scope_no_visualization(self):
        agent, _, _ = _make_agent()
        response = agent.answer("Can you do ROPO analysis?")
        assert response.visualization is None


# ---------------------------------------------------------------------------
# SupervisorAgent — error resilience
# ---------------------------------------------------------------------------


class TestKbErrors:
    def test_kb_timeout_returns_graceful_message(self):
        agent, _, _ = _make_agent(kb_side_effect=KnowledgeBaseTimeoutError("timeout"))
        response = agent.answer("What is sales uplift?")
        assert response.text  # non-empty
        assert (
            "timeout" in response.text.lower() or "try again" in response.text.lower()
        )

    def test_kb_error_returns_graceful_message(self):
        agent, _, _ = _make_agent(kb_side_effect=KnowledgeBaseError("API failure"))
        response = agent.answer("What is sales uplift?")
        assert response.text
        assert (
            "not available" in response.text.lower()
            or ANSWER_UNAVAILABLE in response.text
        )

    def test_kb_error_does_not_raise(self):
        agent, _, _ = _make_agent(kb_side_effect=KnowledgeBaseError("API failure"))
        # Should not raise
        response = agent.answer("What is sales uplift?")
        assert isinstance(response, SupervisorResponse)


class TestGenieErrors:
    def test_genie_timeout_returns_graceful_message(self):
        agent, _, _ = _make_agent(genie_side_effect=GenieTimeoutError("timeout"))
        response = agent.answer("Show weekly sales for the Coca-Cola campaign")
        assert response.text
        assert (
            "timed out" in response.text.lower() or "try again" in response.text.lower()
        )

    def test_genie_no_result_returns_graceful_message(self):
        agent, _, _ = _make_agent(genie_side_effect=GenieNoResultError("no data"))
        response = agent.answer("Show weekly sales for the Coca-Cola campaign")
        assert response.text
        assert (
            "no results" in response.text.lower() or "no data" in response.text.lower()
        )

    def test_genie_error_returns_graceful_message(self):
        agent, _, _ = _make_agent(genie_side_effect=GenieError("API failure"))
        response = agent.answer("Show weekly sales for the Coca-Cola campaign")
        assert response.text
        assert "not available" in response.text.lower()

    def test_genie_error_does_not_raise(self):
        agent, _, _ = _make_agent(genie_side_effect=GenieError("API failure"))
        response = agent.answer("Show weekly sales for the Coca-Cola campaign")
        assert isinstance(response, SupervisorResponse)

    def test_genie_timeout_no_viz(self):
        agent, _, _ = _make_agent(genie_side_effect=GenieTimeoutError("timeout"))
        response = agent.answer("Show me a chart of weekly sales for the campaign")
        assert response.visualization is None

    def test_genie_error_genie_rows_empty(self):
        agent, _, _ = _make_agent(genie_side_effect=GenieError("fail"))
        response = agent.answer("Show weekly sales for the Coca-Cola campaign")
        assert response.genie_rows == []


# ---------------------------------------------------------------------------
# _compose_answer (pure function unit tests)
# ---------------------------------------------------------------------------


class TestComposeAnswer:
    def test_kb_only_returns_kb_answer(self):
        result = _compose_answer(
            intent=Intent.KB_ONLY,
            kb_answer="Uplift is incremental sales.",
            genie_result=None,
            genie_error=None,
        )
        assert "incremental sales" in result

    def test_kb_only_unavailable_returns_fallback(self):
        result = _compose_answer(
            intent=Intent.KB_ONLY,
            kb_answer=ANSWER_UNAVAILABLE,
            genie_result=None,
            genie_error=None,
        )
        assert ANSWER_UNAVAILABLE in result

    def test_data_only_with_rows_returns_table(self):
        genie = _make_genie_result(
            rows=[{"year_week": "202501", "sales_uplift_percentage": "0.12"}]
        )
        result = _compose_answer(
            intent=Intent.DATA_ONLY,
            kb_answer=None,
            genie_result=genie,
            genie_error=None,
        )
        assert "202501" in result

    def test_data_only_with_error_returns_error_message(self):
        result = _compose_answer(
            intent=Intent.DATA_ONLY,
            kb_answer=None,
            genie_result=None,
            genie_error="Data query timed out.",
        )
        assert "timed out" in result

    def test_hybrid_combines_kb_and_data(self):
        genie = _make_genie_result(
            rows=[{"year_week": "202501", "sales_uplift_percentage": "0.12"}]
        )
        result = _compose_answer(
            intent=Intent.HYBRID,
            kb_answer="Uplift is incremental sales.",
            genie_result=genie,
            genie_error=None,
        )
        assert "incremental sales" in result
        assert "202501" in result

    def test_empty_rows_returns_no_data_message(self):
        genie = _make_genie_result(rows=[])
        result = _compose_answer(
            intent=Intent.DATA_ONLY,
            kb_answer=None,
            genie_result=genie,
            genie_error=None,
        )
        assert (
            "no data" in result.lower()
            or "no results" in result.lower()
            or "no data was returned" in result.lower()
        )
