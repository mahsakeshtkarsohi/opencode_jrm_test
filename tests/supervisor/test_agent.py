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

from jrm_advisor.campaign_resolver.client import (
    CampaignMatch,
    CampaignResolution,
    CampaignResolverError,
)
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
    resolver_resolution: CampaignResolution | None = None,
    resolver_side_effect=None,
) -> tuple[SupervisorAgent, MagicMock, MagicMock]:
    """Build a SupervisorAgent with mocked sub-components."""
    mock_kb = MagicMock()
    mock_genie = MagicMock()
    mock_resolver = MagicMock()
    mock_composer = MagicMock()

    if kb_side_effect:
        mock_kb.ask.side_effect = kb_side_effect
    else:
        mock_kb.ask.return_value = kb_answer

    if genie_side_effect:
        mock_genie.ask.side_effect = genie_side_effect
    else:
        mock_genie.ask.return_value = _make_genie_result(rows=genie_rows or [])

    if resolver_side_effect:
        mock_resolver.resolve.side_effect = resolver_side_effect
    elif resolver_resolution is not None:
        mock_resolver.resolve.return_value = resolver_resolution
    else:
        # Default: no match (score below threshold)
        mock_resolver.resolve.return_value = CampaignResolution(
            match=None, candidates=[], is_ambiguous=False, raw_query=""
        )

    # Composer passes through to _compose_answer fallback by default (empty LLM)
    mock_composer.compose.side_effect = lambda **kw: _compose_answer(
        intent=kw["intent"],
        kb_answer=kw.get("kb_answer"),
        genie_result=_make_genie_result(rows=kw.get("genie_rows") or []),
        genie_error=None,
    )

    agent = SupervisorAgent(
        kb_client=mock_kb,
        genie_client=mock_genie,
        resolver_client=mock_resolver,
        composer=mock_composer,
    )
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


# ---------------------------------------------------------------------------
# SupervisorAgent — campaign name resolution (#007)
# ---------------------------------------------------------------------------


def _clear_match(name: str, score: float = 0.91) -> CampaignResolution:
    """Helper: single unambiguous match above threshold."""
    m = CampaignMatch(name=name, score=score, metadata={})
    return CampaignResolution(
        match=m, candidates=[m], is_ambiguous=False, raw_query="test query"
    )


def _ambiguous_resolution(names: list[str]) -> CampaignResolution:
    """Helper: two candidates with close scores → ambiguous."""
    candidates = [
        CampaignMatch(name=n, score=0.88 - i * 0.02, metadata={})
        for i, n in enumerate(names)
    ]
    return CampaignResolution(
        match=None, candidates=candidates, is_ambiguous=True, raw_query="test query"
    )


def _no_match_resolution() -> CampaignResolution:
    return CampaignResolution(
        match=None, candidates=[], is_ambiguous=False, raw_query="test query"
    )


class TestCampaignResolution:
    def test_resolved_name_injected_into_genie_question(self):
        """When resolver finds a match, Genie is called with the resolved name."""
        resolution = _clear_match("Heineken Pilsner W4 2024")
        agent, _, mock_genie = _make_agent(
            genie_rows=[{"year_week": "202504", "actual_sales": "12000"}],
            resolver_resolution=resolution,
        )
        agent.answer("Show me results for Heineken")
        call_args = mock_genie.ask.call_args[0][0]
        assert "Heineken Pilsner W4 2024" in call_args

    def test_resolved_campaign_name_stored_in_response(self):
        resolution = _clear_match("Heineken Pilsner W4 2024")
        agent, _, _ = _make_agent(
            genie_rows=[{"year_week": "202504", "actual_sales": "12000"}],
            resolver_resolution=resolution,
        )
        response = agent.answer("Show me results for Heineken")
        assert response.resolved_campaign_name == "Heineken Pilsner W4 2024"

    def test_no_match_uses_raw_question_for_genie(self):
        """When resolver finds no match, Genie receives the raw question."""
        agent, _, mock_genie = _make_agent(
            genie_rows=[],
            resolver_resolution=_no_match_resolution(),
        )
        agent.answer("Show me results for BrandX")
        call_args = mock_genie.ask.call_args[0][0]
        assert "BrandX" in call_args
        assert "[Campaign:" not in call_args

    def test_resolver_error_falls_back_to_raw_question(self):
        """Resolver errors degrade gracefully — raw question passed to Genie."""
        agent, _, mock_genie = _make_agent(
            genie_rows=[],
            resolver_side_effect=CampaignResolverError("network error"),
        )
        response = agent.answer("Show me results for Heineken")
        # Should not raise; Genie was still called
        mock_genie.ask.assert_called_once()
        assert isinstance(response, SupervisorResponse)

    def test_resolver_not_called_for_kb_only_intent(self):
        """Resolver is only invoked for DATA_ONLY and HYBRID intents."""
        mock_resolver = MagicMock()
        mock_kb = MagicMock()
        mock_kb.ask.return_value = "Uplift is incremental revenue."
        mock_composer = MagicMock()
        mock_composer.compose.return_value = "Uplift is incremental revenue."
        agent = SupervisorAgent(
            kb_client=mock_kb,
            genie_client=MagicMock(),
            resolver_client=mock_resolver,
            composer=mock_composer,
        )
        agent.answer("What is the baseline methodology?")
        mock_resolver.resolve.assert_not_called()

    def test_resolver_none_disables_resolution(self):
        """Passing resolver_client=None disables campaign resolution entirely."""
        mock_genie = MagicMock()
        mock_genie.ask.return_value = _make_genie_result(rows=[])
        mock_composer = MagicMock()
        mock_composer.compose.return_value = "No data."
        agent = SupervisorAgent(
            kb_client=MagicMock(),
            genie_client=mock_genie,
            resolver_client=None,
            composer=mock_composer,
        )
        agent.answer("Show Heineken results")
        # No resolution → Genie called with raw question, no [Campaign:] tag
        call_args = mock_genie.ask.call_args[0][0]
        assert "[Campaign:" not in call_args

    def test_resolved_campaign_name_none_when_no_match(self):
        agent, _, _ = _make_agent(
            genie_rows=[],
            resolver_resolution=_no_match_resolution(),
        )
        response = agent.answer("Show me results for BrandX")
        assert response.resolved_campaign_name is None


# ---------------------------------------------------------------------------
# SupervisorAgent — ambiguity handler (#008)
# ---------------------------------------------------------------------------


class TestAmbiguityHandler:
    def test_ambiguous_resolution_returns_clarification(self):
        resolution = _ambiguous_resolution(
            ["Heineken Pilsner W4 2024", "Heineken 0.0 W4 2024"]
        )
        agent, _, mock_genie = _make_agent(resolver_resolution=resolution)
        response = agent.answer("Show me results for Heineken")
        assert response.needs_clarification is True
        # Genie must NOT be called
        mock_genie.ask.assert_not_called()

    def test_clarification_text_lists_candidate_names(self):
        resolution = _ambiguous_resolution(
            ["Heineken Pilsner W4 2024", "Heineken 0.0 W4 2024"]
        )
        agent, _, _ = _make_agent(resolver_resolution=resolution)
        response = agent.answer("Show me results for Heineken")
        assert "Heineken Pilsner W4 2024" in response.text
        assert "Heineken 0.0 W4 2024" in response.text

    def test_clarification_text_no_scores(self):
        """Scores must never appear in the clarification text."""
        resolution = _ambiguous_resolution(
            ["Heineken Pilsner W4 2024", "Heineken 0.0 W4 2024"]
        )
        agent, _, _ = _make_agent(resolver_resolution=resolution)
        response = agent.answer("Show me results for Heineken")
        assert "0.88" not in response.text
        assert "0.86" not in response.text
        assert "score" not in response.text.lower()

    def test_clarification_text_no_internal_metadata(self):
        """article_cmp_id and other internal keys must not appear in the text."""
        resolution = _ambiguous_resolution(
            ["Heineken Pilsner W4 2024", "Heineken 0.0 W4 2024"]
        )
        agent, _, _ = _make_agent(resolver_resolution=resolution)
        response = agent.answer("Show me results for Heineken")
        assert "article_cmp_id" not in response.text
        assert "metadata" not in response.text.lower()

    def test_clarification_no_visualization(self):
        resolution = _ambiguous_resolution(
            ["Heineken Pilsner W4 2024", "Heineken 0.0 W4 2024"]
        )
        agent, _, _ = _make_agent(resolver_resolution=resolution)
        response = agent.answer("Show me results for Heineken")
        assert response.visualization is None

    def test_clarification_genie_rows_empty(self):
        resolution = _ambiguous_resolution(
            ["Heineken Pilsner W4 2024", "Heineken 0.0 W4 2024"]
        )
        agent, _, _ = _make_agent(resolver_resolution=resolution)
        response = agent.answer("Show me results for Heineken")
        assert response.genie_rows == []

    def test_clear_match_does_not_trigger_clarification(self):
        resolution = _clear_match("Heineken Pilsner W4 2024")
        agent, _, _ = _make_agent(
            genie_rows=[{"year_week": "202504", "actual_sales": "12000"}],
            resolver_resolution=resolution,
        )
        response = agent.answer("Show me results for Heineken")
        assert response.needs_clarification is False

    def test_needs_clarification_false_for_out_of_scope(self):
        agent, _, _ = _make_agent()
        response = agent.answer("Can you do ROPO analysis?")
        assert response.needs_clarification is False

    def test_needs_clarification_false_for_kb_only(self):
        agent, _, _ = _make_agent(kb_answer="Uplift is incremental.")
        response = agent.answer("What is the baseline methodology?")
        assert response.needs_clarification is False


# ---------------------------------------------------------------------------
# SupervisorAgent — AnswerComposer wiring (#009)
# ---------------------------------------------------------------------------


class TestAnswerComposerWiring:
    def test_composer_called_for_data_only_intent(self):
        mock_kb = MagicMock()
        mock_genie = MagicMock()
        mock_genie.ask.return_value = _make_genie_result(
            rows=[{"year_week": "202501", "actual_sales": "5000"}]
        )
        mock_resolver = MagicMock()
        mock_resolver.resolve.return_value = _no_match_resolution()
        mock_composer = MagicMock()
        mock_composer.compose.return_value = "LLM composed answer."

        agent = SupervisorAgent(
            kb_client=mock_kb,
            genie_client=mock_genie,
            resolver_client=mock_resolver,
            composer=mock_composer,
        )
        response = agent.answer("Show weekly sales for the Coca-Cola campaign")
        mock_composer.compose.assert_called_once()
        assert response.text == "LLM composed answer."

    def test_composer_called_for_hybrid_intent(self):
        mock_kb = MagicMock()
        mock_kb.ask.return_value = "Uplift is incremental."
        mock_genie = MagicMock()
        mock_genie.ask.return_value = _make_genie_result(
            rows=[{"year_week": "202501", "actual_sales": "5000"}]
        )
        mock_resolver = MagicMock()
        mock_resolver.resolve.return_value = _no_match_resolution()
        mock_composer = MagicMock()
        mock_composer.compose.return_value = "LLM hybrid answer."

        agent = SupervisorAgent(
            kb_client=mock_kb,
            genie_client=mock_genie,
            resolver_client=mock_resolver,
            composer=mock_composer,
        )
        response = agent.answer(
            "What is uplift and how did my campaign perform this period?"
        )
        mock_composer.compose.assert_called_once()
        assert response.text == "LLM hybrid answer."

    def test_composer_receives_correct_intent(self):
        mock_kb = MagicMock()
        mock_genie = MagicMock()
        mock_genie.ask.return_value = _make_genie_result(rows=[])
        mock_resolver = MagicMock()
        mock_resolver.resolve.return_value = _no_match_resolution()
        mock_composer = MagicMock()
        mock_composer.compose.return_value = "answer"

        agent = SupervisorAgent(
            kb_client=mock_kb,
            genie_client=mock_genie,
            resolver_client=mock_resolver,
            composer=mock_composer,
        )
        agent.answer("Show weekly sales for the Coca-Cola campaign")
        call_kwargs = mock_composer.compose.call_args[1]
        assert call_kwargs["intent"] == Intent.DATA_ONLY

    def test_genie_error_bypasses_composer(self):
        """When Genie fails, the error message is shown directly (no composer call)."""
        mock_kb = MagicMock()
        mock_genie = MagicMock()
        mock_genie.ask.side_effect = GenieTimeoutError("timeout")
        mock_resolver = MagicMock()
        mock_resolver.resolve.return_value = _no_match_resolution()
        mock_composer = MagicMock()
        mock_composer.compose.return_value = "should not appear"

        agent = SupervisorAgent(
            kb_client=mock_kb,
            genie_client=mock_genie,
            resolver_client=mock_resolver,
            composer=mock_composer,
        )
        response = agent.answer("Show weekly sales for the Coca-Cola campaign")
        mock_composer.compose.assert_not_called()
        assert (
            "timed out" in response.text.lower() or "try again" in response.text.lower()
        )


# ---------------------------------------------------------------------------
# SupervisorAgent — graceful degradation when clients fail to initialise
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """SupervisorAgent must not crash when KB or Genie env vars are missing.

    These tests verify that passing ``None`` for a client (which is what
    __init__ stores when construction fails) produces graceful fallback
    responses rather than AttributeError / TypeError at runtime.
    """

    def test_kb_none_kb_only_question_returns_unavailable(self):
        """KB=None → ANSWER_UNAVAILABLE returned, no exception raised."""
        mock_composer = MagicMock()
        mock_composer.compose.return_value = ANSWER_UNAVAILABLE
        agent = SupervisorAgent(
            kb_client=None,
            genie_client=MagicMock(),
            resolver_client=None,
            composer=mock_composer,
        )
        # Force kb_client to None to simulate missing KB_ENDPOINT_URL
        agent._kb = None
        response = agent.answer("What is the baseline methodology?")
        assert isinstance(response, SupervisorResponse)
        assert response.text  # non-empty
        assert (
            ANSWER_UNAVAILABLE in response.text
            or "not available" in response.text.lower()
        )

    def test_genie_none_data_only_question_returns_unavailable(self):
        """Genie=None → data-unavailable message returned, no exception raised."""
        mock_composer = MagicMock()
        mock_composer.compose.return_value = ""  # force fallback path
        agent = SupervisorAgent(
            kb_client=MagicMock(),
            genie_client=MagicMock(),
            resolver_client=None,
            composer=mock_composer,
        )
        agent._genie = None
        response = agent.answer("Show weekly sales for the Coca-Cola campaign")
        assert isinstance(response, SupervisorResponse)
        assert response.text
        assert (
            "not available" in response.text.lower()
            or "try again" in response.text.lower()
        )

    def test_kb_and_genie_none_does_not_raise(self):
        """When both KB and Genie are None the agent still returns a response."""
        mock_composer = MagicMock()
        mock_composer.compose.return_value = ""
        agent = SupervisorAgent(
            kb_client=MagicMock(),
            genie_client=MagicMock(),
            resolver_client=None,
            composer=mock_composer,
        )
        agent._kb = None
        agent._genie = None
        response = agent.answer("What is uplift and how did my campaign perform?")
        assert isinstance(response, SupervisorResponse)
        assert response.text


# ---------------------------------------------------------------------------
# SupervisorAgent — OBO user_token passthrough
# ---------------------------------------------------------------------------


class TestUserTokenPassthrough:
    """user_token is forwarded to all injected sub-clients via __init__."""

    def test_user_token_passed_to_kb_client(self):
        """When user_token is provided and KnowledgeBaseClient is constructed,
        the token is passed as the ``token`` kwarg."""
        from unittest.mock import patch

        with (
            patch("jrm_advisor.supervisor.agent.KnowledgeBaseClient") as mock_kb_cls,
            patch("jrm_advisor.supervisor.agent.GenieClient") as mock_genie_cls,
            patch(
                "jrm_advisor.supervisor.agent.CampaignResolverClient"
            ) as mock_resolver_cls,
            patch("jrm_advisor.supervisor.agent.AnswerComposer") as mock_composer_cls,
        ):
            mock_kb_cls.return_value = MagicMock()
            mock_genie_cls.return_value = MagicMock()
            mock_resolver_cls.return_value = MagicMock()
            mock_composer_cls.return_value = MagicMock()

            SupervisorAgent(user_token="dapi_test_token")

            mock_kb_cls.assert_called_once_with(token="dapi_test_token")

    def test_user_token_passed_to_genie_client(self):
        from unittest.mock import patch

        with (
            patch("jrm_advisor.supervisor.agent.KnowledgeBaseClient") as mock_kb_cls,
            patch("jrm_advisor.supervisor.agent.GenieClient") as mock_genie_cls,
            patch(
                "jrm_advisor.supervisor.agent.CampaignResolverClient"
            ) as mock_resolver_cls,
            patch("jrm_advisor.supervisor.agent.AnswerComposer") as mock_composer_cls,
        ):
            mock_kb_cls.return_value = MagicMock()
            mock_genie_cls.return_value = MagicMock()
            mock_resolver_cls.return_value = MagicMock()
            mock_composer_cls.return_value = MagicMock()

            SupervisorAgent(user_token="dapi_test_token")

            mock_genie_cls.assert_called_once_with(token="dapi_test_token")

    def test_user_token_passed_to_resolver_client(self):
        from unittest.mock import patch

        with (
            patch("jrm_advisor.supervisor.agent.KnowledgeBaseClient") as mock_kb_cls,
            patch("jrm_advisor.supervisor.agent.GenieClient") as mock_genie_cls,
            patch(
                "jrm_advisor.supervisor.agent.CampaignResolverClient"
            ) as mock_resolver_cls,
            patch("jrm_advisor.supervisor.agent.AnswerComposer") as mock_composer_cls,
        ):
            mock_kb_cls.return_value = MagicMock()
            mock_genie_cls.return_value = MagicMock()
            mock_resolver_cls.return_value = MagicMock()
            mock_composer_cls.return_value = MagicMock()

            SupervisorAgent(user_token="dapi_test_token")

            mock_resolver_cls.assert_called_once_with(token="dapi_test_token")

    def test_user_token_passed_to_composer(self):
        from unittest.mock import patch

        with (
            patch("jrm_advisor.supervisor.agent.KnowledgeBaseClient") as mock_kb_cls,
            patch("jrm_advisor.supervisor.agent.GenieClient") as mock_genie_cls,
            patch(
                "jrm_advisor.supervisor.agent.CampaignResolverClient"
            ) as mock_resolver_cls,
            patch("jrm_advisor.supervisor.agent.AnswerComposer") as mock_composer_cls,
        ):
            mock_kb_cls.return_value = MagicMock()
            mock_genie_cls.return_value = MagicMock()
            mock_resolver_cls.return_value = MagicMock()
            mock_composer_cls.return_value = MagicMock()

            SupervisorAgent(user_token="dapi_test_token")

            mock_composer_cls.assert_called_once_with(token="dapi_test_token")

    def test_no_user_token_uses_none(self):
        """When user_token is not passed, clients are constructed with token=None
        (they fall back to DATABRICKS_TOKEN env var themselves)."""
        from unittest.mock import patch

        with (
            patch("jrm_advisor.supervisor.agent.KnowledgeBaseClient") as mock_kb_cls,
            patch("jrm_advisor.supervisor.agent.GenieClient") as mock_genie_cls,
            patch(
                "jrm_advisor.supervisor.agent.CampaignResolverClient"
            ) as mock_resolver_cls,
            patch("jrm_advisor.supervisor.agent.AnswerComposer") as mock_composer_cls,
        ):
            mock_kb_cls.return_value = MagicMock()
            mock_genie_cls.return_value = MagicMock()
            mock_resolver_cls.return_value = MagicMock()
            mock_composer_cls.return_value = MagicMock()

            SupervisorAgent()

            mock_kb_cls.assert_called_once_with(token=None)
            mock_genie_cls.assert_called_once_with(token=None)
            mock_resolver_cls.assert_called_once_with(token=None)
            mock_composer_cls.assert_called_once_with(token=None)

    def test_injected_clients_not_overridden_by_user_token(self):
        """When clients are explicitly injected, user_token is ignored for them."""
        mock_kb = MagicMock()
        mock_genie = MagicMock()
        mock_kb.ask.return_value = "answer"
        mock_genie.ask.return_value = _make_genie_result(rows=[])

        agent = SupervisorAgent(
            kb_client=mock_kb,
            genie_client=mock_genie,
            resolver_client=None,
            user_token="dapi_should_be_ignored",
        )
        # The injected mocks are used directly — user_token didn't replace them
        assert agent._kb is mock_kb
        assert agent._genie is mock_genie
