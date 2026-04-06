"""
Unit tests for src/jrm_advisor/evaluation/scorers.py

All scorers are pure Python — no network calls, no MLflow server required.
The @scorer decorator is patched away so we test the underlying logic directly.
"""

from __future__ import annotations

from mlflow.entities import Feedback

from jrm_advisor.evaluation.scorers import (
    _infer_intent_from_response,
    clean_response,
    intent_routing_accuracy,
    response_not_empty,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _call_scorer(scorer_fn, **kwargs):
    """Call a scorer function that has been wrapped with @scorer.

    The @scorer decorator makes the function callable with keyword args
    matching (inputs, outputs, expectations, trace). We call it directly
    using the underlying __wrapped__ if available, otherwise call it
    normally with only the args it needs.
    """
    # MLflow's @scorer wraps the function but keeps it callable.
    # We pass only the args that the scorer signature expects.
    return scorer_fn(**kwargs)


# ===========================================================================
# response_not_empty
# ===========================================================================


class TestResponseNotEmpty:
    def test_passes_normal_text(self):
        fb = _call_scorer(response_not_empty, outputs={"text": "Sales uplift was 12%."})
        assert fb.value == "yes"

    def test_fails_empty_string(self):
        fb = _call_scorer(response_not_empty, outputs={"text": ""})
        assert fb.value == "no"

    def test_fails_whitespace_only(self):
        fb = _call_scorer(response_not_empty, outputs={"text": "   \n\t  "})
        assert fb.value == "no"

    def test_fails_missing_text_key(self):
        fb = _call_scorer(response_not_empty, outputs={})
        assert fb.value == "no"

    def test_passes_long_answer(self):
        long_text = "A" * 500
        fb = _call_scorer(response_not_empty, outputs={"text": long_text})
        assert fb.value == "yes"

    def test_returns_feedback_object(self):
        fb = _call_scorer(response_not_empty, outputs={"text": "Hello"})
        assert isinstance(fb, Feedback)

    def test_rationale_on_pass(self):
        fb = _call_scorer(response_not_empty, outputs={"text": "Some answer"})
        assert "text" in fb.rationale.lower()

    def test_rationale_on_fail(self):
        fb = _call_scorer(response_not_empty, outputs={"text": ""})
        assert "empty" in fb.rationale.lower()


# ===========================================================================
# clean_response
# ===========================================================================


class TestCleanResponse:
    """Tests for the artifact-detection scorer."""

    def _clean(self, text: str) -> Feedback:
        return _call_scorer(clean_response, outputs={"text": text})

    def test_passes_clean_prose(self):
        fb = self._clean(
            "The sales uplift for the Heineken campaign was 8.3% during week 12."
        )
        assert fb.value == "yes"

    def test_fails_html_tag(self):
        fb = self._clean("Here is <b>bold text</b> in the answer.")
        assert fb.value == "no"
        assert "HTML" in fb.rationale

    def test_fails_citation_marker(self):
        fb = self._clean("Sales increased by 10% [1] during the campaign.")
        assert fb.value == "no"
        assert "citation" in fb.rationale.lower()

    def test_fails_blob_url(self):
        fb = self._clean(
            "See https://mystorageaccount.blob.core.windows.net/container/file.pdf for details."
        )
        assert fb.value == "no"
        assert "blob" in fb.rationale.lower() or "storage" in fb.rationale.lower()

    def test_fails_internal_component_name_kb_client(self):
        fb = self._clean("Calling KnowledgeBaseClient to retrieve the answer.")
        assert fb.value == "no"
        assert "internal component" in fb.rationale.lower()

    def test_fails_internal_component_name_genie(self):
        fb = self._clean("GenieClient returned 0 rows.")
        assert fb.value == "no"

    def test_fails_internal_component_name_supervisor(self):
        fb = self._clean("SupervisorAgent routed to Genie.")
        assert fb.value == "no"

    def test_fails_trace_metadata(self):
        fb = self._clean("The span_id for this call is abc123.")
        assert fb.value == "no"

    def test_passes_answer_unavailable_sentinel(self):
        # The ANSWER_UNAVAILABLE sentinel is user-facing text — must pass
        fb = self._clean(
            "The information you requested is not available in the knowledge base."
        )
        assert fb.value == "yes"

    def test_multiple_violations_listed(self):
        text = "<b>bold</b> and [1] a citation and KnowledgeBaseClient."
        fb = self._clean(text)
        assert fb.value == "no"
        # rationale should mention multiple issues
        assert ";" in fb.rationale or "violation" in fb.rationale.lower()

    def test_returns_feedback_object(self):
        fb = self._clean("Clean text.")
        assert isinstance(fb, Feedback)


# ===========================================================================
# _infer_intent_from_response (internal helper)
# ===========================================================================


class TestInferIntentFromResponse:
    """Tests for the heuristic intent-inference fallback."""

    def test_out_of_scope_not_yet_supported(self):
        assert (
            _infer_intent_from_response(
                "Store-level ROI analysis is not yet supported in the JRM Media Advisor.",
                "out_of_scope",
            )
            == "out_of_scope"
        )

    def test_out_of_scope_not_currently_supported(self):
        assert (
            _infer_intent_from_response(
                "ROPO analysis is not yet stakeholder-aligned and is not currently supported.",
                "out_of_scope",
            )
            == "out_of_scope"
        )

    def test_out_of_scope_in_store_only(self):
        assert (
            _infer_intent_from_response(
                "The JRM Media Advisor currently covers in-store campaigns only.",
                "out_of_scope",
            )
            == "out_of_scope"
        )

    def test_data_only_table_present(self):
        result = _infer_intent_from_response(
            "week | actual_sales | baseline_sales\n--- \n1 | 1000 | 900",
            "data_only",
        )
        assert result == "data_only"

    def test_data_only_no_data_returned(self):
        result = _infer_intent_from_response(
            "No data was returned for this query.",
            "data_only",
        )
        assert result == "data_only"

    def test_kb_only_clean_prose(self):
        result = _infer_intent_from_response(
            "Sales uplift measures the incremental revenue attributable to the campaign "
            "above the expected baseline.",
            "kb_only",
        )
        assert result == "kb_only"

    def test_hybrid_both_markers(self):
        result = _infer_intent_from_response(
            "Sales uplift is the incremental revenue above baseline.\n\n"
            "week | actual_sales\n1 | 1200",
            "hybrid",
        )
        assert result == "hybrid"

    def test_unknown_for_empty_text(self):
        result = _infer_intent_from_response("", "kb_only")
        assert result == "unknown"


# ===========================================================================
# intent_routing_accuracy
# ===========================================================================


class TestIntentRoutingAccuracy:
    """Tests for the intent routing accuracy scorer."""

    def _call(self, inputs, outputs, expectations):
        return _call_scorer(
            intent_routing_accuracy,
            inputs=inputs,
            outputs=outputs,
            expectations=expectations,
        )

    def test_skips_when_no_expected_intent(self):
        fb = self._call(
            inputs={"question": "What is uplift?"},
            outputs={"text": "Uplift is incremental revenue."},
            expectations={},
        )
        assert fb.value == "skip"

    def test_passes_when_intent_field_matches(self):
        fb = self._call(
            inputs={"question": "What is uplift?"},
            outputs={"text": "Uplift is incremental revenue.", "intent": "kb_only"},
            expectations={"expected_intent": "kb_only"},
        )
        assert fb.value == "yes"

    def test_fails_when_intent_field_mismatches(self):
        fb = self._call(
            inputs={"question": "What is uplift?"},
            outputs={"text": "Uplift is incremental revenue.", "intent": "data_only"},
            expectations={"expected_intent": "kb_only"},
        )
        assert fb.value == "no"

    def test_infers_out_of_scope_when_no_intent_field(self):
        fb = self._call(
            inputs={"question": "Can I get store-level ROI?"},
            outputs={
                "text": "Store-level ROI analysis is not yet supported in the JRM Media Advisor."
            },
            expectations={"expected_intent": "out_of_scope"},
        )
        assert fb.value == "yes"

    def test_infers_data_only_from_table(self):
        fb = self._call(
            inputs={"question": "Show weekly sales."},
            outputs={"text": "week | actual_sales\n1 | 1000"},
            expectations={"expected_intent": "data_only"},
        )
        assert fb.value == "yes"

    def test_infers_kb_only_from_prose(self):
        fb = self._call(
            inputs={"question": "What is the baseline methodology?"},
            outputs={
                "text": (
                    "The baseline is calculated using pre-campaign sales data "
                    "adjusted for seasonality."
                )
            },
            expectations={"expected_intent": "kb_only"},
        )
        assert fb.value == "yes"

    def test_returns_feedback_object(self):
        fb = self._call(
            inputs={"question": "anything"},
            outputs={"text": "answer"},
            expectations={"expected_intent": "kb_only"},
        )
        assert isinstance(fb, Feedback)

    def test_rationale_includes_expected_and_actual(self):
        fb = self._call(
            inputs={"question": "What is uplift?"},
            outputs={"text": "Uplift is incremental revenue.", "intent": "data_only"},
            expectations={"expected_intent": "kb_only"},
        )
        assert "kb_only" in fb.rationale
        assert "data_only" in fb.rationale

    def test_none_expectations_skips(self):
        fb = self._call(
            inputs={"question": "anything"},
            outputs={"text": "answer"},
            expectations=None,
        )
        assert fb.value == "skip"
