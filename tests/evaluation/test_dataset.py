"""
Unit tests for src/jrm_advisor/evaluation/dataset.py

Validates the structure and content of the gold evaluation dataset
without requiring any network calls.
"""

from __future__ import annotations


from jrm_advisor.evaluation.dataset import GOLD_DATASET

VALID_INTENTS = {"kb_only", "data_only", "hybrid", "out_of_scope"}


class TestGoldDatasetStructure:
    """Verify that every record follows the MLflow 3 evaluation schema."""

    def test_dataset_is_not_empty(self):
        assert len(GOLD_DATASET) > 0

    def test_all_records_have_inputs_key(self):
        for i, record in enumerate(GOLD_DATASET):
            assert "inputs" in record, f"Record {i} missing 'inputs'"

    def test_all_inputs_have_question(self):
        for i, record in enumerate(GOLD_DATASET):
            assert "question" in record["inputs"], (
                f"Record {i} 'inputs' missing 'question' key"
            )

    def test_all_questions_are_non_empty_strings(self):
        for i, record in enumerate(GOLD_DATASET):
            q = record["inputs"]["question"]
            assert isinstance(q, str) and q.strip(), (
                f"Record {i} question is empty or not a string"
            )

    def test_all_expectations_have_valid_intent(self):
        for i, record in enumerate(GOLD_DATASET):
            exp = record.get("expectations", {})
            intent = exp.get("expected_intent")
            assert intent in VALID_INTENTS, (
                f"Record {i} has invalid expected_intent: {intent!r}"
            )

    def test_expected_facts_are_lists_when_present(self):
        for i, record in enumerate(GOLD_DATASET):
            facts = record.get("expectations", {}).get("expected_facts")
            if facts is not None:
                assert isinstance(facts, list), (
                    f"Record {i} expected_facts must be a list, got {type(facts)}"
                )
                assert all(isinstance(f, str) for f in facts), (
                    f"Record {i} expected_facts must be list[str]"
                )

    def test_no_record_has_outputs_key(self):
        # Gold dataset should not have pre-computed outputs — the runner
        # will call predict_fn to generate them.
        for i, record in enumerate(GOLD_DATASET):
            assert "outputs" not in record, (
                f"Record {i} should not have pre-computed 'outputs'"
            )


class TestGoldDatasetCoverage:
    """Verify that all four routing categories are represented."""

    def _intents(self):
        return {r["expectations"]["expected_intent"] for r in GOLD_DATASET}

    def test_kb_only_questions_present(self):
        assert "kb_only" in self._intents()

    def test_data_only_questions_present(self):
        assert "data_only" in self._intents()

    def test_hybrid_questions_present(self):
        assert "hybrid" in self._intents()

    def test_out_of_scope_questions_present(self):
        assert "out_of_scope" in self._intents()

    def test_minimum_kb_only_count(self):
        count = sum(
            1 for r in GOLD_DATASET if r["expectations"]["expected_intent"] == "kb_only"
        )
        assert count >= 3, f"Need at least 3 kb_only questions, got {count}"

    def test_minimum_out_of_scope_count(self):
        count = sum(
            1
            for r in GOLD_DATASET
            if r["expectations"]["expected_intent"] == "out_of_scope"
        )
        assert count >= 3, f"Need at least 3 out_of_scope questions, got {count}"

    def test_total_question_count_reasonable(self):
        assert len(GOLD_DATASET) >= 10, (
            "Gold dataset should have at least 10 questions for meaningful evaluation"
        )
