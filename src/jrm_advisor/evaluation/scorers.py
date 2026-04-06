"""
Custom MLflow 3 GenAI scorers for the JRM Media Advisor evaluation.

Scorer dimensions aligned with AGENTS.md:
  - clean_response          : rule-based; checks for artifacts that must never appear
  - response_not_empty      : guard scorer; fails if text is blank
  - intent_routing_accuracy : trace-based; verifies the supervisor routed correctly

Usage::

    from jrm_advisor.evaluation.scorers import (
        clean_response,
        response_not_empty,
        intent_routing_accuracy,
    )
    from mlflow.genai.scorers import Guidelines, Correctness

    scorers = [
        response_not_empty,
        clean_response,
        intent_routing_accuracy,
        Correctness(),
        Guidelines(
            name="business_language",
            guidelines=(
                "The response must be written in professional, plain business language. "
                "It must not contain internal system names, raw tool output, or technical "
                "implementation details."
            ),
        ),
        Guidelines(
            name="completeness",
            guidelines=(
                "The response must fully address the request. "
                "If data is unavailable the response must clearly say so rather than "
                "returning a vague or empty answer."
            ),
        ),
    ]
"""

from __future__ import annotations

import re

from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer

# ---------------------------------------------------------------------------
# Artifact patterns that must never appear in a user-facing answer.
# These mirror the "Every answer must NOT contain" rules in AGENTS.md.
# ---------------------------------------------------------------------------
_ARTIFACT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"<[a-z][^>]{0,200}>", re.IGNORECASE), "raw HTML tag"),
    (re.compile(r"\[\d+\]"), "citation marker (e.g. [1])"),
    (
        re.compile(r"https?://[^\s]+\.(blob|core\.windows|dfs\.core)", re.IGNORECASE),
        "storage URL / blob link",
    ),
    (
        re.compile(
            r"\b(KnowledgeBaseClient|GenieClient|SupervisorAgent|classify_intent"
            r"|build_visualization_spec|_call_kb|_call_genie)\b"
        ),
        "internal component name",
    ),
    (re.compile(r"span_id|trace_id|run_id", re.IGNORECASE), "execution metadata"),
]


# ---------------------------------------------------------------------------
# Scorer: response_not_empty
# ---------------------------------------------------------------------------


@scorer
def response_not_empty(outputs: dict) -> Feedback:
    """Fail if the response text is blank or whitespace-only.

    This is a guard scorer — if it fails, the other scorers are unreliable.
    """
    text = outputs.get("text", "") if isinstance(outputs, dict) else str(outputs)
    has_content = bool(text and text.strip())
    return Feedback(
        value="yes" if has_content else "no",
        rationale=(
            "Response contains text."
            if has_content
            else "Response is empty or whitespace-only."
        ),
    )


# ---------------------------------------------------------------------------
# Scorer: clean_response
# ---------------------------------------------------------------------------


@scorer
def clean_response(outputs: dict) -> Feedback:
    """Check that the response contains no raw artifacts.

    Enforces the AGENTS.md rule: answers must NOT contain raw HTML,
    citation markers, storage URLs, internal component names, or
    execution metadata.

    Returns a single Feedback with a list of violations (if any) in the
    rationale.
    """
    text = outputs.get("text", "") if isinstance(outputs, dict) else str(outputs)

    violations: list[str] = []
    for pattern, description in _ARTIFACT_PATTERNS:
        if pattern.search(text):
            violations.append(description)

    if violations:
        return Feedback(
            value="no",
            rationale="Artifact violations found: " + "; ".join(violations),
        )
    return Feedback(
        value="yes",
        rationale="No artifact violations detected.",
    )


# ---------------------------------------------------------------------------
# Scorer: intent_routing_accuracy
# ---------------------------------------------------------------------------


@scorer
def intent_routing_accuracy(
    inputs: dict,
    outputs: dict,
    expectations: dict,
) -> Feedback:
    """Verify the supervisor routed the question to the correct intent.

    Requires ``expectations.expected_intent`` in the evaluation dataset.
    Reads the routing decision from ``outputs.intent`` when present.

    If ``expected_intent`` is missing from expectations this scorer
    returns a 'skip' value so it does not contaminate aggregate metrics.
    """
    expected = (expectations or {}).get("expected_intent")

    if not expected:
        return Feedback(
            name="intent_routing_accuracy",
            value="skip",
            rationale="No expected_intent provided in expectations — scorer skipped.",
        )

    actual = (outputs or {}).get("intent")

    if actual is None:
        # SupervisorResponse does not currently expose intent directly;
        # infer it heuristically from the response content.
        text = (outputs or {}).get("text", "")
        actual = _infer_intent_from_response(text, expected)

    is_correct = actual == expected

    return Feedback(
        name="intent_routing_accuracy",
        value="yes" if is_correct else "no",
        rationale=f"Expected intent '{expected}', inferred '{actual}'.",
    )


def _infer_intent_from_response(text: str, expected: str) -> str:
    """Heuristic fallback to infer routing from the response text.

    Used when the SupervisorResponse does not carry an explicit intent
    field.  This is intentionally conservative — when uncertain it
    returns 'unknown' so the scorer does not silently pass.
    """
    t = text.lower()

    # Out-of-scope answers always contain "not yet supported" or "out of scope"
    out_of_scope_markers = [
        "not yet supported",
        "not currently supported",
        "out of scope",
        "in-store campaigns only",
    ]
    if any(m in t for m in out_of_scope_markers):
        return "out_of_scope"

    # Genie data answers typically contain a table (pipe separator) or
    # the standard "no data" / "campaign data unavailable" message
    genie_markers = [" | ", "no data was returned", "campaign data is not available"]
    has_genie_content = any(m in t for m in genie_markers)

    # KB prose: present when the text is long and NOT the unavailable sentinel.
    # We use 50+ chars of non-table prose to distinguish KB content from
    # short Genie error messages.  Extract the non-table portion first.
    kb_unavailable = "the information you requested is not available"
    # Strip out lines that look like table rows (contain " | ") before measuring
    non_table_lines = [
        ln for ln in t.splitlines() if " | " not in ln and "---" not in ln
    ]
    non_table_text = " ".join(non_table_lines).strip()
    has_kb_content = kb_unavailable not in t and len(non_table_text) >= 50

    if has_kb_content and has_genie_content:
        return "hybrid"
    if has_genie_content:
        return "data_only"
    if has_kb_content:
        return "kb_only"

    return "unknown"
