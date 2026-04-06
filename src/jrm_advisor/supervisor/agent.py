"""
Supervisor Agent — JRM Media Advisor.

The supervisor is the single component that speaks to the user. It:
  1. Classifies the user's intent (rule-based keyword routing).
  2. Calls the appropriate sub-components (Knowledge Base, Genie, or both).
  3. Optionally attaches a visualization spec when Genie returns plottable rows.
  4. Composes a final, clean business-facing answer.

Routing rules (from AGENTS.md):
  ┌──────────────────────────────────────────────────────┬──────────────────────┐
  │ Intent                                               │ Components called    │
  ├──────────────────────────────────────────────────────┼──────────────────────┤
  │ Definition / terminology / methodology / KPI         │ Knowledge Base only  │
  │ Campaign data / uplift / sales / time-series         │ Genie only           │
  │ Hybrid (explain + measure, e.g. "what is uplift      │ Both                 │
  │   and how did mine perform?")                        │                      │
  │ Chart / trend requested AND rows available           │ + Visualization Spec │
  └──────────────────────────────────────────────────────┴──────────────────────┘

  Never call the Visualization Spec Function if there are no result rows.

Out-of-scope topics (AGENTS.md):
  store-level ROI, store-level OTS correlation, WAP / store drill-down,
  ROPO analysis, pre-launch forecasting, digital / OOH channels.

Final answer contract (AGENTS.md):
  - Professional plain business language.
  - Separate facts from interpretation.
  - Define JRM terms when helpful.
  - Recommend only when evidence supports it.
  - State clearly when data is missing or not supported.
  - Must NOT contain: raw HTML, citation markers, storage URLs, parser
    artifacts, internal component names, raw tool traces.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from jrm_advisor.genie.client import (
    GenieClient,
    GenieError,
    GenieNoResultError,
    GenieResult,
    GenieTimeoutError,
)
from jrm_advisor.knowledge_base.client import (
    ANSWER_UNAVAILABLE,
    KnowledgeBaseClient,
    KnowledgeBaseError,
    KnowledgeBaseTimeoutError,
)
from jrm_advisor.visualization.spec import (
    VisualizationSpec,
    build_visualization_spec,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent classification — keyword-based, deterministic, no LLM.
# ---------------------------------------------------------------------------

# Terms that strongly signal a definition / methodology / KPI question.
_KB_KEYWORDS: frozenset[str] = frozenset(
    {
        "what is",
        "what are",
        "define",
        "definition",
        "explain",
        "how is",
        "how are",
        "methodology",
        "calculated",
        "calculation",
        "measured",
        "measurement",
        "metric",
        "kpi",
        "terminology",
        "term",
        "meaning",
        "means",
        "uplift",
        "incremental",
        "baseline",
        "ots",
        "on-shelf",
        "on shelf",
        "wap",
        "ropo",
    }
)

# Terms that strongly signal a data / performance / campaign query.
_DATA_KEYWORDS: frozenset[str] = frozenset(
    {
        "show",
        "display",
        "give me",
        "list",
        "how much",
        "how many",
        "performance",
        "result",
        "results",
        "sales",
        "revenue",
        "weekly",
        "weekly sales",
        "trend",
        "over time",
        "campaign",
        "period",
        "week",
        "month",
        "quarter",
        "during",
        "pre",
        "post",
        "versus",
        "vs",
        "compare",
        "comparison",
        "top",
        "bottom",
        "highest",
        "lowest",
        "best",
        "worst",
        "store",
    }
)

# Visualization trigger words (only matter when Genie also returns rows).
_VIZ_KEYWORDS: frozenset[str] = frozenset(
    {
        "chart",
        "graph",
        "plot",
        "visuali",  # covers visualize / visualise / visualization
        "trend",
        "over time",
        "weekly trend",
    }
)

# Known out-of-scope topics — return a clear refusal rather than fabricating.
_OUT_OF_SCOPE_PATTERNS: list[tuple[str, str]] = [
    (
        r"\bstore[- ]level\s+(roi|return)",
        "Store-level ROI analysis is not yet supported in the JRM Media Advisor.",
    ),
    (
        r"\bots\b.*\bstore\b|\bstore\b.*\bots\b",
        "Store-level OTS vs. sales uplift correlation is not yet supported.",
    ),
    (
        r"\bwap\b|\bstore\s+drill",
        "WAP and store drill-down analysis are not yet supported.",
    ),
    (
        r"\bropo\b",
        "ROPO analysis is not yet stakeholder-aligned and is not currently supported.",
    ),
    (
        r"\bforecast\b|\bpre[- ]launch\b",
        "Pre-launch campaign forecasting is not yet supported.",
    ),
    (
        r"\bdigital\b|\bout[- ]of[- ]home\b|\booh\b",
        "The JRM Media Advisor currently covers in-store campaigns only. "
        "Digital and out-of-home channels are out of scope.",
    ),
]


class Intent:
    """Namespace for intent constants."""

    KB_ONLY = "kb_only"
    DATA_ONLY = "data_only"
    HYBRID = "hybrid"
    OUT_OF_SCOPE = "out_of_scope"


def classify_intent(question: str) -> str:
    """Classify the user question into a routing intent.

    Uses keyword matching only — no LLM calls.

    Args:
        question: Raw user question string.

    Returns:
        One of ``Intent.KB_ONLY``, ``Intent.DATA_ONLY``, ``Intent.HYBRID``,
        or ``Intent.OUT_OF_SCOPE``.
    """
    q = question.lower()

    # Check out-of-scope first — never attempt to answer these.
    for pattern, _ in _OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, q):
            logger.debug("classify_intent: matched out-of-scope pattern=%r", pattern)
            return Intent.OUT_OF_SCOPE

    kb_hit = any(kw in q for kw in _KB_KEYWORDS)
    data_hit = any(kw in q for kw in _DATA_KEYWORDS)

    if kb_hit and data_hit:
        intent = Intent.HYBRID
    elif kb_hit:
        intent = Intent.KB_ONLY
    elif data_hit:
        intent = Intent.DATA_ONLY
    else:
        # Default to KB for ambiguous questions — safer than calling Genie
        # on an unknown intent (Genie may fabricate).
        intent = Intent.KB_ONLY

    logger.debug(
        "classify_intent: question=%r → intent=%s (kb_hit=%s, data_hit=%s)",
        question[:80],
        intent,
        kb_hit,
        data_hit,
    )
    return intent


def wants_visualization(question: str) -> bool:
    """Return True if the user explicitly asked for a chart or trend view."""
    q = question.lower()
    return any(kw in q for kw in _VIZ_KEYWORDS)


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------


@dataclass
class SupervisorResponse:
    """Structured response returned by SupervisorAgent.answer().

    The ``text`` field is always ready for display to the user.
    ``visualization`` is set only when a chart should be rendered.
    ``genie_rows`` and ``genie_sql`` are available for downstream tooling
    (e.g. an application layer that wants to render a table) but must NOT
    be surfaced verbatim to the user.
    """

    text: str
    visualization: VisualizationSpec | dict[str, bool] | None = None
    genie_rows: list[dict[str, Any]] = field(default_factory=list)
    genie_sql: str = ""


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

# Formatting helpers (pure functions, easy to test independently).

_DATA_UNAVAILABLE = (
    "Campaign data is not available at this time. "
    "Please check that the Genie Space is configured and try again."
)

_DATA_TIMEOUT = (
    "The campaign data query timed out. "
    "Please try again — if the problem persists, contact your JRM media advisor."
)

_DATA_NO_RESULT = (
    "The campaign data query returned no results for this question. "
    "Please verify the campaign name and period, then try again."
)


def _format_genie_rows(rows: list[dict[str, Any]]) -> str:
    """Convert Genie result rows into a compact readable table string.

    Only used when the supervisor composes the data section of a hybrid
    or data-only answer. Returns an empty string for empty rows.
    """
    if not rows:
        return ""

    headers = list(rows[0].keys())
    lines: list[str] = []

    # Header row
    lines.append(" | ".join(headers))
    lines.append("-" * max(len(lines[0]), 20))

    # Data rows (cap at 50 to avoid overwhelming prose answers)
    for row in rows[:50]:
        lines.append(" | ".join(str(row.get(h, "")) for h in headers))

    if len(rows) > 50:
        lines.append(f"... ({len(rows) - 50} more rows)")

    return "\n".join(lines)


def _compose_answer(
    *,
    intent: str,
    kb_answer: str | None,
    genie_result: GenieResult | None,
    genie_error: str | None,
) -> str:
    """Compose the final user-facing answer from sub-component outputs.

    Follows the preferred answer structure from AGENTS.md:
      1. Main insight
      2. Supporting evidence
      3. Interpretation
      4. Recommendation (omitted if not supported)

    For definition/methodology questions:
      Definition → How it is measured → Known limitation.

    Args:
        intent:       Routing intent (Intent.*).
        kb_answer:    Clean prose from the Knowledge Base, or None.
        genie_result: Structured result from Genie, or None.
        genie_error:  Human-readable error string if Genie failed, or None.

    Returns:
        Final answer string ready for the user.
    """
    parts: list[str] = []

    # --- Definition / methodology section (KB) ---
    if kb_answer and kb_answer != ANSWER_UNAVAILABLE:
        parts.append(kb_answer)
    elif intent in (Intent.KB_ONLY, Intent.HYBRID) and (
        not kb_answer or kb_answer == ANSWER_UNAVAILABLE
    ):
        parts.append(ANSWER_UNAVAILABLE)

    # --- Campaign data section (Genie) ---
    if intent in (Intent.DATA_ONLY, Intent.HYBRID):
        if genie_result and genie_result.rows:
            table = _format_genie_rows(genie_result.rows)
            if table:
                parts.append(table)
        elif genie_error:
            parts.append(genie_error)
        elif genie_result and not genie_result.rows:
            parts.append(
                "No data was returned for this query. "
                "Please verify the campaign name and time period."
            )

    return "\n\n".join(parts).strip()


class SupervisorAgent:
    """Supervisor agent — orchestrates KB, Genie, and Visualization components.

    The supervisor is the single entry point for all user questions. It
    classifies intent, calls the appropriate sub-components, and returns a
    clean ``SupervisorResponse`` containing business-ready prose and an
    optional chart spec.

    Sub-components are injected at construction to allow test mocking without
    patching internals. When called with no arguments, production clients are
    instantiated automatically from environment variables.

    Args:
        kb_client:    Knowledge Base client. Defaults to ``KnowledgeBaseClient()``.
        genie_client: Genie client. Defaults to ``GenieClient()``.

    Example::

        agent = SupervisorAgent()
        response = agent.answer("What is sales uplift?")
        print(response.text)
    """

    def __init__(
        self,
        kb_client: KnowledgeBaseClient | None = None,
        genie_client: GenieClient | None = None,
    ) -> None:
        self._kb = kb_client if kb_client is not None else KnowledgeBaseClient()
        self._genie = genie_client if genie_client is not None else GenieClient()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(self, question: str) -> SupervisorResponse:
        """Answer a user question end-to-end.

        Classifies intent, routes to sub-components, composes a clean
        business-facing response. Never surfaces raw tool output.

        Args:
            question: Plain-language question from the user.

        Returns:
            ``SupervisorResponse`` with ``text`` ready for display and an
            optional ``visualization`` spec for chart rendering.
        """
        logger.info("SupervisorAgent.answer: question=%r", question[:120])

        intent = classify_intent(question)
        logger.info("SupervisorAgent.answer: intent=%s", intent)

        # --- Out-of-scope: return clear refusal immediately ---
        if intent == Intent.OUT_OF_SCOPE:
            refusal = self._out_of_scope_message(question)
            logger.info("SupervisorAgent.answer: returning out-of-scope refusal")
            return SupervisorResponse(text=refusal)

        kb_answer: str | None = None
        genie_result: GenieResult | None = None
        genie_error: str | None = None

        # --- Knowledge Base call ---
        if intent in (Intent.KB_ONLY, Intent.HYBRID):
            kb_answer = self._call_kb(question)

        # --- Genie call ---
        if intent in (Intent.DATA_ONLY, Intent.HYBRID):
            genie_result, genie_error = self._call_genie(question)

        # --- Compose text answer ---
        text = _compose_answer(
            intent=intent,
            kb_answer=kb_answer,
            genie_result=genie_result,
            genie_error=genie_error,
        )

        # --- Visualization spec (only when rows exist and user wants a chart) ---
        viz: VisualizationSpec | dict[str, bool] | None = None
        rows = genie_result.rows if genie_result else []
        if rows and wants_visualization(question):
            viz = build_visualization_spec(rows, user_requested_chart=True)
            logger.debug(
                "SupervisorAgent.answer: visualization=%s",
                getattr(viz, "chart_type", "none"),
            )

        logger.info("SupervisorAgent.answer: completed, text_length=%d", len(text))
        return SupervisorResponse(
            text=text,
            visualization=viz,
            genie_rows=rows,
            genie_sql=genie_result.sql if genie_result else "",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_kb(self, question: str) -> str:
        """Call the Knowledge Base and return clean prose or ANSWER_UNAVAILABLE."""
        try:
            answer = self._kb.ask(question)
            logger.debug("SupervisorAgent._call_kb: received answer")
            return answer
        except KnowledgeBaseTimeoutError:
            logger.warning("SupervisorAgent._call_kb: timeout")
            return (
                "The knowledge base did not respond in time. Please try again shortly."
            )
        except KnowledgeBaseError as exc:
            logger.warning("SupervisorAgent._call_kb: error — %s", exc)
            return ANSWER_UNAVAILABLE

    def _call_genie(self, question: str) -> tuple[GenieResult | None, str | None]:
        """Call Genie and return (result, error_string).

        Always returns a tuple — never raises. The caller decides how to
        incorporate an error into the final answer.
        """
        try:
            result = self._genie.ask(question)
            logger.debug(
                "SupervisorAgent._call_genie: %d rows returned", len(result.rows)
            )
            return result, None
        except GenieTimeoutError:
            logger.warning("SupervisorAgent._call_genie: timeout")
            return None, _DATA_TIMEOUT
        except GenieNoResultError:
            logger.warning(
                "SupervisorAgent._call_genie: no result (text-only response)"
            )
            return None, _DATA_NO_RESULT
        except GenieError as exc:
            logger.warning("SupervisorAgent._call_genie: error — %s", exc)
            return None, _DATA_UNAVAILABLE

    @staticmethod
    def _out_of_scope_message(question: str) -> str:
        """Return a specific refusal message for out-of-scope questions."""
        q = question.lower()
        for pattern, message in _OUT_OF_SCOPE_PATTERNS:
            if re.search(pattern, q):
                return message
        # Fallback (should not be reached given classify_intent logic)
        return (
            "This question is outside the current scope of the JRM Media Advisor. "
            "Please contact your JRM media advisor for assistance."
        )
