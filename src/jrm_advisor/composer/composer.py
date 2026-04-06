"""
Answer Composer — JRM Media Advisor.

Replaces the rule-based ``_compose_answer()`` concatenation in the supervisor
with an LLM-based answer that follows the AGENTS.md answer contract:

    1. Main insight
    2. Supporting evidence
    3. Interpretation
    4. Recommendation (only when evidence supports it)

For definition/methodology questions:
    Definition → How it is measured → Known limitation.

The composer calls a Databricks Foundation Model API endpoint via the same
``urllib``-based HTTP client pattern used by ``KnowledgeBaseClient``.

Auth
----
``DATABRICKS_HOST`` and ``DATABRICKS_TOKEN`` env vars (shared with the stack).
``COMPOSER_ENDPOINT`` — full invocation URL for the foundation model endpoint,
e.g. ``https://adb-<id>.azuredatabricks.net/serving-endpoints/<name>/invocations``.

Graceful degradation
--------------------
If the LLM call fails for any reason, ``compose()`` falls back to the
rule-based concatenation from ``_compose_answer()``.  The caller is unaware
of whether the LLM or fallback path was taken — both return a ``str``.

Genie row cap
-------------
At most ``MAX_ROWS_IN_PROMPT`` rows are injected into the prompt.  The full
row list is still returned in ``SupervisorResponse.genie_rows`` for the
application layer to render as a table independently.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

import mlflow

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 30
MAX_ROWS_IN_PROMPT = 20

# ---------------------------------------------------------------------------
# Exceptions (internal — callers receive a fallback string, not an exception)
# ---------------------------------------------------------------------------


class AnswerComposerError(Exception):
    """Raised internally when the composer LLM call fails."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AnswerComposerTimeoutError(AnswerComposerError):
    """Raised when the composer endpoint does not respond in time."""


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a JRM (Jumbo Retail Media) Media Advisor. You answer questions about \
in-store retail media campaigns in professional, plain business language.

Answer rules (mandatory):
- Structure every answer as: Main insight → Supporting evidence → Interpretation \
→ Recommendation (omit if not supported by evidence).
- For definition/methodology questions use: Definition → How it is measured → \
Known limitation.
- Write in clear business prose. Do NOT dump raw tables, raw data arrays, or \
column headers into the answer — summarise the data in sentences.
- Separate facts from interpretation explicitly (e.g. "The data shows …; \
this suggests …").
- Define JRM-specific terms (uplift, baseline, OTS, WAP) the first time you use them.
- Only make a recommendation when the data directly supports it. Do not guess.
- If data is partially available, state what is known and what is missing.
- Do NOT include: raw HTML, footnotes, citation markers [1] [2], storage URLs, \
blob links, internal component names, or system metadata.
- Current in-scope channel: in-store campaigns only. Do not speculate about \
digital or OOH channels.
"""

# ---------------------------------------------------------------------------
# Fallback (same logic as supervisor._compose_answer before this component)
# ---------------------------------------------------------------------------


def _fallback_compose(
    *,
    intent: str,
    kb_answer: str | None,
    genie_rows: list[dict[str, Any]] | None,
    genie_sql: str,
) -> str:
    """Rule-based fallback used when the LLM endpoint is unavailable."""
    from jrm_advisor.knowledge_base.client import ANSWER_UNAVAILABLE
    from jrm_advisor.supervisor.agent import Intent, _format_genie_rows

    parts: list[str] = []

    if kb_answer and kb_answer != ANSWER_UNAVAILABLE:
        parts.append(kb_answer)
    elif intent in (Intent.KB_ONLY, Intent.HYBRID) and (
        not kb_answer or kb_answer == ANSWER_UNAVAILABLE
    ):
        parts.append(ANSWER_UNAVAILABLE)

    if intent in (Intent.DATA_ONLY, Intent.HYBRID):
        if genie_rows:
            table = _format_genie_rows(genie_rows)
            if table:
                parts.append(table)
        elif not genie_rows:
            parts.append(
                "No data was returned for this query. "
                "Please verify the campaign name and time period."
            )

    return "\n\n".join(parts).strip()


# ---------------------------------------------------------------------------
# AnswerComposer
# ---------------------------------------------------------------------------


class AnswerComposer:
    """LLM-based answer composer for the JRM Media Advisor supervisor.

    Builds a structured, business-facing prose answer from Knowledge Base
    and Genie sub-component outputs.  Falls back to rule-based concatenation
    if the LLM endpoint is unavailable.

    Args:
        endpoint_url: Full invocation URL of the foundation model endpoint.
            Defaults to the ``COMPOSER_ENDPOINT`` env var.
        token:        Databricks PAT or OAuth token.
            Defaults to the ``DATABRICKS_TOKEN`` env var.
        timeout:      HTTP timeout in seconds (default 30).

    Example::

        composer = AnswerComposer()
        answer = composer.compose(
            question="What is uplift and how did the Heineken campaign perform?",
            intent="hybrid",
            kb_answer="Sales uplift is the incremental revenue ...",
            genie_rows=[{"year_week": "202501", "sales_uplift_percentage": "0.11"}],
            genie_sql="SELECT ...",
        )
    """

    def __init__(
        self,
        endpoint_url: str | None = None,
        token: str | None = None,
        timeout: int = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._endpoint_url = endpoint_url or os.environ.get("COMPOSER_ENDPOINT", "")
        self._token = token or os.environ.get("DATABRICKS_TOKEN", "")
        self._timeout = timeout

        if not self._endpoint_url:
            logger.warning(
                "AnswerComposer: COMPOSER_ENDPOINT not set — will use fallback only"
            )
        if not self._token:
            logger.warning(
                "AnswerComposer: DATABRICKS_TOKEN not set — LLM call will fail"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @mlflow.trace(name="answer_composer", span_type="LLM")
    def compose(
        self,
        *,
        question: str,
        intent: str,
        kb_answer: str | None,
        genie_rows: list[dict[str, Any]] | None,
        genie_sql: str,
    ) -> str:
        """Compose a business-facing answer using the LLM endpoint.

        Falls back to rule-based concatenation if the endpoint is unavailable
        or returns an unusable response.

        Args:
            question:   The original user question.
            intent:     Routing intent (``Intent.*`` constant string).
            kb_answer:  Prose from the Knowledge Base, or None.
            genie_rows: List of dicts from Genie, or None.
            genie_sql:  SQL executed by Genie (for context only).

        Returns:
            Business-facing prose answer ready for display to the user.
        """
        if not self._endpoint_url or not self._token:
            logger.info(
                "AnswerComposer.compose: no endpoint/token configured — using fallback"
            )
            return _fallback_compose(
                intent=intent,
                kb_answer=kb_answer,
                genie_rows=genie_rows,
                genie_sql=genie_sql,
            )

        try:
            return self._call_llm(
                question=question,
                intent=intent,
                kb_answer=kb_answer,
                genie_rows=genie_rows,
                genie_sql=genie_sql,
            )
        except Exception as exc:
            logger.warning(
                "AnswerComposer.compose: LLM call failed (%s) — using fallback", exc
            )
            return _fallback_compose(
                intent=intent,
                kb_answer=kb_answer,
                genie_rows=genie_rows,
                genie_sql=genie_sql,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_user_message(
        self,
        *,
        question: str,
        intent: str,
        kb_answer: str | None,
        genie_rows: list[dict[str, Any]] | None,
        genie_sql: str,
    ) -> str:
        """Build the user-turn message for the LLM prompt."""
        lines: list[str] = [f"User question: {question}", f"Intent: {intent}"]

        if kb_answer:
            lines.append(f"\nKnowledge Base answer:\n{kb_answer}")

        if genie_rows:
            capped = genie_rows[:MAX_ROWS_IN_PROMPT]
            remainder = len(genie_rows) - len(capped)
            rows_json = json.dumps(capped, ensure_ascii=False, indent=2)
            lines.append(f"\nCampaign data ({len(capped)} rows):\n{rows_json}")
            if remainder > 0:
                lines.append(
                    f"({remainder} additional rows omitted — summarise from the rows above)"
                )

        lines.append(
            "\nWrite a complete, business-facing answer following the answer rules "
            "in your system prompt. Do not include the word 'Intent' or any "
            "internal labels in your answer."
        )
        return "\n".join(lines)

    def _call_llm(
        self,
        *,
        question: str,
        intent: str,
        kb_answer: str | None,
        genie_rows: list[dict[str, Any]] | None,
        genie_sql: str,
    ) -> str:
        """POST to the foundation model endpoint and extract the answer text.

        Raises:
            AnswerComposerTimeoutError: HTTP timeout.
            AnswerComposerError: HTTP error or JSON decode failure.
        """
        user_message = self._build_user_message(
            question=question,
            intent=intent,
            kb_answer=kb_answer,
            genie_rows=genie_rows,
            genie_sql=genie_sql,
        )

        payload = json.dumps(
            {
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ]
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            url=self._endpoint_url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as response:
                body = response.read().decode("utf-8")
        except TimeoutError as exc:
            raise AnswerComposerTimeoutError(
                f"Composer endpoint timed out after {self._timeout}s"
            ) from exc
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise AnswerComposerError(
                f"Composer endpoint returned HTTP {exc.code}: {body[:500]}",
                status_code=exc.code,
            ) from exc
        except urllib.error.URLError as exc:
            raise AnswerComposerError(
                f"Composer endpoint unreachable: {exc.reason}"
            ) from exc

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise AnswerComposerError(
                f"Composer endpoint returned non-JSON body: {body[:500]}"
            ) from exc

        return self._extract_answer(data)

    def _extract_answer(self, response: dict[str, Any]) -> str:
        """Extract answer text from an OpenAI-compatible chat completions envelope.

        Expected shape::

            {"choices": [{"message": {"content": "<answer>"}}]}

        Raises ``AnswerComposerError`` on unexpected shape or empty content
        so the ``compose()`` method can activate the fallback path.
        """
        try:
            choices = response.get("choices") or []
            if not choices:
                raise AnswerComposerError(
                    "Composer endpoint returned no choices in response"
                )
            content: str = choices[0].get("message", {}).get("content", "").strip()
            if not content:
                raise AnswerComposerError("Composer endpoint returned empty content")
            return content
        except AnswerComposerError:
            raise
        except (AttributeError, IndexError, TypeError) as exc:
            raise AnswerComposerError(
                f"Composer endpoint returned unexpected response shape: {exc}"
            ) from exc
