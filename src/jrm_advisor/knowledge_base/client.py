"""
Knowledge Base Client — JRM Media Advisor.

Routes definition, terminology, methodology, and KPI questions to a Databricks
Knowledge Assistant (KA) agent deployed as a Model Serving endpoint.

Answer contract (from AGENTS.md — Knowledge Base Behavior Rules):
  - Return clean prose summaries only.
  - NEVER surface raw document fragments, HTML, citation markers ([1], [2]),
    storage URLs, blob links, or parser artifacts.
  - If methodology is only partially available, state what is known and what
    is missing — do not fabricate.

Auth flow:
  - DATABRICKS_HOST  — workspace URL (shared with the rest of the stack)
  - DATABRICKS_TOKEN — PAT or service principal OAuth token (shared)
  - KB_ENDPOINT_URL  — full invocation URL of the KA Model Serving endpoint
                       e.g. https://adb-<id>.azuredatabricks.net/serving-endpoints/<name>/invocations

All three are required. Missing variables raise ``ValueError`` at construction time.

Request/response wire format:
  POST KB_ENDPOINT_URL
  Authorization: Bearer <DATABRICKS_TOKEN>
  Content-Type:  application/json

  Body:
    {"messages": [{"role": "user", "content": "<question>"}]}

  Expected response:
    {"choices": [{"message": {"content": "<answer>"}}]}
    — standard OpenAI-compatible chat completions envelope used by all
      Databricks Model Serving endpoints, including Knowledge Assistants.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class KnowledgeBaseError(Exception):
    """Raised when the Knowledge Base endpoint returns an error or unexpected response."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class KnowledgeBaseTimeoutError(KnowledgeBaseError):
    """Raised when the Knowledge Base endpoint does not respond within the configured timeout."""


class KnowledgeBaseEmptyResponseError(KnowledgeBaseError):
    """Raised when the endpoint returns a well-formed response with no usable answer text."""


# ---------------------------------------------------------------------------
# Sentinel for failed answers — returned instead of raising so the supervisor
# can compose a graceful user-facing message.
# ---------------------------------------------------------------------------

ANSWER_UNAVAILABLE = (
    "This information is not currently available in the JRM knowledge base. "
    "Please contact your JRM media advisor for assistance."
)

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT_SECONDS = 30


class KnowledgeBaseClient:
    """Thin client that calls a Databricks Knowledge Assistant (KA) Model Serving endpoint.

    Credentials and endpoint URL are read from environment variables:
        DATABRICKS_HOST    — workspace URL (e.g. https://adb-<id>.azuredatabricks.net)
        DATABRICKS_TOKEN   — PAT or service principal OAuth token
        KB_ENDPOINT_URL    — full invocation URL for the KA endpoint

    All three are required. Missing variables raise ``ValueError`` at construction time.

    Example::

        client = KnowledgeBaseClient()
        answer = client.ask("What is sales uplift and how is it measured?")
    """

    def __init__(
        self, timeout: int = _DEFAULT_TIMEOUT_SECONDS, token: str | None = None
    ) -> None:
        host = os.environ.get("DATABRICKS_HOST")
        env_token = os.environ.get("DATABRICKS_TOKEN")
        effective_token = token or env_token
        endpoint_url = os.environ.get("KB_ENDPOINT_URL")

        missing = [
            k
            for k, v in [
                ("DATABRICKS_HOST", host),
                ("DATABRICKS_TOKEN (or user token)", effective_token),
                ("KB_ENDPOINT_URL", endpoint_url),
            ]
            if not v
        ]
        if missing:
            raise ValueError(
                f"Missing required environment variable(s): {', '.join(missing)}. "
                "Copy .env.example to .env and fill in the values."
            )

        self._token: str = effective_token  # type: ignore[assignment]
        self._endpoint_url: str = endpoint_url  # type: ignore[assignment]
        self._timeout = timeout

        logger.debug(
            "KnowledgeBaseClient initialised with endpoint_url=%s", self._endpoint_url
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, question: str) -> str:
        """Ask a definition, methodology, or KPI question to the Knowledge Base.

        The returned string is clean business prose — no citations, no HTML,
        no document fragments. If the endpoint returns an unusable response
        the method returns ``ANSWER_UNAVAILABLE`` rather than raising, so the
        supervisor can still compose a graceful user-facing message.

        Args:
            question: Plain-language question directed at the Knowledge Base,
                e.g. "What is sales uplift and how is it calculated?".

        Returns:
            A prose answer string suitable for direct inclusion in the final
            supervisor response.

        Raises:
            KnowledgeBaseTimeoutError: The endpoint did not respond within ``timeout`` seconds.
            KnowledgeBaseError: Any non-timeout HTTP or network error.
        """
        logger.info(
            "KnowledgeBaseClient.ask: sending question to endpoint=%s",
            self._endpoint_url,
        )

        raw_answer = self._call_endpoint(question)
        answer = self._extract_answer(raw_answer)

        logger.info(
            "KnowledgeBaseClient.ask: received answer (length=%d chars)", len(answer)
        )
        return answer

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_endpoint(self, question: str) -> dict[str, Any]:
        """POST the question to the Model Serving endpoint.

        Returns:
            Parsed JSON response dict.

        Raises:
            KnowledgeBaseTimeoutError: On urllib timeout.
            KnowledgeBaseError: On HTTP errors or JSON decode failures.
        """
        payload = json.dumps(
            {"messages": [{"role": "user", "content": question}]}
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
            raise KnowledgeBaseTimeoutError(
                f"Knowledge Base endpoint timed out after {self._timeout}s "
                f"for question={question!r}"
            ) from exc
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise KnowledgeBaseError(
                f"Knowledge Base endpoint returned HTTP {exc.code}: {body[:500]}",
                status_code=exc.code,
            ) from exc
        except urllib.error.URLError as exc:
            raise KnowledgeBaseError(
                f"Knowledge Base endpoint unreachable: {exc.reason}"
            ) from exc

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise KnowledgeBaseError(
                f"Knowledge Base endpoint returned non-JSON body: {body[:500]}"
            ) from exc

    def _extract_answer(self, response: dict[str, Any]) -> str:
        """Extract clean answer text from the OpenAI-compatible response envelope.

        Expected shape::

            {
                "choices": [
                    {"message": {"content": "<answer text>"}}
                ]
            }

        Falls back to ``ANSWER_UNAVAILABLE`` if the shape is unexpected or
        the answer text is empty.

        Never logs the answer content — it may contain proprietary methodology.
        """
        try:
            choices = response.get("choices") or []
            if not choices:
                logger.warning(
                    "KnowledgeBaseClient._extract_answer: response contained no choices"
                )
                return ANSWER_UNAVAILABLE

            content: str = choices[0].get("message", {}).get("content", "").strip()

            if not content:
                logger.warning(
                    "KnowledgeBaseClient._extract_answer: answer content was empty"
                )
                return ANSWER_UNAVAILABLE

            return content

        except (AttributeError, IndexError, TypeError) as exc:
            logger.warning(
                "KnowledgeBaseClient._extract_answer: unexpected response shape — %s",
                exc,
            )
            return ANSWER_UNAVAILABLE
