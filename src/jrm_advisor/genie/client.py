"""
Genie Client — JRM Media Advisor.

Wraps the Databricks Genie Conversation API so the supervisor can ask a natural-language
question and get back structured rows + the generated SQL.

Auth is handled entirely by the Databricks SDK (DATABRICKS_HOST + DATABRICKS_TOKEN env vars).
Never pass credentials as arguments.

Row parsing flow:
  start_conversation_and_wait()
      └─► GenieMessage.attachments[]
              └─► attachment.query  (GenieQueryAttachment)
                      └─► .statement_id
                              └─► get_message_query_result()
                                      └─► StatementResponse.manifest + .result
                                              └─► rows as list[dict]
"""

from __future__ import annotations

import logging
import os
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.dashboards import GenieAttachment, GenieMessage
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GenieError(Exception):
    """Raised when the Genie API returns an error or an unexpected response."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class GenieTimeoutError(GenieError):
    """Raised when Genie does not complete within the configured timeout."""


class GenieNoResultError(GenieError):
    """Raised when Genie responds with text only — no SQL/data attachment."""


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class GenieResult(BaseModel):
    """Structured result returned by GenieClient.ask()."""

    question: str
    sql: str
    rows: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class GenieClient:
    """Thin wrapper around the Databricks Genie Conversation API.

    Credentials and space ID are read from environment variables:
        DATABRICKS_HOST    — workspace URL
        DATABRICKS_TOKEN   — PAT or service principal OAuth token
        GENIE_SPACE_ID     — Genie Space to query

    All three are required. Missing variables raise ``ValueError`` at construction time.
    """

    def __init__(self) -> None:
        host = os.environ.get("DATABRICKS_HOST")
        token = os.environ.get("DATABRICKS_TOKEN")
        space_id = os.environ.get("GENIE_SPACE_ID")

        missing = [
            k
            for k, v in [
                ("DATABRICKS_HOST", host),
                ("DATABRICKS_TOKEN", token),
                ("GENIE_SPACE_ID", space_id),
            ]
            if not v
        ]
        if missing:
            raise ValueError(
                f"Missing required environment variable(s): {', '.join(missing)}. "
                "Copy .env.example to .env and fill in the values."
            )

        self._space_id: str = space_id  # type: ignore[assignment]
        self._ws = WorkspaceClient(host=host, token=token)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, question: str) -> GenieResult:
        """Send a natural-language question to Genie and return structured rows.

        Args:
            question: Plain-language question (e.g. "Show weekly sales uplift for
                the Coca-Cola campaign").

        Returns:
            ``GenieResult`` with ``rows``, ``sql``, and ``question``.

        Raises:
            GenieTimeoutError: Genie did not complete within the SDK timeout.
            GenieNoResultError: Genie returned a text-only response with no data.
            GenieError: Any other API failure.
        """
        logger.info("GenieClient.ask: sending question to space_id=%s", self._space_id)

        message = self._start_and_wait(question)
        sql, statement_id, conversation_id, message_id = self._extract_query_attachment(
            message, question
        )

        logger.debug("GenieClient.ask: generated SQL=%s", sql)

        rows = self._fetch_rows(
            conversation_id=conversation_id,
            message_id=message_id,
            statement_id=statement_id,
            sql=sql,
        )

        logger.info(
            "GenieClient.ask: completed — %d rows returned for question=%r",
            len(rows),
            question,
        )
        return GenieResult(question=question, sql=sql, rows=rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _start_and_wait(self, question: str) -> GenieMessage:
        """Start a Genie conversation and wait for completion."""
        try:
            message: GenieMessage = self._ws.genie.start_conversation_and_wait(
                space_id=self._space_id,
                content=question,
            )
        except TimeoutError as exc:
            raise GenieTimeoutError(
                f"Genie did not respond within the timeout for question={question!r}"
            ) from exc
        except Exception as exc:
            raise GenieError(
                f"Genie API error for question={question!r}: {exc}"
            ) from exc

        if message.error:
            raise GenieError(
                f"Genie returned an error: {message.error}",
            )

        return message

    def _extract_query_attachment(
        self,
        message: GenieMessage,
        question: str,
    ) -> tuple[str, str, str, str]:
        """Extract SQL, statement_id, conversation_id, message_id from the message.

        Returns:
            (sql, statement_id, conversation_id, message_id)

        Raises:
            GenieNoResultError: no query attachment found.
        """
        attachments: list[GenieAttachment] = message.attachments or []

        for attachment in attachments:
            if attachment.query and attachment.query.statement_id:
                sql = attachment.query.query or ""
                return (
                    sql,
                    attachment.query.statement_id,
                    message.conversation_id,
                    message.message_id,
                )

        # Text-only response — Genie answered in prose, no data returned
        text_preview = ""
        for attachment in attachments:
            if attachment.text and attachment.text.content:
                text_preview = attachment.text.content[:200]
                break

        raise GenieNoResultError(
            f"Genie returned no data attachment for question={question!r}. "
            f"Genie text response: {text_preview!r}"
        )

    def _fetch_rows(
        self,
        conversation_id: str,
        message_id: str,
        statement_id: str,
        sql: str,
    ) -> list[dict[str, Any]]:
        """Retrieve query result rows from the Genie statement result.

        Never logs row content — rows may contain campaign / retailer PII.
        """
        try:
            result_response = self._ws.genie.get_message_query_result(
                space_id=self._space_id,
                conversation_id=conversation_id,
                message_id=message_id,
            )
        except Exception as exc:
            raise GenieError(
                f"Failed to retrieve query result for statement_id={statement_id}: {exc}"
            ) from exc

        statement = result_response.statement_response
        if not statement or not statement.manifest or not statement.result:
            logger.debug(
                "GenieClient._fetch_rows: empty result set for statement_id=%s",
                statement_id,
            )
            return []

        # Build column name list from manifest schema
        columns: list[str] = [
            col.name for col in (statement.manifest.schema.columns or []) if col.name
        ]

        data_array = statement.result.data_array or []
        rows: list[dict[str, Any]] = []
        for raw_row in data_array:
            if raw_row is None:
                continue
            row = {col: val for col, val in zip(columns, raw_row)}
            rows.append(row)

        logger.debug(
            "GenieClient._fetch_rows: parsed %d rows with columns=%s",
            len(rows),
            columns,
        )
        return rows
