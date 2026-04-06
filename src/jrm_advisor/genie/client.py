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
                                      └─► StatementResponse.manifest.schema
                                              └─► column names only
                                                  (data_array is None for EXTERNAL
                                                   disposition results)
                              └─► statement_execution.get_statement_result_chunk_n()
                                      └─► ResultData.data_array  ← actual row data
                                              └─► rows as list[dict]
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.dashboards import GenieAttachment, GenieMessage
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# SEC-4: only allow well-formed Databricks workspace URLs so a misconfigured
# host cannot redirect the OBO token to an attacker-controlled endpoint.
_DATABRICKS_HOST_RE = re.compile(
    r"^https://[a-zA-Z0-9\-]+\.[0-9]+\.azuredatabricks\.net$"
    r"|^https://[a-zA-Z0-9\-\.]+\.azuredatabricks\.net$"
    r"|^https://[a-zA-Z0-9\-\.]+\.gcp\.databricks\.com$"
    r"|^https://[a-zA-Z0-9\-\.]+\.cloud\.databricks\.com$"
)


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

    def __init__(self, token: str | None = None) -> None:
        host = os.environ.get("DATABRICKS_HOST")
        env_token = os.environ.get("DATABRICKS_TOKEN")
        effective_token = token or env_token
        space_id = os.environ.get("GENIE_SPACE_ID")

        missing = [
            k
            for k, v in [
                ("DATABRICKS_HOST", host),
                ("authentication token", effective_token),
                ("GENIE_SPACE_ID", space_id),
            ]
            if not v
        ]
        if missing:
            raise ValueError(
                f"Missing required environment variable(s): {', '.join(missing)}. "
                "Copy .env.example to .env and fill in the values."
            )

        # SEC-4: validate host URL to prevent token leakage to wrong endpoint.
        if not _DATABRICKS_HOST_RE.match(host):  # type: ignore[arg-type]
            raise ValueError(
                f"DATABRICKS_HOST={host!r} is not a recognised Databricks workspace URL. "
                "Expected format: https://<workspace>.azuredatabricks.net"
            )

        self._space_id: str = space_id  # type: ignore[assignment]
        # auth_type="pat" prevents the SDK from picking up ambient
        # DATABRICKS_CLIENT_ID / DATABRICKS_CLIENT_SECRET env vars injected by
        # the Databricks App runtime, which would otherwise trigger a
        # "more than one authorization method configured" error.
        self._ws = WorkspaceClient(host=host, token=effective_token, auth_type="pat")

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
        """Retrieve query result rows for the given statement.

        The Genie message query result endpoint returns schema + metadata but
        the actual row data is served via the Statement Execution chunk API.
        We call get_statement_result_chunk_n for each chunk until exhausted.

        Never logs row content — rows may contain campaign / retailer PII.
        """
        # Step 1: get schema (column names) from the Genie result endpoint
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
        if not statement or not statement.manifest:
            logger.debug(
                "GenieClient._fetch_rows: no manifest for statement_id=%s",
                statement_id,
            )
            return []

        columns: list[str] = [
            col.name for col in (statement.manifest.schema.columns or []) if col.name
        ]
        total_rows = statement.manifest.total_row_count or 0

        if total_rows == 0:
            logger.debug(
                "GenieClient._fetch_rows: manifest reports 0 rows for statement_id=%s",
                statement_id,
            )
            return []

        # Step 2: fetch row data chunk by chunk via Statement Execution API.
        # The Genie result endpoint may return data_array=None when the result
        # uses EXTERNAL disposition; the chunk API always returns inline data.
        rows: list[dict[str, Any]] = []
        chunk_index = 0
        while chunk_index is not None:
            try:
                chunk = self._ws.statement_execution.get_statement_result_chunk_n(
                    statement_id=statement_id,
                    chunk_index=chunk_index,
                )
            except Exception as exc:
                raise GenieError(
                    f"Failed to fetch chunk {chunk_index} for statement_id={statement_id}: {exc}"
                ) from exc

            for raw_row in chunk.data_array or []:
                if raw_row is None:
                    continue
                rows.append({col: val for col, val in zip(columns, raw_row)})

            chunk_index = chunk.next_chunk_index  # None when last chunk

        logger.debug(
            "GenieClient._fetch_rows: parsed %d rows with columns=%s",
            len(rows),
            columns,
        )
        return rows
