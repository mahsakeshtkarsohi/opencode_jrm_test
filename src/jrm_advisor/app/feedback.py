"""
Feedback writer for the JRM Media Advisor Streamlit app.

Writes thumbs-up / thumbs-down feedback from users to a Delta table in Unity
Catalog via the Databricks SDK Statement Execution API.

Table schema (auto-created if it does not exist):
    catalog.schema.jrm_advisor_feedback
    ─────────────────────────────────────
    feedback_id     STRING      UUID v4
    ts              TIMESTAMP   UTC submission time
    question        STRING      User's question
    answer_text     STRING      Agent's answer (truncated at 4000 chars)
    rating          STRING      'thumbs_up' | 'thumbs_down'
    comment         STRING      Optional free-text comment
    session_id      STRING      Streamlit session ID
    resolved_campaign STRING    CampaignNameAdj or NULL

The table path is read from environment variables:
    FEEDBACK_CATALOG   (default: dsa_development)
    FEEDBACK_SCHEMA    (default: retail_media)
    FEEDBACK_TABLE     (default: jrm_advisor_feedback)
    DATABRICKS_SQL_WAREHOUSE_ID  — required for writing
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Unity Catalog identifier validation (SEC-1)
# Only alphanumerics and underscores are valid UC identifier characters.
# Validated at module load so a misconfigured env var fails loudly at startup
# rather than silently writing DDL or data to the wrong table.
# ---------------------------------------------------------------------------

_UC_IDENT_RE = re.compile(r"^[a-zA-Z0-9_]+$")


def _validate_uc_ident(value: str, label: str) -> str:
    """Validate a Unity Catalog identifier (catalog, schema, or table name).

    Args:
        value: The identifier string to validate.
        label: Human-readable label used in the error message.

    Returns:
        The validated identifier unchanged.

    Raises:
        ValueError: If the identifier contains characters outside ``[a-zA-Z0-9_]``.
    """
    if not _UC_IDENT_RE.match(value):
        raise ValueError(
            f"Invalid {label} identifier {value!r}: only alphanumeric characters "
            "and underscores are permitted in Unity Catalog identifiers."
        )
    return value


_CATALOG = _validate_uc_ident(
    os.getenv("FEEDBACK_CATALOG", "dsa_development"), "catalog"
)
_SCHEMA = _validate_uc_ident(os.getenv("FEEDBACK_SCHEMA", "retail_media"), "schema")
_TABLE = _validate_uc_ident(
    os.getenv("FEEDBACK_TABLE", "jrm_advisor_feedback"), "table"
)
_FULL_TABLE = f"{_CATALOG}.{_SCHEMA}.{_TABLE}"

# Valid rating values — whitelist enforced in submit_feedback (SEC-3).
_VALID_RATINGS: frozenset[str] = frozenset({"thumbs_up", "thumbs_down"})


def _build_create_ddl(full_table: str) -> str:
    """Return the CREATE TABLE IF NOT EXISTS DDL for the feedback table."""
    return f"""
CREATE TABLE IF NOT EXISTS {full_table} (
    feedback_id        STRING        NOT NULL,
    ts                 TIMESTAMP     NOT NULL,
    question           STRING,
    answer_text        STRING,
    rating             STRING        NOT NULL,
    comment            STRING,
    session_id         STRING,
    resolved_campaign  STRING
)
USING DELTA
TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
"""


def _get_warehouse_id() -> str | None:
    return os.getenv("DATABRICKS_SQL_WAREHOUSE_ID")


def _get_ws_client(user_token: str | None = None):  # type: ignore[return]
    """Return a WorkspaceClient using the OBO user token or ambient SDK credentials.

    In production (Databricks App), ``user_token`` must be present — it is the
    OBO token forwarded via ``x-forwarded-access-token``.  Falling back to
    ambient credentials in production would allow writes under a service account
    with broader permissions than the actual user (SEC-2).

    In local development (``USE_MOCK_BACKEND=true`` or ``DATABRICKS_TOKEN`` set),
    ``user_token`` may be ``None`` and the SDK Config() auto-discovery is used.

    Args:
        user_token: OBO access token, or ``None`` for local dev fallback.

    Raises:
        ValueError: When ``user_token`` is absent and the app is not in local
            dev mode.  Caught by ``submit_feedback`` which returns ``False``.
    """
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.core import Config

    if user_token:
        host = os.getenv("DATABRICKS_HOST")
        return WorkspaceClient(host=host, token=user_token)

    # Allow ambient credentials in local dev mode only.
    if os.getenv("USE_MOCK_BACKEND", "false").lower() == "true" or os.getenv(
        "DATABRICKS_TOKEN"
    ):
        return WorkspaceClient(config=Config())

    raise ValueError(
        "_get_ws_client: no user_token provided and not in local dev mode. "
        "Ensure x-forwarded-access-token is forwarded by the Databricks App runtime."
    )


def _escape(s: str) -> str:
    """Sanitise a string for safe inclusion in a SQL string literal.

    Strips Unicode control characters (category Cc and Cf) then escapes
    single quotes and backslashes so the value is safe inside a
    single-quoted SQL string literal.
    """
    # Remove control characters (Cc = control, Cf = format/invisible)
    cleaned = "".join(c for c in s if unicodedata.category(c) not in {"Cc", "Cf"})
    # Escape SQL special characters
    return cleaned.replace("\\", "\\\\").replace("'", "''")


def submit_feedback(
    *,
    question: str,
    answer_text: str,
    rating: str,
    comment: str = "",
    session_id: str = "",
    resolved_campaign: str | None = None,
    user_token: str | None = None,
) -> bool:
    """Write a feedback record to the Delta feedback table.

    Args:
        question:           The user's original question.
        answer_text:        The agent's answer (will be truncated to 4000 chars).
        rating:             'thumbs_up' or 'thumbs_down'.
        comment:            Optional free-text comment from the user.
        session_id:         Streamlit session ID for grouping turns.
        resolved_campaign:  The resolved CampaignNameAdj, or None.
        user_token:         OBO access token from ``x-forwarded-access-token`` header.
            When provided, used instead of ambient SDK credentials so writes
            are attributed to the logged-in user's identity.

    Returns:
        True on success, False on any error (errors are logged, never raised
        so that feedback failures never interrupt the user's session).
    """
    # SEC-3: whitelist rating values — reject anything outside the known set.
    if rating not in _VALID_RATINGS:
        logger.warning(
            "submit_feedback: invalid rating=%r — must be one of %s; feedback not written",
            rating,
            _VALID_RATINGS,
        )
        return False

    warehouse_id = _get_warehouse_id()
    if not warehouse_id:
        logger.warning(
            "submit_feedback: DATABRICKS_SQL_WAREHOUSE_ID not set — feedback not written"
        )
        return False

    feedback_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    answer_truncated = answer_text[:4000]
    campaign_val = f"'{_escape(resolved_campaign)}'" if resolved_campaign else "NULL"

    insert_sql = f"""
INSERT INTO {_FULL_TABLE}
    (feedback_id, ts, question, answer_text, rating, comment, session_id, resolved_campaign)
VALUES (
    '{feedback_id}',
    CAST('{ts}' AS TIMESTAMP),
    '{_escape(question)}',
    '{_escape(answer_truncated)}',
    '{_escape(rating)}',
    '{_escape(comment)}',
    '{_escape(session_id)}',
    {campaign_val}
)
"""

    try:
        ws = _get_ws_client(user_token=user_token)
        # Ensure table exists
        ws.statement_execution.execute(
            warehouse_id=warehouse_id,
            statement=_build_create_ddl(_FULL_TABLE),
            wait_timeout="10s",
        )
        # Insert feedback row
        ws.statement_execution.execute(
            warehouse_id=warehouse_id,
            statement=insert_sql,
            wait_timeout="10s",
        )
        logger.info(
            "submit_feedback: wrote feedback_id=%s rating=%s", feedback_id, rating
        )
        return True
    except Exception as exc:
        logger.warning("submit_feedback: failed to write feedback — %s", exc)
        return False
