"""
Campaign Resolver — JRM Media Advisor.

Resolves a user's natural-language campaign reference to the exact
``CampaignNameAdj`` value stored in the campaign data tables, using the
Databricks Vector Search UC function ``dsa_development.retail_media.search_campaigns``.

Why this exists
---------------
The supervisor passes questions to Genie, which requires an exact campaign name
match in the underlying data. When the user says "Heineken" but the real campaign
name is "Heineken Pilsner W4 2024 — Display", Genie returns zero rows silently.
This client resolves that ambiguity before the Genie call.

Resolution flow
---------------
1. Call ``search_campaigns(query)`` via the Databricks SDK statement execution API.
2. Parse the result rows into ``CampaignMatch`` objects.
3. Return the top match when its score exceeds ``CAMPAIGN_RESOLVER_THRESHOLD``
   (env var, default 0.7).
4. Return ``CampaignResolution`` with ``is_ambiguous=True`` when the top-2 scores
   are within ``CAMPAIGN_RESOLVER_AMBIGUITY_GAP`` (env var, default 0.05).
5. Return ``CampaignResolution`` with ``match=None`` when no result exceeds the
   threshold — caller falls back to the raw question.

The ``score`` field is **internal** and must never be surfaced to the user.

Auth
----
``DATABRICKS_HOST`` and ``DATABRICKS_TOKEN`` env vars (shared with the rest of
the stack). ``DATABRICKS_SQL_WAREHOUSE_ID`` is required to execute the UC function
via the statement execution API.

UC function signature
---------------------
``dsa_development.retail_media.search_campaigns(query STRING) → TABLE``

Returns columns: ``page_content`` (CampaignNameAdj), ``metadata`` (MAP<STRING,STRING>)
where metadata keys: article_cmp_id, ArticleName, Brand, SegmentName,
PresentationGroup, CMName, score.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLD = 0.7
_DEFAULT_AMBIGUITY_GAP = 0.05
_UC_FUNCTION = "dsa_development.retail_media.search_campaigns"
_STATEMENT_TIMEOUT_SECONDS = 30
# Non-terminal states indicate the statement did not complete within wait_timeout.
# The installed SDK has no TIMEDOUT member; PENDING or RUNNING signal a timeout.
# CANCELED and CLOSED are also terminal — treat them as errors, not timeouts.
_TERMINAL_STATES: frozenset[StatementState] = frozenset(
    {
        StatementState.SUCCEEDED,
        StatementState.FAILED,
        StatementState.CANCELED,
        StatementState.CLOSED,
    }
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CampaignResolverError(Exception):
    """Raised when the campaign resolver encounters an unrecoverable error."""


class CampaignResolverTimeoutError(CampaignResolverError):
    """Raised when the statement execution times out."""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class CampaignMatch:
    """A single candidate campaign returned by ``search_campaigns``.

    Attributes:
        name:     The exact ``CampaignNameAdj`` value from the data table.
        score:    Similarity score from Vector Search (0–1). Internal use only.
        metadata: Raw metadata map from the UC function. Internal use only.
    """

    name: str
    score: float
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class CampaignResolution:
    """Result of a campaign name resolution attempt.

    Attributes:
        match:        Best ``CampaignMatch`` if score ≥ threshold, else ``None``.
        candidates:   All matches above threshold (up to 5).
        is_ambiguous: True when top-2 scores are within ``ambiguity_gap``.
        raw_query:    The original query string passed to the resolver.
    """

    match: CampaignMatch | None
    candidates: list[CampaignMatch] = field(default_factory=list)
    is_ambiguous: bool = False
    raw_query: str = ""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class CampaignResolverClient:
    """Resolves fuzzy campaign name references to exact ``CampaignNameAdj`` values.

    Credentials and configuration are read from environment variables:
        DATABRICKS_HOST                  — workspace URL
        DATABRICKS_TOKEN                 — PAT or service principal OAuth token
        DATABRICKS_SQL_WAREHOUSE_ID      — SQL warehouse for statement execution
        CAMPAIGN_RESOLVER_THRESHOLD      — min score to accept a match (default 0.7)
        CAMPAIGN_RESOLVER_AMBIGUITY_GAP  — max gap between top-2 scores for ambiguity
                                           (default 0.05)

    The first three variables are required. Missing variables raise ``ValueError``
    at construction time.

    Example::

        client = CampaignResolverClient()
        resolution = client.resolve("Show Heineken campaign results")
        if resolution.is_ambiguous:
            # ask user to clarify
            ...
        elif resolution.match:
            # use resolution.match.name in the Genie question
            ...
        else:
            # no match — fall back to raw question
            ...
    """

    def __init__(self) -> None:
        host = os.environ.get("DATABRICKS_HOST")
        token = os.environ.get("DATABRICKS_TOKEN")
        warehouse_id = os.environ.get("DATABRICKS_SQL_WAREHOUSE_ID")

        missing = [
            k
            for k, v in [
                ("DATABRICKS_HOST", host),
                ("DATABRICKS_TOKEN", token),
                ("DATABRICKS_SQL_WAREHOUSE_ID", warehouse_id),
            ]
            if not v
        ]
        if missing:
            raise ValueError(
                f"Missing required environment variable(s): {', '.join(missing)}. "
                "Copy .env.example to .env and fill in the values."
            )

        self._warehouse_id: str = warehouse_id  # type: ignore[assignment]
        self._ws = WorkspaceClient(host=host, token=token)

        self._threshold: float = float(
            os.environ.get("CAMPAIGN_RESOLVER_THRESHOLD", str(_DEFAULT_THRESHOLD))
        )
        self._ambiguity_gap: float = float(
            os.environ.get(
                "CAMPAIGN_RESOLVER_AMBIGUITY_GAP", str(_DEFAULT_AMBIGUITY_GAP)
            )
        )

        logger.debug(
            "CampaignResolverClient initialised: threshold=%.2f ambiguity_gap=%.2f",
            self._threshold,
            self._ambiguity_gap,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, query: str) -> CampaignResolution:
        """Resolve a natural-language query to a campaign name.

        Calls the ``search_campaigns`` UC function and returns a
        ``CampaignResolution`` describing the best match (or ambiguity).

        Args:
            query: The user's question or a portion of it containing a
                campaign reference, e.g. "Show Heineken results".

        Returns:
            ``CampaignResolution`` with ``match``, ``candidates``,
            ``is_ambiguous``, and ``raw_query`` set.

        Raises:
            CampaignResolverTimeoutError: Statement execution timed out.
            CampaignResolverError: Unrecoverable error from the SQL warehouse.
        """
        logger.info("CampaignResolverClient.resolve: query=%r", query[:120])

        raw_matches = self._execute_search(query)
        above_threshold = [m for m in raw_matches if m.score >= self._threshold]

        if not above_threshold:
            logger.info(
                "CampaignResolverClient.resolve: no matches above threshold=%.2f",
                self._threshold,
            )
            return CampaignResolution(match=None, candidates=[], raw_query=query)

        # Check for ambiguity between top-2 results
        is_ambiguous = (
            len(above_threshold) >= 2
            and (above_threshold[0].score - above_threshold[1].score)
            < self._ambiguity_gap
        )

        best = above_threshold[0]
        logger.info(
            "CampaignResolverClient.resolve: best=%r score=%.3f ambiguous=%s",
            best.name,
            best.score,
            is_ambiguous,
        )

        return CampaignResolution(
            match=best if not is_ambiguous else None,
            candidates=above_threshold,
            is_ambiguous=is_ambiguous,
            raw_query=query,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_search(self, query: str) -> list[CampaignMatch]:
        """Execute the ``search_campaigns`` UC function and parse results.

        Returns:
            List of ``CampaignMatch`` objects sorted descending by score.
            Empty list if no rows returned or statement fails.
        """
        # Escape single quotes in query to prevent SQL injection
        safe_query = query.replace("'", "''")
        sql = f"SELECT * FROM {_UC_FUNCTION}('{safe_query}')"

        try:
            response = self._ws.statement_execution.execute_statement(
                warehouse_id=self._warehouse_id,
                statement=sql,
                wait_timeout=f"{_STATEMENT_TIMEOUT_SECONDS}s",
            )
        except Exception as exc:
            raise CampaignResolverError(
                f"Failed to execute search_campaigns: {exc}"
            ) from exc

        state = response.status.state if response.status else None
        if state == StatementState.FAILED:
            error_msg = (
                response.status.error.message
                if response.status and response.status.error
                else "unknown error"
            )
            raise CampaignResolverError(
                f"search_campaigns statement failed: {error_msg}"
            )
        if state in (StatementState.CANCELED, StatementState.CLOSED):
            raise CampaignResolverError(
                f"search_campaigns statement ended unexpectedly (state={state!r}) "
                f"for query={query!r}"
            )
        # The SDK returns PENDING or RUNNING when wait_timeout expires before the
        # statement completes (there is no TIMEDOUT state in the installed SDK).
        if state not in _TERMINAL_STATES:
            raise CampaignResolverTimeoutError(
                f"search_campaigns timed out after {_STATEMENT_TIMEOUT_SECONDS}s "
                f"(state={state!r}) for query={query!r}"
            )

        return self._parse_rows(response)

    def _parse_rows(self, response) -> list[CampaignMatch]:
        """Parse statement execution response into ``CampaignMatch`` objects."""
        manifest = response.manifest
        result = response.result

        if not manifest or not manifest.schema or not manifest.schema.columns:
            logger.debug("CampaignResolverClient._parse_rows: no schema in response")
            return []

        if not result or not result.data_array:
            logger.debug("CampaignResolverClient._parse_rows: no data_array in result")
            return []

        columns = [col.name for col in manifest.schema.columns]
        matches: list[CampaignMatch] = []

        for row in result.data_array:
            row_dict = dict(zip(columns, row))
            name = row_dict.get("page_content", "")
            metadata_raw = row_dict.get("metadata") or {}

            # metadata is a MAP<STRING,STRING> — SDK may return as dict or str
            if isinstance(metadata_raw, str):
                logger.warning(
                    "CampaignResolverClient._parse_rows: metadata returned as "
                    "string (not dict) — score will be 0.0 for this row"
                )
                metadata: dict[str, str] = {}
            else:
                metadata = {str(k): str(v) for k, v in metadata_raw.items()}

            score_str = metadata.get("score", "0")
            try:
                score = float(score_str)
            except (ValueError, TypeError):
                score = 0.0

            if name:
                matches.append(CampaignMatch(name=name, score=score, metadata=metadata))

        # Sort descending by score
        matches.sort(key=lambda m: m.score, reverse=True)
        logger.debug(
            "CampaignResolverClient._parse_rows: %d matches parsed", len(matches)
        )
        return matches
