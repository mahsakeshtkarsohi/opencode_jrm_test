"""
Backend interface for the JRM Media Advisor Streamlit app.

Wraps SupervisorAgent behind a simple ask() call.

Auth model (on Databricks Apps):
  The app uses on-behalf-of (OBO) user auth. The logged-in user's access token
  is forwarded by the Databricks App runtime in the ``x-forwarded-access-token``
  HTTP header. ``app.py`` extracts it and passes it here as ``user_token``.
  All Databricks API calls (Genie, KB, resolver, feedback) are made with that
  token — so data access is gated by the user's own Unity Catalog permissions.
  No service principal or PAT is required.

Local development:
  Set USE_MOCK_BACKEND=true to use the deterministic mock without any
  Databricks connection.  When running against a real workspace locally,
  set DATABRICKS_TOKEN in your .env file — it is used as the fallback token
  when no user token is provided.

Usage::

    from jrm_advisor.app.backend import AppResponse, get_backend
    backend = get_backend(user_token="dapi...")
    result = backend.ask("What is sales uplift?")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Response model — shared between real and mock backends.
# ---------------------------------------------------------------------------


@dataclass
class AppResponse:
    """Normalised response returned to the Streamlit app layer.

    Fields mirror the fields the UI cares about from SupervisorResponse, but
    are expressed as plain Python types so the app never imports supervisor
    internals directly.
    """

    text: str
    needs_clarification: bool = False
    resolved_campaign_name: str | None = None
    genie_rows: list[dict[str, Any]] = field(default_factory=list)
    genie_sql: str = ""
    # Visualization spec as a plain dict (or None when no chart needed).
    # Keys: should_visualize, chart_type, x_field, y_field, y_format, decimals, scale
    visualization: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Real backend — delegates to SupervisorAgent with per-request user token.
# ---------------------------------------------------------------------------


class RealBackend:
    """Wraps SupervisorAgent for use in the Streamlit app.

    A new SupervisorAgent is constructed per ``ask()`` call so that the
    user token (which changes per session) is always fresh.  The agent and
    its sub-clients are lightweight — construction cost is negligible
    compared to the network round-trips they make.

    Args:
        user_token: The on-behalf-of user access token extracted from the
            ``x-forwarded-access-token`` header.  Falls back to the
            ``DATABRICKS_TOKEN`` env var when not provided (local dev).
    """

    def __init__(self, user_token: str | None = None) -> None:
        self._token = user_token or os.getenv("DATABRICKS_TOKEN", "")

    def ask(self, question: str) -> AppResponse:
        from jrm_advisor.supervisor.agent import SupervisorAgent

        agent = SupervisorAgent(user_token=self._token)
        response = agent.answer(question)
        viz: dict[str, Any] | None = None
        if response.visualization is not None:
            if hasattr(response.visualization, "model_dump"):
                viz = response.visualization.model_dump()
            elif isinstance(response.visualization, dict):
                viz = response.visualization
        return AppResponse(
            text=response.text,
            needs_clarification=response.needs_clarification,
            resolved_campaign_name=response.resolved_campaign_name,
            genie_rows=response.genie_rows,
            genie_sql=response.genie_sql,
            visualization=viz,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_backend(user_token: str | None = None) -> RealBackend:
    """Return a backend instance.

    Args:
        user_token: OBO access token from ``x-forwarded-access-token`` header.
            Ignored when USE_MOCK_BACKEND=true.

    The USE_MOCK_BACKEND env var switches to the deterministic mock so the UI
    can be developed locally without a live Databricks connection.
    """
    if os.getenv("USE_MOCK_BACKEND", "false").lower() == "true":
        from jrm_advisor.app.backend_mock import MockBackend

        return MockBackend()  # type: ignore[return-value]
    return RealBackend(user_token=user_token)
