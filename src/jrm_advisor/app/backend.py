"""
Backend interface for the JRM Media Advisor Streamlit app.

Wraps SupervisorAgent behind a simple ask() call.  The USE_MOCK_BACKEND
environment variable (default "true") switches between the real agent and a
deterministic mock so the UI can be developed and tested locally without a
live Databricks connection.

Usage::

    from jrm_advisor.app.backend import get_backend
    backend = get_backend()
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
# Real backend — delegates to SupervisorAgent.
# ---------------------------------------------------------------------------


class RealBackend:
    """Wraps SupervisorAgent for use in the Streamlit app."""

    def __init__(self) -> None:
        from jrm_advisor.supervisor.agent import SupervisorAgent

        self._agent = SupervisorAgent()

    def ask(self, question: str) -> AppResponse:
        response = self._agent.answer(question)
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


def get_backend() -> RealBackend:
    """Return a RealBackend.

    The USE_MOCK_BACKEND env var is read here so that the mock path can be
    imported in ``backend_mock.py`` without this module needing to know about
    it at import time.
    """
    if os.getenv("USE_MOCK_BACKEND", "false").lower() == "true":
        from jrm_advisor.app.backend_mock import MockBackend

        return MockBackend()  # type: ignore[return-value]
    return RealBackend()
