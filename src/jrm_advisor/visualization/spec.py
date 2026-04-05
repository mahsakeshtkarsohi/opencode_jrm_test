"""
Visualization Spec Function — JRM Media Advisor.

Rule-based, deterministic function that decides whether a chart should be shown
and returns a structured spec for the application layer to render. No LLM calls.

Supported charts (exhaustive — nothing else):
  - line:  x=year_week,        y=actual_sales              (currency, identity)
  - bar:   x=year_week|week,   y=sales_uplift_percentage   (percentage, x100)

The application layer is responsible for rendering and applying JUMBO_STYLE.
See ADR-003 for design rationale.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Jumbo brand style — defined here, applied by the application layer.
# ---------------------------------------------------------------------------

JUMBO_STYLE: dict[str, Any] = {
    "palette": [
        "#EEB717",  # Jumbo yellow (primary)
        "#16A8D9",  # blue
        "#0B8B32",  # green
        "#BA0000",  # red
        "#5E5E5E",  # dark gray
        "#F59B00",  # orange
        "#7B2D8E",  # purple
        "#00838F",  # teal
    ],
    "font_family": "Jumbo TheSans, -apple-system, BlinkMacSystemFont, Arial, sans-serif",
    "background": "#FFFFFF",
    "grid_color": "#E3E3E3",
    "text_color": "#000000",
    "axis_color": "#5E5E5E",
    "highlight_color": "#EEB717",
}

# ---------------------------------------------------------------------------
# Supported chart pattern definitions.
# Each pattern is checked in order; the first match wins.
# ---------------------------------------------------------------------------

_SUPPORTED_PATTERNS: list[dict[str, Any]] = [
    {
        "chart_type": "line",
        "x_field": "year_week",
        "y_field": "actual_sales",
        "y_format": "currency",
        "decimals": 0,
        "scale": "identity",
    },
    {
        "chart_type": "bar",
        "x_field": "year_week",
        "y_field": "sales_uplift_percentage",
        "y_format": "percentage",
        "decimals": 1,
        "scale": "x100",
    },
    {
        "chart_type": "bar",
        "x_field": "week",
        "y_field": "sales_uplift_percentage",
        "y_format": "percentage",
        "decimals": 1,
        "scale": "x100",
    },
]

# Sentinel returned when no chart should be shown.
NO_VISUALIZATION: dict[str, bool] = {"should_visualize": False}


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------


class VisualizationSpec(BaseModel):
    """Structured chart specification returned to the application layer."""

    should_visualize: bool = True
    chart_type: str
    x_field: str
    y_field: str
    y_format: str
    decimals: int
    scale: str
    style_preset: str = "jumbo"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_visualization_spec(
    rows: list[dict[str, Any]],
    user_requested_chart: bool = False,  # noqa: ARG001 — reserved for future gating
) -> VisualizationSpec | dict[str, bool]:
    """Decide whether to visualize and return the appropriate spec.

    Args:
        rows: Result rows returned by Genie. Each row is a dict of column→value.
        user_requested_chart: True when the user explicitly asked for a chart or
            trend visualization. Currently does not change the matching logic —
            the function visualizes whenever the column pattern matches — but is
            kept as a parameter for future gating (e.g. suppress auto-charts).

    Returns:
        A ``VisualizationSpec`` if the rows match a supported pattern,
        or ``{"should_visualize": False}`` otherwise.
    """
    if not rows:
        logger.debug("build_visualization_spec: no rows — returning no visualization")
        return NO_VISUALIZATION

    # Collect the union of all column names across all rows to handle sparse sets.
    columns: frozenset[str] = frozenset().union(*(row.keys() for row in rows))
    logger.debug("build_visualization_spec: detected columns=%s", sorted(columns))

    for pattern in _SUPPORTED_PATTERNS:
        if pattern["x_field"] in columns and pattern["y_field"] in columns:
            logger.debug(
                "build_visualization_spec: matched pattern chart_type=%s x=%s y=%s",
                pattern["chart_type"],
                pattern["x_field"],
                pattern["y_field"],
            )
            return VisualizationSpec(**pattern)

    logger.debug(
        "build_visualization_spec: no pattern matched for columns=%s — returning no visualization",
        sorted(columns),
    )
    return NO_VISUALIZATION
