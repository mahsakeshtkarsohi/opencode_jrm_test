"""
Plotly chart rendering for the JRM Media Advisor Streamlit app.

Takes a visualization spec dict (as returned by VisualizationSpec.model_dump())
and Genie result rows, and returns a Plotly Figure ready for st.plotly_chart().

Applies JUMBO_STYLE brand colours and formatting conventions defined in
src/jrm_advisor/visualization/spec.py.

Only the two supported chart patterns are rendered:
  - line:  x=year_week,      y=actual_sales              (currency)
  - bar:   x=year_week|week, y=sales_uplift_percentage   (percentage × 100)
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

# Jumbo brand palette — matches JUMBO_STYLE in spec.py.
_JUMBO_YELLOW = "#EEB717"
_JUMBO_BLUE = "#16A8D9"
_JUMBO_GREEN = "#0B8B32"
_JUMBO_RED = "#BA0000"
_BACKGROUND = "#FFFFFF"
_GRID = "#E3E3E3"
_TEXT = "#000000"
_AXIS = "#5E5E5E"


def _apply_jumbo_layout(fig: go.Figure, title: str, y_axis_title: str) -> go.Figure:
    """Apply shared Jumbo brand layout to a Plotly figure."""
    fig.update_layout(
        title={
            "text": title,
            "font": {"size": 16, "color": _TEXT, "family": "Arial, sans-serif"},
            "x": 0,
        },
        paper_bgcolor=_BACKGROUND,
        plot_bgcolor=_BACKGROUND,
        font={"family": "Arial, sans-serif", "color": _TEXT},
        xaxis={
            "title": {"text": "Week", "font": {"color": _AXIS}},
            "tickfont": {"color": _AXIS},
            "gridcolor": _GRID,
            "linecolor": _AXIS,
        },
        yaxis={
            "title": {"text": y_axis_title, "font": {"color": _AXIS}},
            "tickfont": {"color": _AXIS},
            "gridcolor": _GRID,
            "linecolor": _AXIS,
            "zeroline": True,
            "zerolinecolor": _AXIS,
        },
        margin={"l": 60, "r": 20, "t": 60, "b": 60},
        showlegend=False,
        hovermode="x unified",
    )
    return fig


def _format_y_values(
    values: list[float], y_format: str, scale: str, decimals: int
) -> list[float]:
    """Apply scale transformation to raw y values."""
    if scale == "x100":
        return [v * 100 for v in values]
    return list(values)


def _y_tick_format(y_format: str) -> str:
    """Return a Plotly tickformat string for the y axis."""
    if y_format == "percentage":
        return ".1f"
    if y_format == "currency":
        return ",.0f"
    return ""


def _y_hover_suffix(y_format: str) -> str:
    if y_format == "percentage":
        return " %"
    if y_format == "currency":
        return " €"
    return ""


def build_chart(
    spec: dict[str, Any],
    rows: list[dict[str, Any]],
) -> go.Figure | None:
    """Build a Plotly figure from a VisualizationSpec dict and Genie rows.

    Args:
        spec: Dict with keys: chart_type, x_field, y_field, y_format,
              decimals, scale. Matches VisualizationSpec.model_dump().
        rows: List of row dicts from GenieResult.rows.

    Returns:
        A Plotly Figure, or None if the spec says should_visualize=False
        or the rows are empty.
    """
    if not spec or not spec.get("should_visualize", True):
        return None
    if not rows:
        return None

    chart_type = spec.get("chart_type", "bar")
    x_field = spec["x_field"]
    y_field = spec["y_field"]
    y_format = spec.get("y_format", "number")
    scale = spec.get("scale", "identity")
    decimals = spec.get("decimals", 0)

    x_vals: list[str] = [str(row.get(x_field, "")) for row in rows]
    try:
        y_raw: list[float] = [float(row.get(y_field, 0) or 0) for row in rows]
    except (ValueError, TypeError):
        y_raw = [0.0] * len(rows)

    y_vals = _format_y_values(y_raw, y_format, scale, decimals)
    hover_suffix = _y_hover_suffix(y_format)

    if chart_type == "line":
        trace = go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines+markers",
            line={"color": _JUMBO_YELLOW, "width": 3},
            marker={"size": 7, "color": _JUMBO_YELLOW},
            hovertemplate=f"%{{x}}<br>%{{y:,.{decimals}f}}{hover_suffix}<extra></extra>",
        )
        y_title = (
            "Actual Sales (€)"
            if y_format == "currency"
            else y_field.replace("_", " ").title()
        )
        title = "Weekly Actual Sales"

    elif chart_type == "bar":
        colors = [_JUMBO_GREEN if v >= 0 else _JUMBO_RED for v in y_vals]
        trace = go.Bar(
            x=x_vals,
            y=y_vals,
            marker_color=colors,
            hovertemplate=f"%{{x}}<br>%{{y:.{decimals}f}}{hover_suffix}<extra></extra>",
        )
        y_title = "Sales Uplift (%)"
        title = "Weekly Sales Uplift (%)"

    else:
        return None

    fig = go.Figure(data=[trace])
    fig = _apply_jumbo_layout(fig, title=title, y_axis_title=y_title)

    if y_format == "currency":
        fig.update_yaxes(tickprefix="€", tickformat=_y_tick_format(y_format))
    elif y_format == "percentage":
        fig.update_yaxes(ticksuffix=" %", tickformat=_y_tick_format(y_format))

    return fig
