"""
Tests for jrm_advisor.app.charts — Plotly chart builder.

No Databricks connection required. Uses in-process data only.
"""

from __future__ import annotations

import plotly.graph_objects as go
import pytest

from jrm_advisor.app.charts import build_chart


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def line_spec() -> dict:
    return {
        "should_visualize": True,
        "chart_type": "line",
        "x_field": "year_week",
        "y_field": "actual_sales",
        "y_format": "currency",
        "decimals": 0,
        "scale": "identity",
        "style_preset": "jumbo",
    }


@pytest.fixture()
def bar_spec() -> dict:
    return {
        "should_visualize": True,
        "chart_type": "bar",
        "x_field": "year_week",
        "y_field": "sales_uplift_percentage",
        "y_format": "percentage",
        "decimals": 1,
        "scale": "x100",
        "style_preset": "jumbo",
    }


@pytest.fixture()
def line_rows() -> list[dict]:
    return [
        {"year_week": "202401", "actual_sales": "48320"},
        {"year_week": "202402", "actual_sales": "51740"},
        {"year_week": "202403", "actual_sales": "55180"},
    ]


@pytest.fixture()
def bar_rows() -> list[dict]:
    return [
        {"year_week": "202501", "sales_uplift_percentage": "0.082"},
        {"year_week": "202502", "sales_uplift_percentage": "0.114"},
        {"year_week": "202503", "sales_uplift_percentage": "-0.021"},
    ]


# ---------------------------------------------------------------------------
# build_chart — return types
# ---------------------------------------------------------------------------


class TestBuildChartReturnType:
    def test_line_spec_returns_figure(self, line_spec, line_rows):
        fig = build_chart(line_spec, line_rows)
        assert isinstance(fig, go.Figure)

    def test_bar_spec_returns_figure(self, bar_spec, bar_rows):
        fig = build_chart(bar_spec, bar_rows)
        assert isinstance(fig, go.Figure)

    def test_should_visualize_false_returns_none(self, line_spec, line_rows):
        line_spec["should_visualize"] = False
        assert build_chart(line_spec, line_rows) is None

    def test_empty_rows_returns_none(self, line_spec):
        assert build_chart(line_spec, []) is None

    def test_none_spec_returns_none(self, line_rows):
        assert build_chart(None, line_rows) is None  # type: ignore[arg-type]

    def test_unknown_chart_type_returns_none(self, line_spec, line_rows):
        line_spec["chart_type"] = "scatter_3d"
        assert build_chart(line_spec, line_rows) is None


# ---------------------------------------------------------------------------
# build_chart — line chart properties
# ---------------------------------------------------------------------------


class TestLineChart:
    def test_trace_is_scatter(self, line_spec, line_rows):
        fig = build_chart(line_spec, line_rows)
        assert isinstance(fig.data[0], go.Scatter)

    def test_x_values_match_rows(self, line_spec, line_rows):
        fig = build_chart(line_spec, line_rows)
        assert list(fig.data[0].x) == ["202401", "202402", "202403"]

    def test_y_values_match_rows_no_scale(self, line_spec, line_rows):
        fig = build_chart(line_spec, line_rows)
        # identity scale — values unchanged
        assert list(fig.data[0].y) == [48320.0, 51740.0, 55180.0]

    def test_title_set(self, line_spec, line_rows):
        fig = build_chart(line_spec, line_rows)
        assert fig.layout.title.text  # non-empty

    def test_background_is_white(self, line_spec, line_rows):
        fig = build_chart(line_spec, line_rows)
        assert fig.layout.paper_bgcolor == "#FFFFFF"


# ---------------------------------------------------------------------------
# build_chart — bar chart properties
# ---------------------------------------------------------------------------


class TestBarChart:
    def test_trace_is_bar(self, bar_spec, bar_rows):
        fig = build_chart(bar_spec, bar_rows)
        assert isinstance(fig.data[0], go.Bar)

    def test_x_values_match_rows(self, bar_spec, bar_rows):
        fig = build_chart(bar_spec, bar_rows)
        assert list(fig.data[0].x) == ["202501", "202502", "202503"]

    def test_y_values_scaled_x100(self, bar_spec, bar_rows):
        fig = build_chart(bar_spec, bar_rows)
        y = list(fig.data[0].y)
        # 0.082 * 100 = 8.2, 0.114 * 100 = 11.4, -0.021 * 100 = -2.1
        assert abs(y[0] - 8.2) < 0.01
        assert abs(y[1] - 11.4) < 0.01
        assert abs(y[2] - (-2.1)) < 0.01

    def test_negative_bar_coloured_red(self, bar_spec, bar_rows):
        fig = build_chart(bar_spec, bar_rows)
        colors = list(fig.data[0].marker.color)
        # Third bar is negative → red
        assert colors[2] == "#BA0000"

    def test_positive_bars_coloured_green(self, bar_spec, bar_rows):
        fig = build_chart(bar_spec, bar_rows)
        colors = list(fig.data[0].marker.color)
        assert colors[0] == "#0B8B32"
        assert colors[1] == "#0B8B32"


# ---------------------------------------------------------------------------
# build_chart — week field aliasing (x_field='week' also supported)
# ---------------------------------------------------------------------------


class TestWeekFieldAliasing:
    def test_week_field_bar_chart(self):
        spec = {
            "should_visualize": True,
            "chart_type": "bar",
            "x_field": "week",
            "y_field": "sales_uplift_percentage",
            "y_format": "percentage",
            "decimals": 1,
            "scale": "x100",
            "style_preset": "jumbo",
        }
        rows = [
            {"week": "1", "sales_uplift_percentage": "0.05"},
            {"week": "2", "sales_uplift_percentage": "0.08"},
        ]
        fig = build_chart(spec, rows)
        assert isinstance(fig, go.Figure)
        assert list(fig.data[0].x) == ["1", "2"]
