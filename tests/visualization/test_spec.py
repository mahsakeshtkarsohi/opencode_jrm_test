"""
Unit tests for src/jrm_advisor/visualization/spec.py.

All acceptance criteria from issue #002 are covered here.
Pure function — no mocking required.
"""

from jrm_advisor.visualization.spec import (
    JUMBO_STYLE,
    VisualizationSpec,
    build_visualization_spec,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rows(*column_names: str) -> list[dict]:
    """Build a minimal single-row result set with the given column names."""
    return [{col: None for col in column_names}]


# ---------------------------------------------------------------------------
# AC: Given rows with year_week + actual_sales → line chart spec
# ---------------------------------------------------------------------------


class TestLineChart:
    def test_returns_visualization_spec(self):
        result = build_visualization_spec(_rows("year_week", "actual_sales"))
        assert isinstance(result, VisualizationSpec)

    def test_should_visualize_true(self):
        result = build_visualization_spec(_rows("year_week", "actual_sales"))
        assert result.should_visualize is True

    def test_chart_type_is_line(self):
        result = build_visualization_spec(_rows("year_week", "actual_sales"))
        assert result.chart_type == "line"

    def test_x_field(self):
        result = build_visualization_spec(_rows("year_week", "actual_sales"))
        assert result.x_field == "year_week"

    def test_y_field(self):
        result = build_visualization_spec(_rows("year_week", "actual_sales"))
        assert result.y_field == "actual_sales"

    def test_y_format_is_currency(self):
        result = build_visualization_spec(_rows("year_week", "actual_sales"))
        assert result.y_format == "currency"

    def test_scale_is_identity(self):
        result = build_visualization_spec(_rows("year_week", "actual_sales"))
        assert result.scale == "identity"

    def test_extra_columns_still_match(self):
        rows = _rows("year_week", "actual_sales", "campaign_id", "store_id")
        result = build_visualization_spec(rows)
        assert isinstance(result, VisualizationSpec)
        assert result.chart_type == "line"


# ---------------------------------------------------------------------------
# AC: Given rows with year_week + sales_uplift_percentage → bar chart spec
# ---------------------------------------------------------------------------


class TestBarChartYearWeek:
    def test_returns_visualization_spec(self):
        result = build_visualization_spec(_rows("year_week", "sales_uplift_percentage"))
        assert isinstance(result, VisualizationSpec)

    def test_chart_type_is_bar(self):
        result = build_visualization_spec(_rows("year_week", "sales_uplift_percentage"))
        assert result.chart_type == "bar"

    def test_x_field(self):
        result = build_visualization_spec(_rows("year_week", "sales_uplift_percentage"))
        assert result.x_field == "year_week"

    def test_y_field(self):
        result = build_visualization_spec(_rows("year_week", "sales_uplift_percentage"))
        assert result.y_field == "sales_uplift_percentage"

    def test_y_format_is_percentage(self):
        result = build_visualization_spec(_rows("year_week", "sales_uplift_percentage"))
        assert result.y_format == "percentage"

    def test_scale_is_x100(self):
        result = build_visualization_spec(_rows("year_week", "sales_uplift_percentage"))
        assert result.scale == "x100"


# ---------------------------------------------------------------------------
# AC: Given rows with week + sales_uplift_percentage → bar chart spec
# ---------------------------------------------------------------------------


class TestBarChartWeek:
    def test_returns_visualization_spec(self):
        result = build_visualization_spec(_rows("week", "sales_uplift_percentage"))
        assert isinstance(result, VisualizationSpec)

    def test_chart_type_is_bar(self):
        result = build_visualization_spec(_rows("week", "sales_uplift_percentage"))
        assert result.chart_type == "bar"

    def test_x_field_is_week(self):
        result = build_visualization_spec(_rows("week", "sales_uplift_percentage"))
        assert result.x_field == "week"

    def test_y_format_is_percentage(self):
        result = build_visualization_spec(_rows("week", "sales_uplift_percentage"))
        assert result.y_format == "percentage"

    def test_scale_is_x100(self):
        result = build_visualization_spec(_rows("week", "sales_uplift_percentage"))
        assert result.scale == "x100"


# ---------------------------------------------------------------------------
# AC: Given rows that match no supported pattern → should_visualize: false
# ---------------------------------------------------------------------------


class TestUnsupportedPattern:
    def test_unknown_columns_returns_no_visualization(self):
        result = build_visualization_spec(_rows("store_id", "revenue"))
        assert result == {"should_visualize": False}

    def test_partial_match_x_only_returns_no_visualization(self):
        result = build_visualization_spec(_rows("year_week", "unknown_metric"))
        assert result == {"should_visualize": False}

    def test_partial_match_y_only_returns_no_visualization(self):
        result = build_visualization_spec(_rows("unknown_dim", "actual_sales"))
        assert result == {"should_visualize": False}

    def test_wrong_x_for_uplift_returns_no_visualization(self):
        # year_month is not a supported x-axis field
        result = build_visualization_spec(
            _rows("year_month", "sales_uplift_percentage")
        )
        assert result == {"should_visualize": False}


# ---------------------------------------------------------------------------
# AC: Given empty rows → should_visualize: false
# ---------------------------------------------------------------------------


class TestEmptyRows:
    def test_empty_list_returns_no_visualization(self):
        result = build_visualization_spec([])
        assert result == {"should_visualize": False}

    def test_user_requested_chart_still_false_on_empty(self):
        result = build_visualization_spec([], user_requested_chart=True)
        assert result == {"should_visualize": False}


# ---------------------------------------------------------------------------
# AC: JUMBO_STYLE constant is defined and accessible
# ---------------------------------------------------------------------------


class TestJumboStyle:
    def test_jumbo_style_is_dict(self):
        assert isinstance(JUMBO_STYLE, dict)

    def test_jumbo_style_has_palette(self):
        assert "palette" in JUMBO_STYLE
        assert isinstance(JUMBO_STYLE["palette"], list)
        assert len(JUMBO_STYLE["palette"]) > 0

    def test_jumbo_yellow_is_primary(self):
        assert JUMBO_STYLE["palette"][0] == "#EEB717"

    def test_jumbo_style_has_required_keys(self):
        required = {"palette", "font_family", "background", "grid_color", "text_color"}
        assert required.issubset(JUMBO_STYLE.keys())


# ---------------------------------------------------------------------------
# AC: Spec includes all required fields
# ---------------------------------------------------------------------------


class TestSpecFields:
    def test_all_required_fields_present_line(self):
        result = build_visualization_spec(_rows("year_week", "actual_sales"))
        assert isinstance(result, VisualizationSpec)
        assert result.chart_type == "line"
        assert result.x_field == "year_week"
        assert result.y_field == "actual_sales"
        assert result.y_format == "currency"
        assert isinstance(result.decimals, int)
        assert result.scale == "identity"
        assert result.style_preset == "jumbo"

    def test_all_required_fields_present_bar(self):
        result = build_visualization_spec(_rows("year_week", "sales_uplift_percentage"))
        assert isinstance(result, VisualizationSpec)
        assert result.chart_type == "bar"
        assert result.x_field == "year_week"
        assert result.y_field == "sales_uplift_percentage"
        assert result.y_format == "percentage"
        assert isinstance(result.decimals, int)
        assert result.scale == "x100"
        assert result.style_preset == "jumbo"

    def test_decimals_line_chart(self):
        result = build_visualization_spec(_rows("year_week", "actual_sales"))
        assert result.decimals == 0

    def test_decimals_bar_chart(self):
        result = build_visualization_spec(_rows("year_week", "sales_uplift_percentage"))
        assert result.decimals == 1
