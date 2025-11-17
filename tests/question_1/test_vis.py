import pytest

from jpm.question_1 import vis
from jpm.question_1.models.metrics import Metric

unit = pytest.mark.unit
integration = pytest.mark.integration


@unit
def test_fmt_and_format_pct_handle_basic_cases():
    """fmt/format_pct should humanize identifiers and percentages."""
    assert vis.fmt("total_assets") == "Total Assets"
    assert vis.format_pct(0.123) == "12.3%"
    assert vis.format_pct(float("nan")) == "N/A"


@unit
def test_colour_helpers_apply_thresholds():
    """colour helpers should vary styling based on thresholds."""
    green = vis.colour_pct(0.05)
    orange = vis.colour_pct(0.3)
    assert "32m" in green or "green" in green.lower()
    assert "yellow" not in green.lower()
    assert orange != green

    mae_green = vis.colour_mae(5e8)
    mae_orange = vis.colour_mae(6e9)
    assert mae_green != mae_orange


@unit
def test_make_row_uses_metric_fields():
    """make_row should format values, mae, and pct from Metric."""
    metric = Metric(value=1000.0, mae=5e8, pct=0.05)
    row = vis.make_row("Assets", metric)
    assert row[0] == "Assets"
    assert row[1].startswith("$")
    assert "%" in row[3]


@unit
def test_build_section_rows_respects_structure_and_skips_missing():
    """build_section_rows should respect structure labels and skip missing stats."""
    sections = {
        "current_assets": ["cash", "inventory"],
        "non_current_assets": ["ppe"],
    }
    stats = {
        "cash": Metric(10, 1, 0.01),
        "inventory": Metric(20, 2, 0.02),
    }
    rows = vis.build_section_rows(sections, stats)
    assert len(rows) == 2
    assert all("Current" in row[0] for row in rows)


@unit
def test_build_equity_rows_without_stats_returns_empty():
    """build_equity_rows should only render categories with metrics."""
    stats = {"common_stock": Metric(10, 1, 0.01)}
    rows = vis.build_equity_rows(["missing", "common_stock"], stats)
    assert len(rows) == 1
    assert rows[0][0] == "Common Stock"


@integration
def test_print_table_formats_output(capsys):
    """print_table should render titles and rows in aligned columns."""
    rows = [
        ["Assets", "$1k", "0.1bn", "10.0%"],
        ["Liabilities", "$2k", "0.2bn", "20.0%"],
    ]
    vis.print_table("Overall", rows)
    captured = capsys.readouterr().out
    assert "Overall" in captured
    assert "Assets" in captured
    assert "Liabilities" in captured
