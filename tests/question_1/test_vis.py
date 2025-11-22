import pytest

from jpm.question_1 import vis
from jpm.question_1.models.metrics import Metric

unit = pytest.mark.unit
integration = pytest.mark.integration


@unit
def test_fmt_humanizes_identifiers():
    """fmt should humanize identifiers."""
    assert vis.fmt("total_assets") == "Total Assets"


@unit
def test_colour_helpers_apply_thresholds():
    """colour helpers should vary styling based on thresholds."""
    mae_green = vis.colour_mae(5e8)
    mae_orange = vis.colour_mae(6e9)
    assert mae_green != mae_orange


@unit
def test_make_row_uses_metric_fields():
    """make_row should format values and errors from Metric."""
    metric = Metric(value=1000.0, mae=5e8, gt=900.0)
    row = vis.make_row("Assets", metric)
    assert row[0] == "Assets"
    assert vis._strip_ansi(row[1]).startswith("$")
    assert vis._strip_ansi(row[2]).startswith("$")
    assert vis._strip_ansi(row[3]).startswith("$")


@unit
def test_build_baseline_rows_with_ground_truth_uses_error_column():
    baseline_rows = vis.build_baseline_rows(
        baseline_mae={"baseline": 2.0},
        skills={"baseline": -0.5},
        model_mae=1.0,
        ground_truth=100.0,
        baseline_pred={"baseline": 90.0},
        model_pred=110.0,
    )

    model_err = vis._strip_ansi(baseline_rows[0][3])
    baseline_err = vis._strip_ansi(baseline_rows[1][3])
    assert model_err.startswith("$")
    assert baseline_err.startswith("$")


@unit
def test_build_section_rows_respects_structure_and_skips_missing():
    """build_section_rows should respect structure labels and skip missing stats."""
    sections = {
        "current_assets": ["cash", "inventory"],
        "non_current_assets": ["ppe"],
    }
    stats = {
        "cash": Metric(10, 1),
        "inventory": Metric(20, 2),
    }
    rows = vis.build_section_rows(sections, stats)
    assert len(rows) == 2
    assert all("Current" in row[0] for row in rows)


@unit
def test_build_equity_rows_without_stats_returns_empty():
    """build_equity_rows should only render categories with metrics."""
    stats = {"common_stock": Metric(10, 1)}
    rows = vis.build_equity_rows(["missing", "common_stock"], stats)
    assert len(rows) == 1
    assert rows[0][0] == "Common Stock"


@integration
def test_print_table_formats_output(capsys):
    """print_table should render titles and rows in aligned columns."""
    rows = [
        ["Assets", "$1k", "$1.1k", "$0.1bn"],
        ["Liabilities", "$2k", "$2.1k", "$0.2bn"],
    ]
    vis.print_table("Overall", rows)
    captured = capsys.readouterr().out
    assert "Overall" in captured
    assert "Assets" in captured
    assert "Liabilities" in captured
