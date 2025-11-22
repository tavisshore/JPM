import re
from typing import Sequence

from jpm.question_1.misc import format_money
from jpm.question_1.models.metrics import Metric

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
DEFAULT_HEADERS = ("Category", "Ground Truth", "Predicted", "Error")


def _strip_ansi(s: str) -> str:
    """Remove ANSI escape sequences to measure visible width."""
    return ANSI_RE.sub("", s)


def fmt(name: str) -> str:
    """Humanize snake_case identifiers."""
    return name.replace("_", " ").title()


def colour(text: str, fg: str = "") -> str:
    """Wrap text with ANSI colour codes when a colour is provided."""
    COLOURS = {
        "red": "\033[31m",
        "orange": "\033[38;5;208m",
        "yellow": "\033[33m",
        "green": "\033[32m",
    }
    reset = "\033[0m"
    return f"{COLOURS.get(fg, '')}{text}{reset}" if fg else text


def colour_mae(val: float) -> str:
    """Format an error value in billions with colour thresholds."""
    bn = val / 1e9
    s = f"${bn:.1f}bn"
    if bn < 1:
        return colour(s, "green")
    elif bn < 5:
        return colour(s, "yellow")
    return colour(s, "orange")


def colour_skill(skill: float) -> str:
    """Colour a skill value: green for positive, orange near zero, red otherwise."""
    pct_str = f"{skill * 100:.1f}%"
    if skill > 0.0:
        return colour(pct_str, "green")
    elif skill > -0.1:
        return colour(pct_str, "orange")
    return colour(pct_str, "red")


def make_row(category: str, metric: Metric) -> list[str]:
    """Build a display row for a metric with GT, prediction, and error."""
    return [
        category,
        format_money(metric.gt),
        format_money(metric.value),
        colour_mae(metric.mae),
    ]


def build_section_rows(
    sections: dict[str, list[str]],
    feature_stats: dict[str, dict],
) -> list[list[str]]:
    """Create rows for assets/liabilities sections from feature metrics."""
    rows: list[list[str]] = []
    for section_key, feats in sections.items():
        section_name = fmt(section_key)
        if "non_current" in section_key.lower():
            section_name = "Non-Current"
        elif "current" in section_key.lower():
            section_name = "Current"

        for feat in feats:
            m = feature_stats.get(feat)
            if m is None:
                continue
            category = f"{section_name} - {fmt(feat)}"
            rows.append(make_row(category, m))
    return rows


def build_equity_rows(
    equity_feats: list[str],
    feature_stats: dict[str, dict],
) -> list[list[str]]:
    """Create rows for equity metrics."""
    rows: list[list[str]] = []
    for feat in equity_feats:
        m = feature_stats.get(feat)
        if m is None:
            continue
        category = f"{fmt(feat)}"
        rows.append(make_row(category, m))
    return rows


def build_baseline_rows(
    baseline_mae: dict[str, float],
    skills: dict[str, float],
    model_mae: float,
    *,
    ground_truth: float | None = None,
    baseline_pred: dict[str, float] | None = None,
    model_pred: float | None = None,
) -> list[list[str]]:
    """Render baseline comparison rows, optionally including GT and predictions."""
    rows: list[list[str]] = []
    if ground_truth is None:
        rows.append(["Model (reference)", colour_mae(model_mae), colour_skill(0.0)])
        for name in sorted(baseline_mae):
            mae = baseline_mae[name]
            skill = skills.get(name, float("nan"))
            rows.append(
                [
                    fmt(name),
                    colour_mae(mae),
                    colour_skill(skill),
                ]
            )
    else:
        rows.append(
            [
                "Model (reference)",
                format_money(ground_truth),
                format_money(model_pred or 0.0),
                colour_mae(abs((model_pred or 0.0) - ground_truth)),
            ]
        )
        preds = baseline_pred or {}
        for name in sorted(baseline_mae):
            pred_val = preds.get(name, 0.0)
            rows.append(
                [
                    fmt(name),
                    format_money(ground_truth),
                    format_money(pred_val),
                    colour_mae(abs(pred_val - ground_truth)),
                ]
            )
    return rows


def print_table(
    title: str,
    rows: list[list[str]],
    headers: Sequence[str] | None = None,
) -> None:
    """Render a simple aligned ASCII table."""
    if not rows:
        return

    header_values = list(headers) if headers is not None else list(DEFAULT_HEADERS)

    col_widths: list[int] = []
    for col_idx in range(len(header_values)):
        max_len = len(header_values[col_idx])
        for row in rows:
            cell = str(row[col_idx])
            vis_len = len(_strip_ansi(cell))
            if vis_len > max_len:
                max_len = vis_len
        col_widths.append(max_len)

    def _fmt_cell(value: str, width: int) -> str:
        s = str(value)
        vis_len = len(_strip_ansi(s))
        pad = max(0, width - vis_len)
        return s + " " * pad

    def _fmt_row(values: list[str]) -> str:
        return " | ".join(
            _fmt_cell(v, w) for v, w in zip(values, col_widths, strict=True)
        )

    sep = "-+-".join("-" * w for w in col_widths)

    print(f"\n{title}")
    print(_fmt_row(header_values))
    print(sep)
    for row in rows:
        print(_fmt_row(row))
