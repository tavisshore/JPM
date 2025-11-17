import re
from typing import Sequence

import numpy as np

from jpm.question_1.misc import format_money
from jpm.question_1.models.metrics import Metric

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
DEFAULT_HEADERS = ("Category", "Real Value", "MAE", "Proportion")


def _strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def fmt(name: str) -> str:
    return name.replace("_", " ").title()


def format_pct(p: float | float) -> str:
    if np.isnan(p):
        return "N/A"
    return f"{p * 100:.1f}%"


def colour(text: str, fg: str = "") -> str:
    COLORS = {"red": "\033[31m", "orange": "\033[38;5;208m", "green": "\033[32m"}
    reset = "\033[0m"
    return f"{COLORS.get(fg, '')}{text}{reset}" if fg else text


def colour_pct(pct: float) -> str:
    pct_str = f"{pct * 100:.1f}%"
    if pct < 0.08:
        return colour(pct_str, "green")
    elif pct < 0.20:
        return colour(pct_str, "yellow")
    else:
        return colour(pct_str, "orange")


def colour_mae(val: float) -> str:
    bn = val / 1e9
    s = f"{bn:.1f}bn"
    if bn < 1:
        return colour(s, "green")
    elif bn < 5:
        return colour(s, "yellow")
    return colour(s, "orange")


def make_row(category: str, metric: Metric) -> list[str]:
    return [
        category,
        format_money(metric.value),
        colour_mae(metric.mae),
        colour_pct(metric.pct),
    ]


def build_section_rows(
    sections: dict[str, list[str]],
    feature_stats: dict[str, dict],
) -> list[list[str]]:
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
    rows: list[list[str]] = []
    for feat in equity_feats:
        m = feature_stats.get(feat)
        if m is None:
            continue
        category = f"{fmt(feat)}"
        rows.append(make_row(category, m))
    return rows


def print_table(
    title: str,
    rows: list[list[str]],
    headers: Sequence[str] | None = None,
) -> None:
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
