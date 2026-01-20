from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from jpm.config.question_1 import Config
from jpm.question_1.data.datasets.statements import StatementsDataset
from jpm.question_1.misc import format_money
from jpm.question_1.models.metrics import Metric, TickerResults
from jpm.question_1.vis import build_baseline_rows, colour, colour_mae, print_table


@dataclass
class IncomeStatementSection:
    """Container for grouped income statement metrics."""

    items: Dict[str, Metric]

    @property
    def total_pred(self) -> float:
        return sum(m.value for m in self.items.values())

    @property
    def total_gt(self) -> float:
        return sum(m.gt for m in self.items.values())


def _default_metric() -> "Metric":
    """Return a zeroed metric for missing concepts."""
    return Metric(value=0.0, mae=0.0, gt=0.0)


class IncomeStatement:
    """View helper that aggregates and prints income statement metrics."""

    def __init__(
        self, config: Config, dataset: StatementsDataset, results: TickerResults
    ) -> None:
        self.config = config
        self.results = results

        self._feature_metrics = results.features
        self.is_structure = dataset.is_structure

        self.revenues = self._build_section("Revenues")
        self.expenses = self._build_section("Expenses")
        # Sections mirror the presentation structure for readability

    def _build_section(self, name: str) -> IncomeStatementSection:
        """Assemble a section from available metrics, defaulting missing ones."""
        concepts = self.is_structure.get(name, [])
        items = {
            concept: self._feature_metrics.get(concept, _default_metric())
            for concept in concepts
        }
        return IncomeStatementSection(items=items)

    @property
    def net_income_pred(self) -> float:
        return self.revenues.total_pred - self.expenses.total_pred

    @property
    def net_income_gt(self) -> float:
        return self.revenues.total_gt - self.expenses.total_gt

    def view(self) -> None:
        rows = []
        rows.append(
            [
                "Total Revenue",
                format_money(self.revenues.total_gt),
                format_money(self.revenues.total_pred),
                colour_mae(abs(self.revenues.total_pred - self.revenues.total_gt)),
                "",
            ]
        )
        rows.append(
            [
                "Total Expenses",
                format_money(self.expenses.total_gt),
                format_money(self.expenses.total_pred),
                colour_mae(abs(self.expenses.total_pred - self.expenses.total_gt)),
                "",
            ]
        )

        ni_pred = self.net_income_pred
        ni_gt = self.net_income_gt
        net_colour = "green" if ni_pred >= 0 else "red"
        rows.append(
            [
                "Net Income (Loss)",
                format_money(ni_gt),
                colour(format_money(ni_pred), net_colour),
                colour_mae(abs(ni_pred - ni_gt)),
                "",
            ]
        )

        print_table(
            title="Income Statement",
            rows=rows,
            headers=["Category", "Ground Truth", "Predicted", "Error", "Unc."],
        )

        if self.results.net_income_baseline_mae:
            baseline_rows = build_baseline_rows(
                self.results.net_income_baseline_mae,
                self.results.net_income_skill,
                self.results.net_income_model_mae,
                ground_truth=self.net_income_gt,
                baseline_pred=self.results.net_income_baseline_pred,
                model_pred=self.net_income_pred,
            )
            print_table(
                title="Baseline Comparison (Net Income)",
                rows=baseline_rows,
                headers=("Method", "Ground Truth", "Estimate", "Error"),
            )

    def get_results(self) -> TickerResults:
        return self.results

    def view_predict(self) -> None:
        """Same as view but no error or uncertainty columns."""
        rows = []
        rows.append(
            [
                "Total Revenue",
                format_money(self.revenues.total_gt),
                format_money(self.revenues.total_pred),
            ]
        )
        rows.append(
            [
                "Total Expenses",
                format_money(self.expenses.total_gt),
                format_money(self.expenses.total_pred),
            ]
        )

        ni_pred = self.net_income_pred
        ni_gt = self.net_income_gt
        net_colour = "green" if ni_pred >= 0 else "red"
        rows.append(
            [
                "Net Income (Loss)",
                format_money(ni_gt),
                colour(format_money(ni_pred), net_colour),
            ]
        )

        print_table(
            title="Income Statement Predictions",
            rows=rows,
            headers=["Category", "Ground Truth", "Predicted"],
        )
