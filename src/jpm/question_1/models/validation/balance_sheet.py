from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from jpm.question_1.config import Config
from jpm.question_1.data.datasets.statements import StatementsDataset
from jpm.question_1.data.ed import EdgarData
from jpm.question_1.misc import format_money
from jpm.question_1.models.metrics import TickerResults
from jpm.question_1.vis import colour, print_table


@dataclass
class Assets:
    """Balance sheet assets split by current and non-current."""

    assets: Dict[str, float]
    # current_assets: Dict[str, float] = None
    # non_current_assets: Dict[str, float] = None

    @property
    def total(self) -> float:
        return sum(self.assets.values())  # + sum(self.non_current_assets.values())


@dataclass
class Liabilities:
    """Balance sheet liabilities split by current and non-current."""

    liabilities: Dict[str, float]
    # non_current_liabilities: Dict[str, float]

    @property
    def total(self) -> float:
        return sum(self.liabilities.values())


@dataclass
class Equity:
    """Equity items."""

    items: Dict[str, float]

    @property
    def total(self) -> float:
        return sum(self.items.values())


class BalanceSheet:
    """Construct and validate a balance sheet from model results."""

    def __init__(
        self,
        config: Config,
        data: EdgarData,
        dataset: StatementsDataset,
        results: TickerResults,
    ) -> None:
        self.config = config
        self.results = results
        self.data = data
        self.dataset = dataset

        self._feature_values = results.feature_values()

        # Ticker-specific structure -> eventually make universal (pt2?)
        self.assets = self._build_assets()
        self.liabilities = self._build_liabilities()
        self.equity = self._build_equity()

    @property
    def total_assets(self) -> float:
        return self.assets.total

    @property
    def total_liabilities(self) -> float:
        return self.liabilities.total

    @property
    def total_equity(self) -> float:
        return self.equity.total

    @property
    def total_liabilities_and_equity(self) -> float:
        return self.total_liabilities + self.total_equity

    def check_identity(self, atol: float = 1e3) -> float:
        """Check Assets ≈ Liabilities + Equity."""
        A = self.total_assets
        L_plus_E = self.total_liabilities_and_equity
        diff = A - L_plus_E
        passed = abs(diff) <= atol

        diff_pct = (diff / A) * 100 if A != 0 else 0.0
        diff_amt = f"${diff:,.2f}"
        diff_pct_str = f"{diff_pct:.2f}%"

        if passed:
            diff_col = colour(f"{diff_amt} | {diff_pct_str}", "green")
        else:
            if diff_pct < 1:
                diff_col = colour(f"{diff_amt} | {diff_pct_str}", "orange")
            else:
                diff_col = colour(f"{diff_amt} | {diff_pct_str}", "red")

        rows = [
            [
                "Accounting Identity (A = L + E)",
                f"{format_money(A)}",
                f"{format_money(L_plus_E)}",
                diff_col,
            ]
        ]

        print_table(
            title="Balance Sheet Identity Check",
            rows=rows,
            headers=["", "Assets", "Liabilities + Equity", "Difference"],
        )
        # Single-row table keeps the check consistent with other views
        return diff_pct

    def _get_value(self, name: str) -> float:
        return float(self._feature_values.get(name, 0.0))

    def _build_assets(self) -> Assets:
        assets_struct = self.dataset.bs_structure["Assets"]

        # current_names = assets_struct.get("current_assets", [])
        # non_current_names = assets_struct.get("non_current_assets", [])

        assets = {name: self._get_value(name) for name in assets_struct}
        # non_current = {name: self._get_value(name) for name in non_current_names}
        # Preserve category split for later reporting

        return Assets(
            assets=assets,
            # non_current_assets=non_current,
        )

    def _build_liabilities(self) -> Liabilities:
        liab_struct = self.dataset.bs_structure["Liabilities"]

        # current_names = liab_struct.get("current_liabilities", [])
        # non_current_names = liab_struct.get("non_current_liabilities", [])

        liabilities = {name: self._get_value(name) for name in liab_struct}
        # non_current = {name: self._get_value(name) for name in non_current_names}

        return Liabilities(
            liabilities=liabilities,
            # non_current_liabilities=non_current,
        )

    def _build_equity(self) -> Equity:
        equity_names = self.dataset.bs_structure.get("Equity", [])
        items = {name: self._get_value(name) for name in equity_names}
        return Equity(items=items)
