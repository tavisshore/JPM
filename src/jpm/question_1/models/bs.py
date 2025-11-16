from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from src.jpm.question_1.config import Config
from src.jpm.question_1.data.utils import get_bs_structure
from src.jpm.question_1.models.metrics import TickerResults
from src.jpm.question_1.vis import colour, print_table


@dataclass
class Assets:
    current_assets: Dict[str, float]
    non_current_assets: Dict[str, float]

    @property
    def total(self) -> float:
        return sum(self.current_assets.values()) + sum(self.non_current_assets.values())


@dataclass
class Liabilities:
    current_liabilities: Dict[str, float]
    non_current_liabilities: Dict[str, float]

    @property
    def total(self) -> float:
        return sum(self.current_liabilities.values()) + sum(
            self.non_current_liabilities.values()
        )


@dataclass
class Equity:
    items: Dict[str, float]

    @property
    def total(self) -> float:
        return sum(self.items.values())


class BalanceSheet:
    def __init__(
        self,
        config: Config,
        results: TickerResults,
    ) -> None:
        self.config = config
        self.results = results

        self._feature_values: Dict[str, float] = results.feature_values()

        # Ticker-specific structure -> eventually make universal (pt2?)
        self.bs_structure = get_bs_structure(ticker=self.config.data.ticker)

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

    def check_identity(self, atol: float = 1e-3) -> None:
        """
        Check Assets ≈ Liabilities + Equity.
        """
        A = self.total_assets
        L_plus_E = self.total_liabilities_and_equity
        diff = A - L_plus_E
        passed = abs(diff) <= atol

        # Colour rules
        if passed:
            status = colour("PASS", "green")
            diff_col = colour(f"{diff / 1e9:.2f}bn", "green")
        else:
            status = colour("FAIL", "red")
            diff_col = colour(f"{diff / 1e9:.2f}bn", "red")

        rows = [
            [
                "Accounting Identity (A = L + E)",
                f"${A / 1e9:.2f}bn",
                f"${L_plus_E / 1e9:.2f}bn",
                diff_col,
                status,
            ]
        ]

        print_table(
            title="Balance Sheet Identity Check",
            rows=rows,
            headers=["", "Assets", "Liabilities + Equity", "Difference"],
        )

    def _get_value(self, name: str) -> float:
        return float(self._feature_values.get(name, 0.0))

    def _build_assets(self) -> Assets:
        assets_struct = self.bs_structure["assets"]

        current_names = assets_struct.get("current_assets", [])
        non_current_names = assets_struct.get("non_current_assets", [])

        current = {name: self._get_value(name) for name in current_names}
        non_current = {name: self._get_value(name) for name in non_current_names}

        return Assets(
            current_assets=current,
            non_current_assets=non_current,
        )

    def _build_liabilities(self) -> Liabilities:
        liab_struct = self.bs_structure["liabilities"]

        current_names = liab_struct.get("current_liabilities", [])
        non_current_names = liab_struct.get("non_current_liabilities", [])

        current = {name: self._get_value(name) for name in current_names}
        non_current = {name: self._get_value(name) for name in non_current_names}

        return Liabilities(
            current_liabilities=current,
            non_current_liabilities=non_current,
        )

    def _build_equity(self) -> Equity:
        equity_names = self.bs_structure.get("equity", [])
        items = {name: self._get_value(name) for name in equity_names}
        return Equity(items=items)
