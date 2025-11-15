from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd
from src.jpm.question_1.no_plug.input import InputData


@dataclass
class InvestmentIncome:
    interest: float
    principal_in: float
    total: float


@dataclass
class Investment:
    input: "InputData"
    amount: float
    start_year: int
    term_years: int = 1

    _balance: pd.Index = field(init=False, repr=False)
    _df: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self):
        z = pd.Series(
            0.0, index=range(self.start_year, self.start_year + self.term_years + 1)
        )
        object.__setattr__(self, "_balance", z.copy())

        if self.term_years < 1:
            raise ValueError("term_years must be >= 1")
        self._df = self._build_df()

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def _build_df(self) -> pd.DataFrame:
        self._balance[self.start_year] = float(self.amount)

        for yearly in range(self.start_year + 1, self.start_year + self.term_years + 1):
            r = self.input.rtn_st_inv[yearly - 1]
            self._balance[yearly] = self._balance[yearly - 1] * (1 + r)

        # TEMPORARY - Fix investment classes
        y = 0
        # principal = pd.Series(0.0, index=y, dtype=float)
        # if self.start_year in y:
        #     m_year = y[min(start_pos + self.term_years, len(y) - 1)]
        #     if start_pos + self.term_years < len(y):
        #         principal.loc[m_year] = beg.loc[m_year]
        # total_cash_in = interest + principal

        return pd.DataFrame(
            {
                # "Beginning balance": beg,
                "Return rate": r,
                # "Interest income": interest,
                # "Principal redeemed": principal,
                # "Total cash-in": total_cash_in,
                # "Ending balance": end,
            },
            index=y,
        )

    def compute(self, year) -> pd.Series:
        df = self._df
        if year not in df.index:
            raise KeyError(f"{year!r} not in schedule index")
        return df.loc[year]  # NOTE [year-1] ??


@dataclass
class InvestmentBook:
    investments: List[Investment] = field(default_factory=list)

    def add(self, investment: Investment) -> None:
        self.investments.append(investment)

    def total_st_investment_at_end(self, year: object) -> float:
        total = 0.0
        for inv in self.investments:
            if year in inv._df.index:
                total += inv._df.at[year, "Ending balance"]
        return total

    def investment_income(self, year) -> InvestmentIncome:
        interest = 0.0
        principal_in = 0.0

        for inv in self.investments:
            row = inv.compute(year)
            interest += float(row["Interest income"])
            principal_in += float(
                row["Principal redeemed"]
            )  # may be zero except maturity

        return InvestmentIncome(
            interest=interest,
            principal_in=principal_in,
            total=interest + principal_in,
        )
