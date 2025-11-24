from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd

from jpm.question_1.models.no_plug.input import InputData
from jpm.question_1.models.no_plug.loans import _make_year_index


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

    _years: pd.Index = field(init=False, repr=False)
    _df: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self):
        if self.term_years < 1:
            raise ValueError("term_years must be >= 1")
        if self.amount < 0:
            raise ValueError("investment amount cannot be negative")
        self._years = _make_year_index(
            self.start_year, self.term_years + 1, like=self.input.years
        )
        self._df = self._build_df()

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def _build_df(self) -> pd.DataFrame:
        years = self._years
        beg = pd.Series(0.0, index=years, dtype=float)
        beg.iloc[0] = float(self.amount)

        rates = self.input.rtn_st_inv.reindex(years, fill_value=0.0)

        interest = pd.Series(0.0, index=years, dtype=float)
        principal = pd.Series(0.0, index=years, dtype=float)
        end = pd.Series(0.0, index=years, dtype=float)

        for i, _year in enumerate(years):
            if i == 0:
                end.iloc[i] = beg.iloc[i]
                continue

            beg.iloc[i] = end.iloc[i - 1]
            rate = float(rates.iloc[i])
            interest.iloc[i] = beg.iloc[i] * rate

            if i == len(years) - 1:
                principal.iloc[i] = beg.iloc[i]
                end.iloc[i] = 0.0
            else:
                end.iloc[i] = beg.iloc[i]

        total_cash_in = interest + principal

        return pd.DataFrame(
            {
                "Beginning balance": beg,
                "Return rate": rates,
                "Interest income": interest,
                "Principal redeemed": principal,
                "Total cash-in": total_cash_in,
                "Ending balance": end,
            },
            index=years,
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
        if not isinstance(investment, Investment):
            raise TypeError(f"Expected Investment instance, got {type(investment)}")
        self.investments.append(investment)

    def total_st_investment_at_end(self, year: object) -> float:
        total = 0.0
        for inv in self.investments:
            if year in inv.df.index:
                total += inv.df.at[year, "Ending balance"]
        return total

    def investment_income(self, year) -> InvestmentIncome:
        interest = 0.0
        principal_in = 0.0

        for inv in self.investments:
            if year in inv.df.index:
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
