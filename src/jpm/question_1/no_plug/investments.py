from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd

from src.jpm.question_1.no_plug.input import InputData


@dataclass
class InvestmentIncome:
    st_interest: float
    st_principal_in: float
    total: float


@dataclass
class Investment:
    """
    Fixed-term short-term investment.
    - Earn interest each year on start balance at InputData.rtn_st_inv
    - Redeem principal at maturity - vp assumes 1-year at the moment
    """

    input: "InputData"
    amount: float
    start_year: object
    term_years: int = 1

    _years: pd.Index = field(init=False, repr=False)
    _df: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self):
        self._years = self.input.years
        if self.start_year not in set(self._years):
            raise ValueError(f"start_year {self.start_year!r} not in schedule index.")
        if self.term_years < 1:
            raise ValueError("term_years must be >= 1")
        self._df = self._build_df()

    @property
    def years(self) -> pd.Index:
        return self._years

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def _build_df(self) -> pd.DataFrame:
        y = self._years
        r = self.input.rtn_st_inv.reindex(y, fill_value=0.0)

        deposits = pd.Series(0.0, index=y, dtype=float)
        deposits.loc[self.start_year] = float(self.amount)

        end = pd.Series(0.0, index=y, dtype=float)
        if self.start_year in y:
            start_pos = y.get_loc(self.start_year)
            end_pos = start_pos + self.term_years - 1
            end.iloc[start_pos : min(end_pos + 1, len(y))] = float(self.amount)

        # Starting balance is previous year's ending balance
        beg = end.shift(1).fillna(0.0)
        interest = beg * r

        principal = pd.Series(0.0, index=y, dtype=float)
        if self.start_year in y:
            m_year = y[min(start_pos + self.term_years, len(y) - 1)]
            if start_pos + self.term_years < len(y):
                principal.loc[m_year] = beg.loc[m_year]
        total_cash_in = interest + principal

        return pd.DataFrame(
            {
                "Beginning balance": beg,
                "Return rate": r,
                "Interest income": interest,
                "Principal redeemed": principal,
                "Total cash-in": total_cash_in,
                "Ending balance": end,
                "Deposits": deposits,
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
    st_investments: List[Investment] = field(default_factory=list)

    def add(self, investment: Investment) -> None:
        self.st_investments.append(investment)

    def total_st_investment_at_end(self, year: object) -> float:
        total = 0.0
        for inv in self.st_investments:
            if year in inv._df.index:
                total += inv._df.at[year, "Ending balance"]
        return total

    def investment_income(self, year) -> InvestmentIncome:
        st_interest = 0.0
        st_principal_in = 0.0

        for inv in self.st_investments:
            row = inv.compute(year)
            st_interest += float(row["Interest income"])
            st_principal_in += float(
                row["Principal redeemed"]
            )  # may be zero except maturity

        return InvestmentIncome(
            st_interest=st_interest,
            st_principal_in=st_principal_in,
            total=st_interest + st_principal_in,
        )
