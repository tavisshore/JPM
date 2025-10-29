from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List

import pandas as pd

from src.jpm.question_1.components.input import InputData


def _make_year_index(start, periods: int, like: pd.Index) -> pd.Index:
    """
    Create a year sequence starting at `start` with `periods` length,
    matching (Int/Datetime/Period)
    """
    if isinstance(like, pd.DatetimeIndex) or isinstance(start, pd.Timestamp):
        start_ts = pd.Timestamp(start)
        return pd.date_range(start=start_ts, periods=periods, freq="YS")
    if isinstance(like, pd.PeriodIndex) or isinstance(start, pd.Period):
        start_p = (
            pd.Period(start, freq="Y")
            if not isinstance(start, pd.Period)
            else start.asfreq("Y")
        )
        return pd.period_range(start=start_p, periods=periods, freq="Y")
    start_int = int(start)
    return pd.Index(range(start_int, start_int + periods), dtype=int)


@dataclass
class STLoan:
    """
    Short-term loans class
    Within vp model, these are assumed to be 1-year loans
    drawn at the start of year t, repaid in full at the end of year t (TODO?)
    """

    input: "InputData"
    amount: float
    start_year: object

    _years: pd.Index = field(init=False, repr=False)
    _df: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self):
        self._years = self.input.years
        if self.start_year not in set(self._years):
            raise ValueError(f"start_year {self.start_year!r} not in schedule index.")
        self._df = self._build_df()

    @property
    def years(self) -> pd.Index:
        return self._years

    def _build_df(self) -> pd.DataFrame:
        y = self._years
        kd = self.input.kd.reindex(y, fill_value=0.0)

        # Single draw only in start_year; zero otherwise
        draws = pd.Series(0.0, index=y, dtype=float)
        draws.loc[self.start_year] = float(self.amount)

        # Beginning-of-year t balance is last year's draw; repaid in full in year t
        beg = draws.shift(1).fillna(0.0)
        interest = beg * kd
        principal = beg.copy()
        total = interest + principal
        end = draws.copy()

        return pd.DataFrame(
            {
                "Beginning balance": beg,
                "Interest rate (kd)": kd,
                "Interest payment ST loan": interest,
                "Principal payments ST loan": principal,
                "Total payment ST loan": total,
                "Ending balance": end,
                "Draws ST loan": draws,
            },
            index=y,
        )

    def compute(self, year) -> pd.Series:
        df = self._df
        if year not in df.index:
            raise KeyError(f"{year!r} not in schedule index")
        return df.loc[year]


@dataclass
class LTLoan:
    """
    Long-term loans class
    Within vp model, these are assumed to be taken at start
    Length of loan set in InputData.lt_loan_term_years
    """

    input: "InputData"
    start_year: object
    initial_draw: float

    _years: pd.Index = field(init=False, repr=False)
    _df: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self):
        self._years = _make_year_index(
            self.start_year, self.input.lt_loan_term_years, like=self.input.years
        )
        self._df = self._build_df()

    @property
    def years(self) -> pd.Index:
        return self._years

    def _build_df(self) -> pd.DataFrame:
        y = self.years

        beg = pd.Series(0.0, index=y)
        beg.iloc[0] = float(self.initial_draw)

        term = int(self.input.lt_loan_term_years)
        annual_principal = (self.initial_draw / term) if term > 0 else 0.0

        principal = pd.Series(0.0, index=y)
        interest = pd.Series(0.0, index=y)
        end = pd.Series(0.0, index=y)

        kd = self.input.kd.reindex(y, fill_value=0.0)

        for i, _ in enumerate(y):
            if i == 0:
                end.iloc[i] = beg.iloc[i]  # no payment in year 0
                continue

            beg.iloc[i] = end.iloc[i - 1]
            r = float(kd.iloc[i])
            interest.iloc[i] = beg.iloc[i] * r
            principal.iloc[i] = min(annual_principal, beg.iloc[i])
            end.iloc[i] = beg.iloc[i] - principal.iloc[i]

        total = interest + principal

        return pd.DataFrame(
            {
                "Beginning balance": beg,
                "Interest payment LT loan": interest,
                "Principal payments LT loan": principal,
                "Total payment LT loan": total,
                "Ending balance": end,
                "Interest rate": kd,
            },
            index=y,
        )

    def compute(self, year) -> pd.Series:
        df = self._df
        if year not in df.index:
            raise KeyError(f"{year!r} not in schedule index")
        return df.loc[year]


@dataclass
class LoanBook:
    """
    Class for storing multiple loans (ST and LT)
    and retrieving their due payments per year
    """

    loans: List[STLoan | LTLoan] = field(default_factory=list)

    def add(self, loan: STLoan | LTLoan) -> None:
        self.loans.append(loan)

    def extend(self, items: Iterable[STLoan | LTLoan]) -> None:
        self.loans.extend(items)

    def __len__(self) -> int:
        return len(self.loans)

    def __iter__(self):
        return iter(self.loans)

    def of_type(self, cls) -> list[STLoan | LTLoan]:
        return [loan for loan in self.loans if isinstance(loan, cls)]

    def dues(self, year) -> list[pd.Series[Any]]:
        return [loan.compute(year) for loan in self.loans]
