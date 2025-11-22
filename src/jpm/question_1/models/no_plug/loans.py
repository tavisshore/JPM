from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from jpm.question_1.models.no_plug.input import InputData


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
class DebtPay:
    st_interest: float
    st_principal: float
    lt_interest: float
    lt_principal: float
    total: float


@dataclass
class DebtBalances:
    st_debt: float
    lt_debt: float
    total: float


@dataclass
class Loan:
    """
    Long-term loans class
    Within vp model, these are assumed to be taken at start
    Length of loan set in InputData.lt_loan_term_years
    """

    input: "InputData"
    start_year: object
    initial_draw: float
    category: str

    _years: pd.Index = field(init=False, repr=False)
    _df: pd.DataFrame = field(init=False, repr=False, default=None)

    def __post_init__(self):
        length = (
            self.input.lt_loan_term
            if self.category == "LT"
            else self.input.st_loan_term + 1
        )
        self._years = _make_year_index(self.start_year, length, like=self.input.years)
        self._df = self._build_df()

    @property
    def years(self) -> pd.Index:
        return self._years

    def _build_df(self) -> pd.DataFrame:
        y = self.years

        beg = pd.Series(0.0, index=y)
        beg.iloc[0] = float(self.initial_draw)

        if self.category == "LT":
            term = int(self.input.lt_loan_term)
        else:
            term = int(self.input.st_loan_term)
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
                f"Interest payment {self.category} loan": interest,
                f"Principal payments {self.category} loan": principal,
                f"Total payment {self.category} loan": total,
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

    def loan_ongoing(self, year) -> bool:
        df = self._df
        return year in df.index


@dataclass
class LoanBook:
    loans: list[Loan] = field(default_factory=list)

    def add(self, loan: Loan) -> None:
        if isinstance(loan, Loan):
            self.loans.append(loan)
        else:
            raise TypeError(f"Unsupported loan type: {type(loan)}")

    def extend(self, loans: Iterable[Loan]) -> None:
        for loan in loans:
            self.add(loan)

    def all(self) -> list[Loan]:
        return self.loans

    def __len__(self) -> int:
        return len(self.loans)

    def __iter__(self):
        return iter(self.all())

    def debt_payments(self, year) -> DebtPay:
        """
        Aggregates debt payments for a given year using the loan schedules.
        """
        st_interest, st_principal = 0.0, 0.0
        lt_interest, lt_principal = 0.0, 0.0

        for loan in self.loans:
            if loan.loan_ongoing(year):  # Remove spent loans?
                row = loan.compute(year)
                if loan.category == "LT":
                    lt_interest += float(row["Interest payment LT loan"])
                    lt_principal += float(row["Principal payments LT loan"])
                elif loan.category == "ST":
                    st_interest += float(row["Interest payment ST loan"])
                    st_principal += float(row["Principal payments ST loan"])

        return DebtPay(
            st_interest=st_interest,
            st_principal=st_principal,
            lt_interest=lt_interest,
            lt_principal=lt_principal,
            total=st_interest + st_principal + lt_interest + lt_principal,
        )

    def remaining_debt(self, year):
        """
        Return ST and LT debt totals
        """
        st_total, lt_total = 0.0, 0.0

        for loan in self.loans:
            if loan.loan_ongoing(year):  # Remove spent loans?
                row = loan.compute(year)
                # print(row)
                # Beginning balance             10.0
                # Interest payment ST loan       0.0
                # Principal payments ST loan     0.0
                # Total payment ST loan          0.0
                # Ending balance                10.0
                # Interest rate                  0.0
                if loan.category == "LT":
                    lt_total += float(row["Ending balance"])  # Or end?
                elif loan.category == "ST":
                    st_total += float(row["Ending balance"])

        return DebtBalances(
            st_debt=st_total, lt_debt=lt_total, total=st_total + lt_total
        )
