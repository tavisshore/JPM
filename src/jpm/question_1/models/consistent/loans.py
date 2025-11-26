from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd
from forecasting import Forecasting
from input import InputData


def _make_year_index(start, periods: int, like: pd.Index) -> pd.Index:
    """
    Create a year sequence starting at `start` with `periods` length,
    matching (Int/Datetime/Period)
    """
    if periods <= 0:
        raise ValueError("periods must be positive")
    if start is None:
        raise ValueError("start year is required to build a schedule")
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
class ShortTermLoan:
    beginning_balance: float
    interest_payments: float
    principal_payments: float
    ending_balance: float
    cost_of_debt: float


@dataclass
class LongTermLoan:
    beginning_balance: float
    interest_payments: float
    new_debt: float
    principal_payments: float
    ending_balance: float


@dataclass
class LoanTable:
    short_term: ShortTermLoan
    long_term: LongTermLoan


@dataclass
class Loan:
    input_data: InputData
    forecast: Forecasting
    start_year: object
    initial_draw: float
    category: str

    _years: pd.Index = field(init=False, repr=False)
    _df: pd.DataFrame = field(init=False, repr=False, default=None)

    def __post_init__(self):
        if self.category not in {"LT", "ST"}:
            raise ValueError("category must be either 'LT' or 'ST'")
        if self.initial_draw < 0:
            raise ValueError("initial_draw cannot be negative")

        length = (
            self.input_data.lt_years_loan_3
            if self.category == "LT"
            else self.input_data.st_years_loan_2 + 1
        )
        self._years = _make_year_index(
            self.start_year, length, like=self.input_data.years
        )
        self._df = self._build_df()

    @property
    def years(self) -> pd.Index:
        return self._years

    def _build_df(self) -> pd.DataFrame:
        y = self.years

        beg = pd.Series(0.0, index=y)
        beg.iloc[0] = float(self.initial_draw)

        if self.category == "LT":
            term = int(self.input_data.lt_years_loan_3)
        else:
            term = int(self.input_data.st_years_loan_2)
        annual_principal = (self.initial_draw / term) if term > 0 else 0.0

        principal = pd.Series(0.0, index=y)
        end = pd.Series(0.0, index=y)

        for i, _ in enumerate(y):
            if i == 0:
                end.iloc[i] = beg.iloc[i]  # no payment in year 0
                continue

            beg.iloc[i] = end.iloc[i - 1]
            principal.iloc[i] = min(annual_principal, beg.iloc[i])
            end.iloc[i] = beg.iloc[i] - principal.iloc[i]

        return pd.DataFrame(
            {
                "Beginning balance": beg,
                f"Principal payments {self.category} loan": principal,
                "Ending balance": end,
            },
            index=y,
        )

    def compute(self, year) -> pd.Series:
        df = self._df
        if year not in df.index:
            raise KeyError(f"{year!r} not in schedule index")
        return df.loc[year]

    def interest_for_year(self, year) -> float:
        """
        Compute interest on demand using the current year's rate.
        Interest is assumed to start accruing after the initial draw year.
        """
        if not self.loan_ongoing(year):
            return 0.0
        row = self.compute(year)
        kd = self.forecast.cost_of_debt.reindex(self.years, fill_value=0.0)
        rate = float(kd.loc[year])
        # mimic prior behaviour: no interest accrued in the draw year
        if year == self.years[0]:
            return 0.0
        return float(row["Beginning balance"]) * rate

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
            if loan.loan_ongoing(year):
                row = loan.compute(year)
                principal_key = f"Principal payments {loan.category} loan"
                principal_payment = float(row[principal_key])
                interest_payment = loan.interest_for_year(year)

                if loan.category == "LT":
                    lt_interest += interest_payment
                    lt_principal += principal_payment
                elif loan.category == "ST":
                    st_interest += interest_payment
                    st_principal += principal_payment

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
            if loan.loan_ongoing(year):
                row = loan.compute(year)
                if loan.category == "LT":
                    lt_total += float(row["Ending balance"])
                elif loan.category == "ST":
                    st_total += float(row["Ending balance"])

        return DebtBalances(
            st_debt=st_total, lt_debt=lt_total, total=st_total + lt_total
        )

    def new_lt_debt(self, year) -> float:
        """
        Sum the initial draw of LT loans that start in the provided year.
        """
        total = 0.0
        for loan in self.loans:
            if loan.category == "LT" and loan.years[0] == year:
                total += float(loan.initial_draw)
        return total

    def schedule_summary(self, year) -> LoanTable:
        st_beg = st_int = st_pri = st_end = 0.0
        lt_beg = lt_int = lt_pri = lt_end = 0.0

        # Try to pick a cost of debt rate for the year (assumes shared forecast)
        kd_rate = 0.0
        for loan in self.loans:
            if year in loan.years:
                kd_rate = float(
                    loan.forecast.cost_of_debt.reindex(loan.years, fill_value=0.0).loc[
                        year
                    ]
                )
                break

        for loan in self.loans:
            if not loan.loan_ongoing(year):
                continue

            row = loan.compute(year)
            principal_key = f"Principal payments {loan.category} loan"
            # Exclude loans originated in the same year from beginning balance
            beg = 0.0 if loan.years[0] == year else float(row["Beginning balance"])
            pri = float(row[principal_key])
            end = float(row["Ending balance"])
            interest_payment = loan.interest_for_year(year)

            if loan.category == "LT":
                lt_beg += beg
                lt_pri += pri
                lt_end += end
                lt_int += interest_payment
            elif loan.category == "ST":
                st_beg += beg
                st_pri += pri
                st_end += end
                st_int += interest_payment

        short_term = ShortTermLoan(
            beginning_balance=st_beg,
            interest_payments=st_int,
            principal_payments=st_pri,
            ending_balance=st_end,
            cost_of_debt=kd_rate,
        )
        long_term = LongTermLoan(
            beginning_balance=lt_beg,
            interest_payments=lt_int,
            new_debt=self.new_lt_debt(year),
            principal_payments=lt_pri,
            ending_balance=lt_end,
        )
        loans_out = LoanTable(short_term=short_term, long_term=long_term)

        return loans_out


class LoanSchedules:
    """
    Wrapper that keeps the LoanBook interface while making it easy to
    spin up new loan instances when external financing is needed.
    """

    def __init__(self):
        self.book = LoanBook()

    def new_loan(
        self,
        *,
        category: str,
        start_year,
        amount: float,
        input_data: InputData,
        forecast: Forecasting,
    ) -> Loan:
        loan = Loan(
            input_data=input_data,
            forecast=forecast,
            start_year=start_year,
            initial_draw=amount,
            category=category,
        )
        self.book.add(loan)
        return loan

    def extend(self, loans: Iterable[Loan]) -> None:
        self.book.extend(loans)

    def debt_payments(self, year) -> DebtPay:
        return self.book.debt_payments(year)

    def remaining_debt(self, year):
        return self.book.remaining_debt(year)

    def new_lt_debt(self, year) -> float:
        return self.book.new_lt_debt(year)

    def schedule_summary(self, year) -> dict:
        return self.book.schedule_summary(year)

    def all(self) -> list[Loan]:
        return self.book.all()

    def __len__(self) -> int:
        return len(self.book)

    def __iter__(self):
        return iter(self.book)
