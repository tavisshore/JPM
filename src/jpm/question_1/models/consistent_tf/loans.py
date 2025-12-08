from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Union

import pandas as pd

from .forecasting import Forecasting
from .input import InputData


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
    """Aggregated debt payments for a given year."""

    st_interest: float
    st_principal: float
    lt_interest: float
    lt_principal: float
    total: float


@dataclass
class DebtBalances:
    """Outstanding short- and long-term debt balances."""

    st_debt: float
    lt_debt: float
    total: float


@dataclass
class ShortTermLoan:
    """Short-term loan schedule summary."""

    beginning_balance: Union[float, pd.Series]
    interest_payments: Union[float, pd.Series]
    principal_payments: Union[float, pd.Series]
    ending_balance: Union[float, pd.Series]
    cost_of_debt: Union[float, pd.Series]


@dataclass
class LongTermLoan:
    """Long-term loan schedule summary."""

    beginning_balance: Union[float, pd.Series]
    interest_payments: Union[float, pd.Series]
    new_debt: Union[float, pd.Series]
    principal_payments: Union[float, pd.Series]
    ending_balance: Union[float, pd.Series]


@dataclass
class LoanTable:
    """Container bundling short- and long-term loan schedules."""

    short_term: ShortTermLoan
    long_term: LongTermLoan


@dataclass
class Loan:
    """Individual loan schedule with derived interest calculations."""

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
                end.iloc[i] = beg.iloc[i]
                continue

            beg.iloc[i] = end.iloc[i - 1]
            # print(annual_principal)
            # print()
            principal.iloc[i] = min(float(annual_principal), beg.iloc[i])
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

        if year == self.years[0]:
            return 0.0
        return float(row["Beginning balance"]) * rate

    def loan_ongoing(self, year) -> bool:
        df = self._df
        return year in df.index


@dataclass
class LoanBook:
    """Collection of loans with helper aggregation utilities."""

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

    def debt_payments(self, year: int) -> DebtPay:
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

    def remaining_debt(self, year: int) -> DebtBalances:
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

    def new_lt_debt(self, year: int) -> float:
        """
        Sum the initial draw of LT loans that start in the provided year.
        """
        total = 0.0
        for loan in self.loans:
            if loan.category == "LT" and loan.years[0] == year:
                total += float(loan.initial_draw)
        return total

    def schedule_summary(self, year: int) -> LoanTable:
        st_beg = st_int = st_pri = st_end = 0.0
        lt_beg = lt_int = lt_pri = lt_end = 0.0

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
        # print(f"year: {year}")
        # print(f"LT: {long_term}")
        # print(f"new: {self.new_lt_debt(year)}")
        # print(f"new+1: {self.new_lt_debt(year + 1)}")

        # breakpoint()
        loans_out = LoanTable(short_term=short_term, long_term=long_term)
        return loans_out

    def schedule_summary_series(self, years: Iterable[int]) -> LoanTable:
        years_idx = pd.Index(years)

        st_beg = pd.Series(0.0, index=years_idx)
        st_int = pd.Series(0.0, index=years_idx)
        st_pri = pd.Series(0.0, index=years_idx)
        st_end = pd.Series(0.0, index=years_idx)

        lt_beg = pd.Series(0.0, index=years_idx)
        lt_int = pd.Series(0.0, index=years_idx)
        lt_pri = pd.Series(0.0, index=years_idx)
        lt_end = pd.Series(0.0, index=years_idx)
        kd_rate = pd.Series(0.0, index=years_idx)
        new_lt_debt = pd.Series(0.0, index=years_idx)

        for yr in years_idx:
            summary = self.schedule_summary(yr)
            short = summary.short_term
            long = summary.long_term

            st_beg.loc[yr] = short.beginning_balance
            st_int.loc[yr] = short.interest_payments
            st_pri.loc[yr] = short.principal_payments
            st_end.loc[yr] = short.ending_balance

            lt_beg.loc[yr] = long.beginning_balance
            lt_int.loc[yr] = long.interest_payments
            lt_pri.loc[yr] = long.principal_payments
            lt_end.loc[yr] = long.ending_balance

            kd_rate.loc[yr] = short.cost_of_debt
            new_lt_debt.loc[yr] = long.new_debt

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
            new_debt=new_lt_debt,
            principal_payments=lt_pri,
            ending_balance=lt_end,
        )
        return LoanTable(short_term=short_term, long_term=long_term)


class LoanSchedules:
    """Facade that builds and aggregates loan schedules when financing is needed."""

    def __init__(self):
        self.book = LoanBook()

    def new_loan(
        self,
        *,
        category: str,
        start_year: int,
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

    def debt_payments(self, year: int) -> DebtPay:
        return self.book.debt_payments(year)

    def remaining_debt(self, year: int) -> DebtBalances:
        return self.book.remaining_debt(year)

    def new_lt_debt(self, year: int) -> float:
        return self.book.new_lt_debt(year)

    def schedule_summary(self, year: int) -> LoanTable:
        return self.book.schedule_summary(year)

    def schedule_summary_series(self, years: Iterable[int]) -> LoanTable:
        return self.book.schedule_summary_series(years)

    def all(self) -> list[Loan]:
        return self.book.all()

    def __len__(self) -> int:
        return len(self.book)

    def __iter__(self):
        return iter(self.book)
