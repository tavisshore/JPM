# from __future__ import annotations

# from dataclasses import dataclass, field
# from typing import Iterable

# import pandas as pd
# from forecasting import Forecasting
# from input import InputData


# class LoanSchedules:
#     def __init__(self, cash_budget):
#         self.years = pd.Index([0])

#         # ---------- Short-Term Loan Schedule (11a) ----------
#         self.st_bb = pd.Series(0.0)  # Row 168 – Beginning balance
#         self.st_ip = pd.Series(0.0)  # Row 169 – Interest payment
#         self.st_pp = pd.Series(0.0)  # Row 170 – Principal payment
#         first_eb = (
#             cash_budget.st_loan_inflow.iloc[0]
#             if cash_budget.st_loan_inflow is not None
#             else 0.0
#         )
#         self.st_eb: pd.Series(first_eb)  # Row 171 – Ending balance

#     # ---------- Long-Term Loan Schedule (11b) ----------
#     # lt_bb: pd.Series  # Row 176 – Beginning LT debt
#     # lt_pps: dict  # Row 178–185: individual principal payments per loan cohort
#     # lt_new_loans: dict  # Row 179–185: cohorts of new LT loans
#     # lt_total_interest: pd.Series  # Row 189 – Total LT interest
#     # lt_new_debt: pd.Series  # Row 190 – New LT debt (column sum of new loan cohorts)
#     # lt_total_pp: pd.Series  # Row 191 – Total principal payment
#     # lt_eb: pd.Series  # Row 192 – Ending LT debt
#         bb_lt_debt = pd.Series(0.0)  # Row 176
#         lt_loans = [cash_budget.lt_new_loan_3]  # Row 179–185


#     def add_year(
#         self,
#         year,
#         input_data,
#         cash_budget,  # Module 3 provides loan inflows
#         forecast,  # Table 4 (Kd series)
#     ):
#         # ----- short-term parameters -----
#         kd = forecast.cost_of_debt.iloc[year]  # Kd_t

#         # ST loan inflow from Module 3 (row 139)
#         st_inflow = cash_budget.st_loan_inflow.loc[year]

#         # concat

#         self.st_bb

#         # ---------- SHORT-TERM SCHEDULE (Table 11a) ----------
#         st_bb = self.st_eb.iloc[year - 1]  # previous EB
#         st_ip = st_bb * kd
#         st_pp = st_bb / input_data.st_years_loan_2  # ST loan term
#         if year == 0:
#             st_eb = st_inflow
#         else:
#             st_eb = self.eb.iloc[year - 1] + st_inflow - st_pp + st_ip
#         # if year == 0: eb = st_inflow

#         # Add values at current year
#         self.st_bb = pd.concat(
#             [self.st_bb, pd.Series([st_bb], index=[year])], ignore_index=False
#         )
#         self.st_ip = pd.concat(
#             [self.st_ip, pd.Series([st_ip], index=[year])], ignore_index=False
#         )
#         self.st_pp = pd.concat(
#             [self.st_pp, pd.Series([st_pp], index=[year])], ignore_index=False
#         )
#         self.st_eb = pd.concat(
#             [self.st_eb, pd.Series([st_eb], index=[year])], ignore_index=False
#         )

#         # ---------- LONG-TERM SCHEDULE (Table 11b) ----------


# def _make_year_index(start, periods: int, like: pd.Index) -> pd.Index:
#     """
#     Create a year sequence starting at `start` with `periods` length,
#     matching (Int/Datetime/Period)
#     """
#     if periods <= 0:
#         raise ValueError("periods must be positive")
#     if start is None:
#         raise ValueError("start year is required to build a schedule")
#     if isinstance(like, pd.DatetimeIndex) or isinstance(start, pd.Timestamp):
#         start_ts = pd.Timestamp(start)
#         return pd.date_range(start=start_ts, periods=periods, freq="YS")
#     if isinstance(like, pd.PeriodIndex) or isinstance(start, pd.Period):
#         start_p = (
#             pd.Period(start, freq="Y")
#             if not isinstance(start, pd.Period)
#             else start.asfreq("Y")
#         )
#         return pd.period_range(start=start_p, periods=periods, freq="Y")
#     start_int = int(start)
#     return pd.Index(range(start_int, start_int + periods), dtype=int)


# @dataclass
# class Loan:
#     start_year: int
#     input_data: "InputData"  # Loan terms
#     forecast: Forecasting
#     loan_amount: float

#     _years: pd.Index = field(init=False, repr=False)
#     _df: pd.DataFrame = field(init=False, repr=False, default=None)

#     def __post_init__(self):
#         self._years = _make_year_index(
#             self.start_year, self.input_data.lt_years_loan_3,
# like=self.input_data.years
#         )
#         self._df = self._build_df()

#     @property
#     def years(self) -> pd.Index:
#         return self._years

#     def _build_df(self, kd) -> pd.DataFrame:
#         y = self.years

#         beg = pd.Series(0.0, index=y)
#         # Taken at the end of the year? First payment in year 1
#         beg.iloc[0] = self.loan_amount

#         term = int(self.input_data.lt_years_loan_3)
#         annual_principal = (self.loan_amount / term) if term > 0 else 0.0

#         principal = pd.Series(0.0, index=y)
#         interest = pd.Series(0.0, index=y)
#         end = pd.Series(0.0, index=y)

#         for i, _ in enumerate(y):
#             if i == 0:
#                 end.iloc[i] = beg.iloc[i]  # no payment in year 0
#                 continue

#             beg.iloc[i] = end.iloc[i - 1]
#             r = float(kd.iloc[i])
#             interest.iloc[i] = beg.iloc[i] * r
#             principal.iloc[i] = min(annual_principal, beg.iloc[i])
#             end.iloc[i] = beg.iloc[i] - principal.iloc[i]

#         total = interest + principal

#         return pd.DataFrame(
#             {
#                 "Beginning balance": beg,
#                 f"Interest payment {self.category} loan": interest,
#                 f"Principal payments {self.category} loan": principal,
#                 f"Total payment {self.category} loan": total,
#                 "Ending balance": end,
#                 "Interest rate": kd,
#             },
#             index=y,
#         )

#     def compute(self, year) -> pd.Series:
#         df = self._df
#         if year not in df.index:
#             raise KeyError(f"{year!r} not in schedule index")
#         return df.loc[year]

#     def loan_ongoing(self, year) -> bool:
#         df = self._df
#         return year in df.index


# @dataclass
# class LoanBook:
#     loans: list[Loan] = field(default_factory=list)

#     def add(self, loan: Loan) -> None:
#         if isinstance(loan, Loan):
#             self.loans.append(loan)
#         else:
#             raise TypeError(f"Unsupported loan type: {type(loan)}")

#     def extend(self, loans: Iterable[Loan]) -> None:
#         for loan in loans:
#             self.add(loan)

#     def all(self) -> list[Loan]:
#         return self.loans

#     def __len__(self) -> int:
#         return len(self.loans)

#     def __iter__(self):
#         return iter(self.all())

#     def debt_payments(self, year) -> DebtPay:
#         """
#         Aggregates debt payments for a given year using the loan schedules.
#         """
#         st_interest, st_principal = 0.0, 0.0
#         lt_interest, lt_principal = 0.0, 0.0

#         for loan in self.loans:
#             if loan.loan_ongoing(year):  # Remove spent loans?
#                 row = loan.compute(year)
#                 if loan.category == "LT":
#                     lt_interest += float(row["Interest payment LT loan"])
#                     lt_principal += float(row["Principal payments LT loan"])
#                 elif loan.category == "ST":
#                     st_interest += float(row["Interest payment ST loan"])
#                     st_principal += float(row["Principal payments ST loan"])

#         return DebtPay(
#             st_interest=st_interest,
#             st_principal=st_principal,
#             lt_interest=lt_interest,
#             lt_principal=lt_principal,
#             total=st_interest + st_principal + lt_interest + lt_principal,
#         )

#     def remaining_debt(self, year):
#         """
#         Return ST and LT debt totals
#         """
#         st_total, lt_total = 0.0, 0.0

#         for loan in self.loans:
#             if loan.loan_ongoing(year):  # Remove spent loans?
#                 row = loan.compute(year)
#                 if loan.category == "LT":
#                     lt_total += float(row["Ending balance"])  # Or end?
#                 elif loan.category == "ST":
#                     st_total += float(row["Ending balance"])

#         return DebtBalances(
#             st_debt=st_total, lt_debt=lt_total, total=st_total + lt_total
#         )
#         )
