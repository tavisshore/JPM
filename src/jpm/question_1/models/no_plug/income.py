from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from jpm.question_1.models.no_plug.investments import InvestmentBook
from jpm.question_1.models.no_plug.loans import LoanBook


@dataclass
class IncomeStatement:
    years: pd.Index
    ebit: pd.Series
    loanbook: LoanBook
    investmentbook: InvestmentBook

    vp_model: bool = True
    retained_earnings: pd.Series = field(init=False, repr=False)
    net_income: pd.Series = field(init=False, repr=False)
    dividends: pd.Series = field(init=False, repr=False)

    def __post_init__(self):
        z = pd.Series(0.0, index=self.years)
        object.__setattr__(self, "retained_earnings", z.copy())
        object.__setattr__(self, "net_income", z.copy())
        object.__setattr__(self, "dividends", z.copy())

    def generate(self, year, dividends: float = 0.0) -> pd.Series:
        """
        vp example gives 100% of net income as dividends next year
        """
        ebit = float(self.ebit.reindex(self.years).get(year, 0.0))
        investment_returns = self.investmentbook.investment_income(year)
        st_returns_interest = investment_returns.interest

        loans_values = self.loanbook.debt_payments(year)
        loan_interest = loans_values.st_interest + loans_values.lt_interest

        net_income = ebit + st_returns_interest - loan_interest

        if self.vp_model:  # Following the paper's assumptions
            dividends = self.net_income.loc[year - 1]

        self.net_income.at[year] = net_income
        self.dividends.at[year] = dividends

        retained = (
            self.retained_earnings.loc[year - 1]
            + self.net_income.at[year - 1]
            - dividends
        )
        self.retained_earnings.at[year] = retained

        return pd.Series(
            {
                "EBIT": ebit,
                "Return (interest) from ST investment": st_returns_interest,
                "Interest payments (ST+LT)": loan_interest,
                "Net income": net_income,
                "Dividends (payment next year) ==> CB": dividends,
                "Cumulated retained earnings": self.retained_earnings.at[year - 1],
            }
        )
