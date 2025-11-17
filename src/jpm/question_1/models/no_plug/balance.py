from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .cash import CashBudget
from .income import IncomeStatement
from .input import InputData
from .investments import InvestmentBook
from .loans import LoanBook


@dataclass
class BalanceSheet:
    years: pd.Index
    input_data: InputData
    cashbudget: CashBudget
    income_statement: IncomeStatement
    loanbook: LoanBook
    investmentbook: InvestmentBook

    history: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self):
        z = pd.Series(0.0, index=self.years)
        object.__setattr__(self, "history", z.copy())

    def generate(self, year) -> pd.Series:
        """
        vp example gives 100% of net income as dividends next year
        """
        # Assets
        cash = self.cashbudget.history.loc[year]["Cumulated NCB"]
        st_investments = self.cashbudget.history.loc[year]["ST investments"]
        total_fixed_assets = self.input_data.net_fixed_assets.loc[year]
        total_assets = cash + st_investments + total_fixed_assets

        # Liabilities & Equity
        debt = self.loanbook.remaining_debt(year)
        equity_investment = self.input_data.equity_investment.iloc[: year + 1].sum()

        net_income_current_year = self.income_statement.net_income.at[year]
        retained_earnings = self.income_statement.retained_earnings.at[year]
        total_liabilities_and_equity = (
            debt.total + equity_investment + net_income_current_year + retained_earnings
        )

        check = total_liabilities_and_equity - total_assets

        output = pd.Series(
            {
                # Assets
                "Cash": cash,
                "ST Investments": st_investments,
                "Total Fixed Assets": total_fixed_assets,
                "Total": total_assets,
                # Liabilities
                "Short-Term Debt": debt.st_debt,
                "Long-Term Debt": debt.lt_debt,
                # Equity
                "Equity Investment": equity_investment,
                "Net Income": net_income_current_year,
                "Retained Earnings": retained_earnings,
                "Total Liabilities & Equity": total_liabilities_and_equity,
                # Check
                "Check": check,
            }
        )
        # self.history[year] = output
        return output
