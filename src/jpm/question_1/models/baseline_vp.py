"""
Valez-Pareja Full Financial Statements Model
- This is broken into 6 modules:
1. Input Data
2. Intermediate Tables
3. Cash Budget
4. Debt Schedule
5. Income Statement
6. Balance Sheet

Assumptions made with the example:
1. A startup firm (starting from zero).
2. Taxes are paid the same year as accrued
3. All the expenses and sales are paid and received on a cash basis.
4. Dividends are 100% of the Net Income of previous year and are paid the next year
after the Net Income is generated.
5. Any deficit is covered by new debt.
6. Deficit in the operating module (Module 1) should be covered with short term loans.
Short term loans will be repaid the following year.
7. Deficit in the investment in fixed assets module (Module 2) should be covered with
long term loans. Long term loans are repaid in 5 years.
8. Any cash excess above the targeted level is invested in market securities.
9. In this example we only consider two types of debt: one long term loan and short
term loans (for illustration purposes).
10. Short term portion of debt is not considered in the current liabilities.
"""

from __future__ import annotations

import pandas as pd

from src.jpm.question_1.components import (
    CashBudget,
    IncomeStatement,
    InputData,
    InvestmentBook,
    LoanBook,
)
from src.jpm.question_1.misc import as_series

if __name__ == "__main__":
    years = pd.Index([0, 1, 2], name="year")

    input_data = InputData(
        years=years,
        ebit=as_series({0: 0, 1: 5, 2: 9}, years),
        depreciation=as_series({0: 0, 1: 9, 2: 9}, years),
        net_fixed_assets=as_series({0: 45, 1: 36, 2: 27}, years),
        min_cash=as_series({0: 10, 1: 10, 2: 10}, years),
        kd=as_series({0: 0, 1: 0.13, 2: 0.13}, years),
        rtn_st_inv=as_series({0: 0, 1: 0.08, 2: 0.08}, years),
        equity_investment=25.0,
        st_loan_term=1,
        lt_loan_term=5,
    )

    loanbook = LoanBook()
    investmentbook = InvestmentBook()

    cb = CashBudget(input_data, years)

    i_s = IncomeStatement(
        years=years,
        ebit=input_data.ebit,
        loanbook=loanbook,
        investmentbook=investmentbook,
    )

    # Year 0
    cb0 = cb.year0(loanbook, investmentbook)

    # Year 1
    is1 = i_s.generate_statement(year=1, dividends=0.0)
    cb1 = cb.project_cb(
        year=1,
        equity_contrib=0,
        dividends=0,
        loanbook=loanbook,
        investmentbook=investmentbook,
    )

    # Year 2
    is2 = i_s.generate_statement(year=2, dividends=0.0)

    cb2 = cb.project_cb(
        year=2,
        equity_contrib=0,
        dividends=0.0,
        loanbook=loanbook,
        investmentbook=investmentbook,
    )

    print(is1)
    print()
    print(is2)

    # is1 = IncomeStatement(
    #     years=years,
    #     ebit=idata.ebit,                     # {1:5, 2:9}
    #     rtn_rate_st=idata.rtn_st_inv,        # {1:0.08, 2:0.08}
    #     st_invest_end_prev=st_invest_bs,     # 0 at year 0 ⇒ return in year 1 = 0
    #     st_interest=st_loan_sched.compute()["Interest payment ST loan"],
    #     lt_interest=lt_loan_sched.compute()["Interest payment LT loan"],
    #     dividends=0.0,
    # ).compute()

    # # Subsequent years - loop with some example data

    # print(is1.loc[1].round(1))
    print()
