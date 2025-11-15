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
from src.jpm.question_1.no_plug import (
    BalanceSheet,
    CashBudget,
    IncomeStatement,
    InputData,
    InvestmentBook,
    LoanBook,
)

from src.jpm.question_1.misc import as_series

if __name__ == "__main__":
    years = pd.Index([0, 1, 2, 3], name="year")

    input_data = InputData(
        years=years,
        ebit=as_series({0: 0, 1: 5, 2: 9, 3: 12.0}, years),
        depreciation=as_series({0: 0, 1: 9, 2: 9, 3: 9}, years),
        net_fixed_assets=as_series({0: 45, 1: 36, 2: 27, 3: 18}, years),
        min_cash=as_series({0: 10, 1: 10, 2: 10, 3: 10}, years),
        kd=as_series({0: 0, 1: 0.13, 2: 0.13, 3: 0.13}, years),
        rtn_st_inv=as_series({0: 0, 1: 0.08, 2: 0.08, 3: 0.08}, years),
        equity_investment=as_series({0: 25, 1: 0, 2: 0, 3: 0}, years),
        st_loan_term=1,
        lt_loan_term=5,
    )

    loanbook = LoanBook()
    investmentbook = InvestmentBook()

    i_s = IncomeStatement(
        years=years,
        ebit=input_data.ebit,
        loanbook=loanbook,
        investmentbook=investmentbook,
    )

    cb = CashBudget(
        input=input_data, years=years, loanbook=loanbook, investmentbook=investmentbook
    )

    bs = BalanceSheet(
        years=years,
        input_data=input_data,
        cashbudget=cb,
        income_statement=i_s,
        loanbook=loanbook,
        investmentbook=investmentbook,
    )

    # Year 0 - has different characteristics with VP
    cb0 = cb.generate_0()

    # Year 1
    is1 = i_s.generate(year=1, dividends=0.0)
    cb1 = cb.generate(
        year=1,
        equity_contrib=0,
        dividends=0,
    )

    # Year 2
    is2 = i_s.generate(year=2)
    cb2 = cb.generate(
        year=2,
        equity_contrib=0,
        dividends=0.0,
    )

    # # Year 3
    # is3 = i_s.generate(year=3)
    # cb3 = cb.generate(
    #     year=3,
    #     equity_contrib=0,
    #     dividends=0.0,
    # )

    # # Calculate the final BS
    # b0 = bs.generate(year=0)
    # b1 = bs.generate(year=1)
    # b2 = bs.generate(year=2)
    # b3 = bs.generate(year=3)

    # print()
    # print()

    # print(b0)
    # print()

    # print(b1)
    # print()

    # print(b2)
    # print()

    # print(b3)
    # print()
