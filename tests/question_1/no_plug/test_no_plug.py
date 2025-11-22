import pandas as pd
import pytest

from jpm.question_1.models.no_plug import (
    BalanceSheet,
    CashBudget,
    IncomeStatement,
    InputData,
    Investment,
    InvestmentBook,
    Loan,
    LoanBook,
)

unit = pytest.mark.unit
integration = pytest.mark.integration


def _as_series(years, values):
    data = dict.fromkeys(years, 0.0)
    data.update(values or {})
    return pd.Series(data, index=years, dtype=float)


def make_input_data(
    years,
    *,
    ebit=None,
    depreciation=None,
    net_fixed_assets=None,
    min_cash=None,
    kd=None,
    rtn_st_inv=None,
    equity_investment=None,
    st_loan_term=1,
    lt_loan_term=5,
):
    return InputData(
        years=pd.Index(years, name="year"),
        ebit=_as_series(years, ebit or {}),
        depreciation=_as_series(years, depreciation or {}),
        net_fixed_assets=_as_series(years, net_fixed_assets or {}),
        min_cash=_as_series(years, min_cash or {}),
        kd=_as_series(years, kd or {}),
        rtn_st_inv=_as_series(years, rtn_st_inv or {}),
        equity_investment=_as_series(years, equity_investment or {}),
        st_loan_term=st_loan_term,
        lt_loan_term=lt_loan_term,
    )


@unit
def test_investment_schedule_returns_principal_on_maturity():
    """Investment schedule should accrue interest and redeem
    principal at maturity."""
    years = [0, 1]
    input_data = make_input_data(years, rtn_st_inv={1: 0.08})

    inv = Investment(input=input_data, amount=100, start_year=0, term_years=1)
    df = inv.df

    assert df.at[1, "Interest income"] == pytest.approx(8.0)
    assert df.at[1, "Principal redeemed"] == pytest.approx(100.0)
    assert df.at[1, "Ending balance"] == pytest.approx(0.0)


@unit
def test_investment_book_income_aggregates_active_investments():
    """InvestmentBook should sum income only for positions active in
    the queried year."""
    years = [0, 1, 2]
    data = make_input_data(years, rtn_st_inv={1: 0.05, 2: 0.05})
    book = InvestmentBook()
    book.add(Investment(input=data, amount=50, start_year=0, term_years=1))
    book.add(Investment(input=data, amount=20, start_year=1, term_years=1))

    income_year_1 = book.investment_income(1)
    assert income_year_1.interest == pytest.approx(50 * 0.05)
    assert income_year_1.principal_in == pytest.approx(50.0)

    income_year_2 = book.investment_income(2)
    assert income_year_2.interest == pytest.approx(20 * 0.05)
    assert income_year_2.principal_in == pytest.approx(20.0)


@unit
def test_loanbook_debt_payments_split_short_and_long_term():
    """LoanBook should separate short- and long-term principal/interest payments."""
    years = [0, 1, 2]
    data = make_input_data(
        years,
        kd={1: 0.1, 2: 0.1},
        st_loan_term=1,
        lt_loan_term=3,
    )
    loanbook = LoanBook()
    loanbook.add(
        Loan(input=data, start_year=0, initial_draw=10.0, category="ST"),
    )
    loanbook.add(
        Loan(input=data, start_year=0, initial_draw=30.0, category="LT"),
    )

    payments = loanbook.debt_payments(year=1)
    assert payments.st_principal == pytest.approx(10.0)
    assert payments.st_interest == pytest.approx(1.0)  # 10 * 0.1
    assert payments.lt_principal == pytest.approx(10.0)  # 30 / lt_term
    assert payments.lt_interest == pytest.approx(3.0)  # 30 * 0.1


@integration
def test_cash_budget_draws_short_term_loan_on_cash_shortfall():
    """Cash budget should raise a new short-term loan whenever
    min cash cannot be met."""
    years = [0, 1]
    data = make_input_data(
        years,
        min_cash={0: 5.0, 1: 5.0},
        kd={1: 0.1},
    )
    loanbook = LoanBook()
    investmentbook = InvestmentBook()
    cb = CashBudget(
        input=data,
        years=pd.Index(years),
        loanbook=loanbook,
        investmentbook=investmentbook,
    )

    cb.generate_0()
    existing_loans = len(loanbook.loans)

    cb.generate(year=1, equity_contrib=0.0, dividends=0.0)

    assert len(loanbook.loans) == existing_loans + 1
    new_loan = loanbook.loans[-1]
    assert new_loan.category == "ST"
    assert new_loan.initial_draw > 0.0
    assert cb.history.loc[1, "ST Loan"] == pytest.approx(new_loan.initial_draw)


@integration
def test_balance_sheet_remains_in_balance_for_simple_scenario():
    """Full no_plug flow should produce a balanced balance sheet for
    each simulated year."""
    years = pd.Index([0, 1, 2], name="year")
    data = make_input_data(
        years,
        ebit={1: 5.0, 2: 4.0},
        depreciation={1: 1.0, 2: 1.0},
        net_fixed_assets={0: 6.0, 1: 5.0, 2: 4.0},
        min_cash={0: 2.0, 1: 2.0, 2: 2.0},
        kd={1: 0.05, 2: 0.05},
        rtn_st_inv={1: 0.02, 2: 0.02},
        equity_investment={0: 8.0},
        st_loan_term=1,
        lt_loan_term=3,
    )
    loanbook = LoanBook()
    investmentbook = InvestmentBook()
    income = IncomeStatement(
        years=years,
        ebit=data.ebit,
        loanbook=loanbook,
        investmentbook=investmentbook,
    )
    cb = CashBudget(
        input=data,
        years=years,
        loanbook=loanbook,
        investmentbook=investmentbook,
    )
    cb.generate_0()

    for year in years[1:]:
        income.generate(year=year)
        dividends = float(income.dividends.reindex(years).at[year])
        equity_contrib = float(data.equity_investment.reindex(years).at[year])
        cb.generate(year=year, equity_contrib=equity_contrib, dividends=dividends)

    bs = BalanceSheet(
        years=years,
        input_data=data,
        cashbudget=cb,
        income_statement=income,
        loanbook=loanbook,
        investmentbook=investmentbook,
    )

    for year in years:
        row = bs.generate(year)
        assert row["Total"] == pytest.approx(
            row["Total Liabilities & Equity"], rel=1e-8, abs=1e-8
        )
