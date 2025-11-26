from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from income_statement import IncomeStatement
    from loans import LoanSchedules
    from trans import Transactions
    from value import DepreciationSchedule, InventorySchedule, SalesPurchasesSchedule


@dataclass(frozen=True)
class BalanceSheet:
    """Balance sheet assembled from transaction, loan, and inventory schedules."""

    years: pd.Index

    cash_cb: pd.Series
    ar_it: pd.Series
    inventory_it: pd.Series
    app_it: pd.Series
    st_investments_cb: pd.Series
    current_assets: pd.Series
    net_fixed_assets_it: pd.Series
    total_assets: pd.Series

    ap_it: pd.Series
    apr: pd.Series
    short_term_debt_cb: pd.Series
    current_liabilities: pd.Series
    long_term_debt_cb: pd.Series
    total_liabilities: pd.Series

    equity_investment_cb: pd.Series
    retained_earnings_is: pd.Series
    current_year_ni: pd.Series
    repurchase_of_equity: pd.Series

    total_liabilities_and_equity: pd.Series
    check: pd.Series

    @classmethod
    def from_inputs(
        cls,
        years: pd.Index,
        transactions: "Transactions",
        sales_purchases: "SalesPurchasesSchedule",
        inventory: "InventorySchedule",
        depreciation: "DepreciationSchedule",
        loanbook: "LoanSchedules",
        income_statement: "IncomeStatement",
    ) -> "BalanceSheet":
        years = pd.Index(years)

        cash_cb = transactions.discretionary.cumulated_ncb

        ar_it = sales_purchases.credit_sales
        inventory_it = inventory.final_inventory_value
        app_it = sales_purchases.advance_payment_to_suppliers
        st_investments_cb = transactions.discretionary.st_investments
        net_fixed_assets_it = depreciation.net_fixed_assets
        current_assets = cash_cb + ar_it + inventory_it + app_it + st_investments_cb
        total_assets = current_assets + net_fixed_assets_it

        ap_it = sales_purchases.purchases_on_credit
        apr = sales_purchases.advance_payments_from_customers
        loan_summary = loanbook.book.schedule_summary_series(years)
        short_term_debt_cb = loan_summary.short_term.ending_balance
        long_term_debt_cb = loan_summary.long_term.ending_balance

        current_liabilities = ap_it + apr + short_term_debt_cb
        total_liabilities = current_liabilities + long_term_debt_cb

        equity_cb_vals = []
        eq_cb_prev = 0.0
        for y in years:
            eq_cb_curr = eq_cb_prev + transactions.owner.invested_equity.loc[y]
            equity_cb_vals.append(eq_cb_curr)
            eq_cb_prev = eq_cb_curr
        equity_investment_cb = pd.Series(equity_cb_vals, index=years)

        retained_earnings_is = income_statement.cre
        current_year_ni = income_statement.net_income

        repurchase_vals = []
        rep_prev = 0.0
        for y in years:
            rep_curr = rep_prev - transactions.owner.repurchased_stock.loc[y]
            repurchase_vals.append(rep_curr)
            rep_prev = rep_curr
        repurchase_of_equity = pd.Series(repurchase_vals, index=years)

        total_liabilities_and_equity = (
            total_liabilities
            + equity_investment_cb
            + retained_earnings_is
            + current_year_ni
            + repurchase_of_equity
        )

        check = total_liabilities_and_equity - total_assets

        return cls(
            years=years,
            cash_cb=cash_cb,
            ar_it=ar_it,
            inventory_it=inventory_it,
            app_it=app_it,
            st_investments_cb=st_investments_cb,
            current_assets=current_assets,
            net_fixed_assets_it=net_fixed_assets_it,
            total_assets=total_assets,
            ap_it=ap_it,
            apr=apr,
            short_term_debt_cb=short_term_debt_cb,
            current_liabilities=current_liabilities,
            long_term_debt_cb=long_term_debt_cb,
            total_liabilities=total_liabilities,
            equity_investment_cb=equity_investment_cb,
            retained_earnings_is=retained_earnings_is,
            current_year_ni=current_year_ni,
            repurchase_of_equity=repurchase_of_equity,
            total_liabilities_and_equity=total_liabilities_and_equity,
            check=check,
        )
