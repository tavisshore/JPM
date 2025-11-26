from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BalanceSheet:
    years: pd.Index

    # Assets
    cash_cb: pd.Series  # row 213  (Cash CB)          = D163
    ar_it: pd.Series  # row 214  (AR IT)            = D105 (intermediate)
    inventory_it: pd.Series  # row 215  (Inventory IT)     = D93
    app_it: pd.Series  # row 216  (APP IT)           = D119
    st_investments_cb: pd.Series  # row 217  (ST invest. CB)    = D160
    current_assets: pd.Series  # row 218  = SUM(D213:D217)
    net_fixed_assets_it: pd.Series  # row 219  (Net fixed assets) = D77
    total_assets: pd.Series  # row 220  = D219 + D218

    # Liabilities
    ap_it: pd.Series  # row 222  (AP IT)            = D109
    apr: pd.Series  # row 223  (APR)              = D115
    short_term_debt_cb: pd.Series  # row 224  (ST debt CB)       = D171
    current_liabilities: pd.Series  # row 225  = SUM(D222:D224)
    long_term_debt_cb: pd.Series  # row 226  (LT debt CB)       = D192
    total_liabilities: pd.Series  # row 227  = D226 + D225

    # Equity
    equity_investment_cb: pd.Series  # row 228  = C228 + D150
    retained_earnings_is: pd.Series  # row 229  (CRE)              = D208
    current_year_ni: pd.Series  # row 230  (NI)               = D206
    repurchase_of_equity: pd.Series  # row 231  = C231 - D152

    # Totals & check
    total_liabilities_and_equity: pd.Series  # row 232 = SUM(D228:D231)+D227
    check: pd.Series  # row 233 = D232 - D220

    # -----------------------------------------------------------------
    # Constructor from other modules
    # -----------------------------------------------------------------
    @classmethod
    def from_inputs(
        cls,
        years: pd.Index,
        transactions,
        # Assets-side inputs
        cash_cb: pd.Series,  # usually DiscretionaryTransactions.cumulated_ncb
        ar_it: pd.Series,  # Accounts receivable IT (Table 8)
        inventory_it: pd.Series,  # F
        app_it: pd.Series,  # Advance payments paid IT (to suppliers)
        st_investments_cb: pd.Series,  # DiscretionaryTransactions.st_investments
        net_fixed_assets_it: pd.Series,  # DepreciationSchedule.net_fixed_assets
        # Liabilities inputs
        ap_it: pd.Series,  # Accounts payable IT
        apr: pd.Series,  # Advance payments received (from customers)
        short_term_debt_cb: pd.Series,  # LoanSchedules.st_eb
        long_term_debt_cb: pd.Series,  # LoanSchedules.lt_eb
        # Equity inputs
        invested_equity: pd.Series,  # OwnerTransactions.invested_equity (row 150)
        retained_earnings_is: pd.Series,  # CRE (row 229) from IS / CRE module
        current_year_ni: pd.Series,  # IncomeStatement.net_income (row 206)
        repurchased_stock: pd.Series,  # OwnerTransactions.repurchased_stock (row 152)
    ) -> "BalanceSheet":
        years = pd.Index(years)

        # make sure all series are aligned & indexed by years
        def align(s: pd.Series) -> pd.Series:
            return s.reindex(years).fillna(0.0)

        cash_cb = align(cash_cb)
        ar_it = align(ar_it)
        inventory_it = align(inventory_it)
        app_it = align(app_it)
        st_investments_cb = align(st_investments_cb)
        net_fixed_assets_it = align(net_fixed_assets_it)
        ap_it = align(ap_it)
        apr = align(apr)
        short_term_debt_cb = align(short_term_debt_cb)
        long_term_debt_cb = align(long_term_debt_cb)
        invested_equity = align(invested_equity)
        retained_earnings_is = align(retained_earnings_is)
        current_year_ni = align(current_year_ni)
        repurchased_stock = align(repurchased_stock)

        # ---------------- ASSETS ----------------
        current_assets = (
            cash_cb + ar_it + inventory_it + app_it + st_investments_cb
        )  # row 218
        total_assets = current_assets + net_fixed_assets_it  # row 220

        # ---------------- LIABILITIES ----------------
        current_liabilities = ap_it + apr + short_term_debt_cb  # row 225
        total_liabilities = current_liabilities + long_term_debt_cb  # row 227

        # ---------------- EQUITY FLOWS (cumulative) ----------------
        # row 228: Equity investment CB = previous CB + current invested equity
        equity_cb_vals = []
        eq_cb_prev = 0.0
        for y in years:
            eq_cb_curr = eq_cb_prev + invested_equity.loc[y]
            equity_cb_vals.append(eq_cb_curr)
            eq_cb_prev = eq_cb_curr
        equity_investment_cb = pd.Series(equity_cb_vals, index=years)

        # row 231: Repurchase of equity = previous CB - current repurchased stock
        repurchase_vals = []
        rep_prev = 0.0
        for y in years:
            rep_curr = rep_prev - repurchased_stock.loc[y]
            repurchase_vals.append(rep_curr)
            rep_prev = rep_curr
        repurchase_of_equity = pd.Series(repurchase_vals, index=years)

        # ---------------- TOTAL L&E AND CHECK ----------------
        total_liabilities_and_equity = (
            total_liabilities
            + equity_investment_cb
            + retained_earnings_is
            + current_year_ni
            + repurchase_of_equity
        )  # row 232

        check = total_liabilities_and_equity - total_assets  # row 233

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
