from dataclasses import dataclass

import pandas as pd


@dataclass
class Transactions:
    def __init__(self, years, owners, discretionary, forecasting):
        self.years = years
        self.calculated_cumulated_ncb = pd.Series(dtype=float)
        self.check_with_mct = pd.Series(dtype=float)

        self.owner_transactions = owners
        self.discretionary_transactions = discretionary
        self.add_year(0, forecasting)

    def add_year(self, year, forecasting):
        if year == 0:
            calculated_cum_ncb = self.discretionary_transactions.year_ncb.loc[year]
        else:
            previous_cum_ncb = self.calculated_cumulated_ncb.loc[year - 1]
            calculated_cum_ncb = (
                previous_cum_ncb + self.discretionary_transactions.year_ncb.loc[year]
            )

        if self.calculated_cumulated_ncb.empty:
            self.calculated_cumulated_ncb = pd.Series(
                [calculated_cum_ncb], index=[year], dtype=float
            )
        else:
            self.calculated_cumulated_ncb = pd.concat(
                [
                    self.calculated_cumulated_ncb,
                    pd.Series([calculated_cum_ncb], index=[year], dtype=float),
                ]
            )

        check_with_mct = (
            calculated_cum_ncb - forecasting.minimum_cash_required.loc[year]
        )

        if self.check_with_mct.empty:
            self.check_with_mct = pd.Series([check_with_mct], index=[year], dtype=float)
        else:
            self.check_with_mct = pd.concat(
                [
                    self.check_with_mct,
                    pd.Series([check_with_mct], index=[year], dtype=float),
                ]
            )


@dataclass
class OwnerTransactions:
    years: pd.Index

    invested_equity: pd.Series  # Row 150
    dividends: pd.Series  # Row 151
    repurchased_stock: pd.Series  # Row 152
    payments_to_owners: pd.Series  # Row 153
    ncb_with_owners: pd.Series  # Row 154
    ncb_previous_modules: pd.Series  # Row 155

    # -------------------------------------------------------------
    #  YEAR 0 INITIALIZATION
    # -------------------------------------------------------------
    @classmethod
    def initial(
        cls,
        policy,
        cash_budget,
        depreciation,
    ):
        year = 0
        idx = pd.Index([year])

        debt_financing_pct = policy.debt_financing_pct
        stock_repurchase_pct = policy.stock_repurchase_pct
        # Row 150 – Invested equity
        invested_equity_0 = (
            (cash_budget.lt_loan_inflow.loc[year] / debt_financing_pct)
            * (1 - debt_financing_pct)
            if debt_financing_pct > 0
            else 0.0
        )

        # Override with minimum initial cash (like Excel)
        # minimum_initial_cash = policy.minimum_initial_cash
        # invested_equity_0 = minimum_initial_cash

        # Row 151 – Dividends = 0 at year 0
        dividends_0 = 0.0  # From the future - previous years row 207

        # print(stock_repurchase_pct)
        # Row 152 – Repurchased stock
        repurchased_stock_0 = (
            depreciation.annual_depreciation.loc[year] * stock_repurchase_pct.loc[year]
        )

        # Row 153 – Payments to owners
        payments_0 = dividends_0 + repurchased_stock_0

        # Row 154 – NCB with owners = IE - payments
        ncb_with_owners_0 = invested_equity_0 - payments_0

        # Row 155 – NCB previous modules = row154 + module3 + module2
        ncb_financing_activities = cash_budget.ncb_financing_activities.loc[year]
        ncb_after_capex = cash_budget.ncb_after_capex.loc[year]
        ncb_prev_0 = ncb_with_owners_0 + ncb_financing_activities + ncb_after_capex

        return cls(
            years=idx,
            invested_equity=pd.Series([invested_equity_0], index=idx),
            dividends=pd.Series([dividends_0], index=idx),
            repurchased_stock=pd.Series([repurchased_stock_0], index=idx),
            payments_to_owners=pd.Series([payments_0], index=idx),
            ncb_with_owners=pd.Series([ncb_with_owners_0], index=idx),
            ncb_previous_modules=pd.Series([ncb_prev_0], index=idx),
        )

    # -------------------------------------------------------------
    #  ADD YEAR t > 0
    # -------------------------------------------------------------
    def add_year(self, year, policy, cash_budget):
        # Row 150 – Invested equity
        debt_financing_pct = policy.debt_financing_pct

        # Row 150 – Invested equity
        invested_equity_t = (
            (cash_budget.lt_loan_inflow.loc[year] / debt_financing_pct)
            * (1 - debt_financing_pct)
            if debt_financing_pct > 0
            else 0.0
        )

        # Row 154 – NCB with owners
        payments_t = self.payments_to_owners.iloc[year]  # from current year addition
        ncb_with_owners_t = invested_equity_t - payments_t

        # Row 155 – NCB previous modules
        ncb_prev_t = (
            ncb_with_owners_t
            + cash_budget.ncb_financing_activities.loc[year]
            + cash_budget.ncb_after_capex.loc[year]
        )
        ncb_prev_t = float(ncb_prev_t)

        # new index
        new_index = self.years.append(pd.Index([year]))

        self.years = new_index
        self.invested_equity = pd.concat(
            [
                self.invested_equity,
                pd.Series([invested_equity_t], index=pd.Index([year])),
            ]
        )
        self.ncb_with_owners = pd.concat(
            [
                self.ncb_with_owners,
                pd.Series([ncb_with_owners_t], index=pd.Index([year])),
            ]
        )
        self.ncb_previous_modules = pd.concat(
            [self.ncb_previous_modules, pd.Series([ncb_prev_t], index=pd.Index([year]))]
        )

    def owner_payments(self, year, income_statement, depreciation, policy) -> float:
        stock_repurchase_pct = policy.stock_repurchase_pct.loc[year]
        # Row 151 – Dividends
        dividends_t = income_statement.next_year_dividends.loc[year - 1]

        # Row 152 – Repurchased stock = depreciation * repurchase %
        repurchased_stock_t = (
            depreciation.annual_depreciation.loc[year] * stock_repurchase_pct
        )

        # Row 153 – Payments to owners
        payments_t = dividends_t + repurchased_stock_t

        self.payments_to_owners = pd.concat(
            [self.payments_to_owners, pd.Series([payments_t], index=pd.Index([year]))]
        )
        self.dividends = pd.concat(
            [self.dividends, pd.Series([dividends_t], index=pd.Index([year]))]
        )
        self.repurchased_stock = pd.concat(
            [
                self.repurchased_stock,
                pd.Series([repurchased_stock_t], index=pd.Index([year])),
            ]
        )
        return payments_t


@dataclass
class DiscretionaryTransactions:
    years: pd.Index
    redemption_st_investment: pd.Series
    return_from_st_investment: pd.Series
    total_inflow_st_investment: pd.Series
    st_investments: pd.Series
    ncb_discretionary_transactions: pd.Series
    year_ncb: pd.Series
    cumulated_ncb: pd.Series

    @classmethod
    def from_inputs(
        cls,
        policy,
        cash_budget,
        owner_tx,
        forecast,
    ) -> "DiscretionaryTransactions":
        year = 0

        min_cash = float(policy.minimum_initial_cash)  # D50

        # --- 157: Redemption of ST investment ---
        # Last years st_investments
        redemption_from_previous = 0.0
        return_from_st = redemption_from_previous * float(
            forecast.return_st_investment.loc[year]
        )
        total_inflow = redemption_from_previous + return_from_st

        # row 160: ST investments
        # IF(D139 + D140 + D150 > 0, 0,
        #    C163 + D155 + D159 - D50)
        external_financing = (
            cash_budget.st_loan_inflow.loc[year]
            + cash_budget.lt_loan_inflow.loc[year]
            + owner_tx.invested_equity.loc[year]
        )
        st_inv = 0.0
        if external_financing > 0:
            previous_cum_ncb = 0.0

            available = (
                previous_cum_ncb
                + owner_tx.ncb_previous_modules.loc[year]
                + total_inflow
                - min_cash
            )
            st_inv = max(0.0, available)  # final check??

        # row 161: NCB of discretionary transactions
        ncb_disc = total_inflow - st_inv

        # row 162: Year NCB = NCB_prev_modules + NCB_discretionary
        year_ncb = owner_tx.ncb_previous_modules.loc[year] + ncb_disc

        # row 163: Cumulated NCB
        cum_ncb = forecast.minimum_cash_required.loc[year]

        return cls(
            years=pd.Index([year]),
            redemption_st_investment=pd.Series(
                [redemption_from_previous], index=pd.Index([year])
            ),
            return_from_st_investment=pd.Series(
                [return_from_st], index=pd.Index([year])
            ),
            total_inflow_st_investment=pd.Series(
                [total_inflow], index=pd.Index([year])
            ),
            st_investments=pd.Series([st_inv], index=pd.Index([year])),
            ncb_discretionary_transactions=pd.Series(
                [ncb_disc], index=pd.Index([year])
            ),
            year_ncb=pd.Series([year_ncb], index=pd.Index([year])),
            cumulated_ncb=pd.Series([cum_ncb], index=pd.Index([year])),
        )

    def add_year(self, year, policy, cash_budget, depreciation, income_statement):
        min_cash = float(policy.minimum_initial_cash)  # D50

        previous_st_investment = self.st_investments.loc[year - 1]
        redemption_from_previous = previous_st_investment
        return_from_st = self.return_from_st_investment.loc[
            year
        ]  # Add already with below

        total_inflow = redemption_from_previous + return_from_st

        # row 160: ST investments
        # IF(D139 + D140 + D150 > 0, 0,
        #    C163 + D155 + D159 - D50)
        external_financing = (
            cash_budget.st_loan_inflow.loc[year]
            + cash_budget.lt_loan_inflow.loc[year]
            + income_statement.invested_equity.loc[year]
        )
        previous_cum_ncb = self.cumulated_ncb.loc[year - 1]
        st_inv = 0.0
        if external_financing > 0:
            available = (
                previous_cum_ncb
                + income_statement.ncb_previous_modules.loc[year]
                + total_inflow
                - min_cash
            )
            st_inv = max(0.0, available)  # final check??

        # row 161: NCB of discretionary transactions
        ncb_disc = total_inflow - st_inv

        # row 162: Year NCB = NCB_prev_modules + NCB_discretionary
        year_ncb = income_statement.ncb_previous_modules.loc[year] + ncb_disc

        # row 163: Cumulated NCB
        cum_ncb = previous_cum_ncb + year_ncb
        # new index
        new_index = self.years.append(pd.Index([year]))

        self.years = new_index
        self.redemption_st_investment = pd.concat(
            [
                self.redemption_st_investment,
                pd.Series([redemption_from_previous], index=pd.Index([year])),
            ]
        )
        self.total_inflow_st_investment = pd.concat(
            [
                self.total_inflow_st_investment,
                pd.Series([total_inflow], index=pd.Index([year])),
            ]
        )
        self.st_investments = pd.concat(
            [self.st_investments, pd.Series([st_inv], index=pd.Index([year]))]
        )
        self.ncb_discretionary_transactions = pd.concat(
            [
                self.ncb_discretionary_transactions,
                pd.Series([ncb_disc], index=pd.Index([year])),
            ]
        )
        self.year_ncb = pd.concat(
            [self.year_ncb, pd.Series([year_ncb], index=pd.Index([year]))]
        )
        self.cumulated_ncb = pd.concat(
            [self.cumulated_ncb, pd.Series([cum_ncb], index=pd.Index([year]))]
        )

    def get_st_returns(self, year, income_statement) -> float:
        # --- 157: Redemption of ST investment ---
        previous_st_investment = self.st_investments.loc[year - 1]
        redemption_from_previous = previous_st_investment
        return_from_st = redemption_from_previous * float(
            income_statement.return_from_st_investment.loc[year]
        )
        self.return_from_st_investment = pd.concat(
            [
                self.return_from_st_investment,
                pd.Series([return_from_st], index=pd.Index([year])),
            ]
        )
        return return_from_st
