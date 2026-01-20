from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from cash import CashBudget
    from forecasting import Forecasting
    from income_statement import IncomeStatement
    from input import PolicyTable
    from value import DepreciationSchedule


@dataclass
class Transactions:
    """Aggregates owner and discretionary transactions with cumulative cash checks."""

    def __init__(
        self,
        years: pd.Index,
        owners: "OwnerTransactions",
        discretionary: "DiscretionaryTransactions",
        forecasting: "Forecasting",
    ):
        self.years: pd.Index = years
        self.calculated_cumulated_ncb = pd.Series(dtype=float)
        self.check_with_mct = pd.Series(dtype=float)

        self.owner: OwnerTransactions = owners
        self.discretionary: DiscretionaryTransactions = discretionary
        self.add_year(0, forecasting)

    def add_year(
        self,
        year: int,
        forecasting: "Forecasting" | None = None,
        policy_table: "PolicyTable" | None = None,
        cash_budget: "CashBudget" | None = None,
    ) -> None:
        if year == 0:
            calculated_cum_ncb = self.discretionary.year_ncb.loc[year]
        else:
            self.owner.add_year(
                year=year,
                policy=policy_table,
                cash_budget=cash_budget,
            )

            self.discretionary.add_year(
                year=year,
                policy=policy_table,
                cash_budget=cash_budget,
                forecast=forecasting,
                owners=self.owner,
            )

            previous_cum_ncb = self.calculated_cumulated_ncb.loc[year - 1]

            calculated_cum_ncb = (
                previous_cum_ncb + self.discretionary.year_ncb.loc[year]
            )

        if self.calculated_cumulated_ncb.empty:
            self.calculated_cumulated_ncb = pd.Series(
                [float(calculated_cum_ncb)], index=[year], dtype=float
            )
        else:
            self.calculated_cumulated_ncb = pd.concat(
                [
                    self.calculated_cumulated_ncb,
                    pd.Series([float(calculated_cum_ncb)], index=[year], dtype=float),
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
    """Tracks owner-related equity injections, dividends, and repurchases."""

    years: pd.Index

    invested_equity: pd.Series
    dividends: pd.Series
    repurchased_stock: pd.Series
    payments_to_owners: pd.Series
    ncb_with_owners: pd.Series
    ncb_previous_modules: pd.Series

    @classmethod
    def initial(
        cls,
        policy: "PolicyTable",
        cash_budget: "CashBudget",
        depreciation: "DepreciationSchedule",
    ) -> "OwnerTransactions":
        year = 0
        idx = pd.Index([year])

        debt_financing_pct = policy.debt_financing_pct
        stock_repurchase_pct = policy.stock_repurchase_pct

        invested_equity_0 = (
            (cash_budget.lt_loan_inflow / debt_financing_pct) * (1 - debt_financing_pct)
            if debt_financing_pct > 0
            else 0.0
        )

        dividends_0 = 0.0

        repurchased_stock_0 = (
            depreciation.annual_depreciation.loc[year] * stock_repurchase_pct.loc[year]
        )

        payments_0 = dividends_0 + repurchased_stock_0

        ncb_with_owners_0 = invested_equity_0 - payments_0

        ncb_financing_activities = cash_budget.ncb_financing_activities
        ncb_after_capex = cash_budget.ncb_after_capex
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

    def add_year(
        self, year: int, policy: "PolicyTable", cash_budget: "CashBudget"
    ) -> None:
        debt_financing_pct = policy.debt_financing_pct

        invested_equity_t = (
            (cash_budget.lt_loan_inflow / debt_financing_pct) * (1 - debt_financing_pct)
            if debt_financing_pct > 0
            else 0.0
        )

        payments_t = self.payments_to_owners.iloc[year]
        ncb_with_owners_t = invested_equity_t - payments_t

        ncb_prev_t = (
            ncb_with_owners_t
            + cash_budget.ncb_financing_activities
            + cash_budget.ncb_after_capex
        )
        ncb_prev_t = float(ncb_prev_t)

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

    def owner_payments(
        self,
        year: int,
        previous_is: "IncomeStatement",
        depreciation: "DepreciationSchedule",
        policy: "PolicyTable",
    ) -> float:
        stock_repurchase_pct = policy.stock_repurchase_pct.loc[year]

        dividends_t = previous_is.next_year_dividends

        repurchased_stock_t = (
            depreciation.annual_depreciation.loc[year] * stock_repurchase_pct
        )
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

    def pretty_print(self) -> str:
        """Pretty print OwnerTransactions data."""
        lines = ["=" * 60, "OwnerTransactions", "=" * 60]

        lines.append("\nInvested Equity:")
        lines.append(str(self.invested_equity))

        lines.append("\nDividends:")
        lines.append(str(self.dividends))

        lines.append("\nRepurchased Stock:")
        lines.append(str(self.repurchased_stock))

        lines.append("\nPayments to Owners:")
        lines.append(str(self.payments_to_owners))

        lines.append("\nNCB with Owners:")
        lines.append(str(self.ncb_with_owners))

        lines.append("\nNCB Previous Modules:")
        lines.append(str(self.ncb_previous_modules))

        lines.append("=" * 60)

        return "\n".join(lines)


@dataclass
class DiscretionaryTransactions:
    """Manages short-term investments and discretionary cash movements."""

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
        policy: "PolicyTable",
        cash_budget: "CashBudget",
        owner_tx: "OwnerTransactions",
        forecast: "Forecasting",
    ) -> "DiscretionaryTransactions":
        year = 0

        min_cash = float(policy.minimum_initial_cash)

        redemption_from_previous = 0.0
        return_from_st = redemption_from_previous * float(
            forecast.return_st_investment.loc[year]
        )
        total_inflow = redemption_from_previous + return_from_st
        external_financing = (
            cash_budget.st_loan_inflow
            + cash_budget.lt_loan_inflow
            + owner_tx.invested_equity.loc[year]
        )
        st_inv = 0.0
        if external_financing < 0:
            previous_cum_ncb = 0.0
            available = (
                previous_cum_ncb
                + owner_tx.ncb_previous_modules.loc[year]
                + total_inflow
                - min_cash
            )
            st_inv = max(0.0, available)

        ncb_disc = total_inflow - st_inv

        year_ncb = owner_tx.ncb_previous_modules.loc[year] + ncb_disc

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

    def add_year(
        self,
        year: int,
        policy: "PolicyTable",
        cash_budget: "CashBudget",
        forecast: "Forecasting",
        owners: "OwnerTransactions",
    ) -> None:
        self.years = self.years.append(pd.Index([year]))

        cum_ncb = forecast.minimum_cash_required.loc[year]

        redemption_from_st = self.st_investments.loc[year - 1] if year > 0 else 0.0
        return_from_st = forecast.return_st_investment.loc[year] * redemption_from_st
        total_inflow = redemption_from_st + return_from_st

        external_investing = (
            cash_budget.st_loan_inflow
            + cash_budget.lt_loan_inflow
            + owners.invested_equity.loc[year]
        )

        st_inv = 0.0
        if external_investing <= 0:
            previous_cum_ncb = self.cumulated_ncb.loc[year - 1]
            ncb_previous_modules = owners.ncb_previous_modules.loc[year]
            available = previous_cum_ncb + ncb_previous_modules + total_inflow - cum_ncb
            st_inv = max(0.0, available)

        ncb_disc = total_inflow - st_inv
        year_ncb = owners.ncb_previous_modules.loc[year] + ncb_disc

        self.redemption_st_investment = pd.concat(
            [
                self.redemption_st_investment,
                pd.Series([redemption_from_st], index=pd.Index([year])),
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

    def get_st_returns(self, year: int, forecast: "Forecasting") -> float:
        redemption_from_previous = self.st_investments.loc[year - 1]
        return_from_st = redemption_from_previous * float(
            forecast.return_st_investment.loc[year]
        )
        self.return_from_st_investment = pd.concat(
            [
                self.return_from_st_investment,
                pd.Series([return_from_st], index=pd.Index([year])),
            ]
        )
        return return_from_st

    def pretty_print(self) -> str:
        """Pretty print DiscretionaryTransactions data."""
        lines = ["=" * 60, "DiscretionaryTransactions", "=" * 60]

        lines.append("\nRedemption ST Investment:")
        lines.append(str(self.redemption_st_investment))

        lines.append("\nReturn from ST Investment:")
        lines.append(str(self.return_from_st_investment))

        lines.append("\nTotal Inflow ST Investment:")
        lines.append(str(self.total_inflow_st_investment))

        lines.append("\nST Investments:")
        lines.append(str(self.st_investments))

        lines.append("\nNCB Discretionary Transactions:")
        lines.append(str(self.ncb_discretionary_transactions))

        lines.append("\nYear NCB:")
        lines.append(str(self.year_ncb))

        lines.append("\nCumulated NCB:")
        lines.append(str(self.cumulated_ncb))

        lines.append("=" * 60)

        return "\n".join(lines)
