from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from expenses import AdminSellingExpenses
    from forecasting import Forecasting
    from input import InputData, PolicyTable
    from loans import LoanSchedules
    from value import DepreciationSchedule, InventorySchedule


@dataclass
class IncomeStatement:
    """Income statement calculations over the forecast horizon."""

    years: pd.Index

    sales_revenues: pd.Series
    cogs: pd.Series
    gross_income: pd.Series
    admin_selling_expenses: pd.Series
    depreciation: pd.Series
    ebit: pd.Series
    interest_payments: pd.Series
    return_from_st_investment: pd.Series
    ebt: pd.Series
    income_taxes: pd.Series
    net_income: pd.Series
    next_year_dividends: pd.Series
    cre: pd.Series

    @classmethod
    def init_year(
        cls,
        input_data: "InputData",
        policy: "PolicyTable",
        forecast_sales: "Forecasting",
        inv_fifo: "InventorySchedule",
        admin_selling: "AdminSellingExpenses",
        depr_sched: "DepreciationSchedule",
        loan_schedules: "LoanSchedules",
    ) -> "IncomeStatement":
        full_years = input_data.years
        y0 = full_years[0]
        years = pd.Index([y0])

        sales_revenues = pd.Series([forecast_sales.total_sales.loc[y0]], index=years)
        cogs = pd.Series([inv_fifo.cogs.loc[y0]], index=years)
        gross_income = sales_revenues - cogs
        admin_selling_expenses = pd.Series(
            [admin_selling.total_as_expenses.loc[y0]], index=years
        )
        depreciation = pd.Series([depr_sched.annual_depreciation.loc[y0]], index=years)
        ebit = gross_income - admin_selling_expenses - depreciation
        dp = loan_schedules.debt_payments(y0)
        interest_payments = pd.Series([dp.st_interest + dp.lt_interest], index=years)
        return_from_st_investment = pd.Series([0.0], index=years)
        ebt = ebit + return_from_st_investment - interest_payments
        tax_rate = input_data.corporate_tax_rate
        income_taxes = ebt.clip(lower=0.0) * tax_rate
        net_income = ebt - income_taxes
        payout_ratio = float(policy.payout_ratio.iloc[1])
        next_year_dividends = net_income * payout_ratio
        cre = pd.Series([net_income.iloc[0] - next_year_dividends.iloc[0]], index=years)

        return cls(
            years=years,
            sales_revenues=sales_revenues,
            cogs=cogs,
            gross_income=gross_income,
            admin_selling_expenses=admin_selling_expenses,
            depreciation=depreciation,
            ebit=ebit,
            interest_payments=interest_payments,
            return_from_st_investment=return_from_st_investment,
            ebt=ebt,
            income_taxes=income_taxes,
            net_income=net_income,
            next_year_dividends=next_year_dividends,
            cre=cre,
        )

    def add_year(
        self,
        year,
        input_data: "InputData",
        policy: "PolicyTable",
        forecast_sales: "Forecasting",
        inv_fifo: "InventorySchedule",
        admin_selling: "AdminSellingExpenses",
        depr_sched: "DepreciationSchedule",
        loan_schedules: "LoanSchedules",
    ) -> None:

        self.years = self.years.append(pd.Index([year]))

        sales_revenues_t = forecast_sales.total_sales.loc[year]
        cogs_t = inv_fifo.cogs.loc[year]
        gross_income_t = sales_revenues_t - cogs_t
        admin_selling_expenses_t = admin_selling.total_as_expenses.loc[year]
        depreciation_t = depr_sched.annual_depreciation.loc[year]
        ebit_t = gross_income_t - admin_selling_expenses_t - depreciation_t
        dp = loan_schedules.debt_payments(year)
        interest_payments_t = dp.st_interest + dp.lt_interest
        return_from_st_investment_t = 0.0
        ebt_t = ebit_t + return_from_st_investment_t - interest_payments_t
        tax_rate = input_data.corporate_tax_rate
        income_taxes_t = max(0.0, ebt_t) * tax_rate
        net_income_t = ebt_t - income_taxes_t
        payout_ratio = float(policy.payout_ratio.iloc[1])
        next_year_dividends_t = net_income_t * payout_ratio

        cre_prev = float(self.cre.iloc[year - 1])
        net_income_prev = float(self.net_income.iloc[year - 1])
        dividends_prev = float(self.next_year_dividends.iloc[year - 1])
        cre_t = cre_prev + net_income_prev - dividends_prev

        self.sales_revenues = pd.concat(
            [self.sales_revenues, pd.Series([sales_revenues_t], index=[year])]
        )
        self.cogs = pd.concat([self.cogs, pd.Series([cogs_t], index=[year])])
        self.gross_income = pd.concat(
            [self.gross_income, pd.Series([gross_income_t], index=[year])]
        )
        self.admin_selling_expenses = pd.concat(
            [
                self.admin_selling_expenses,
                pd.Series([admin_selling_expenses_t], index=[year]),
            ]
        )
        self.depreciation = pd.concat(
            [self.depreciation, pd.Series([depreciation_t], index=[year])]
        )
        self.ebit = pd.concat([self.ebit, pd.Series([ebit_t], index=[year])])
        self.interest_payments = pd.concat(
            [self.interest_payments, pd.Series([interest_payments_t], index=[year])]
        )
        self.return_from_st_investment = pd.concat(
            [
                self.return_from_st_investment,
                pd.Series([return_from_st_investment_t], index=[year]),
            ]
        )
        self.ebt = pd.concat([self.ebt, pd.Series([ebt_t], index=[year])])
        self.income_taxes = pd.concat(
            [self.income_taxes, pd.Series([income_taxes_t], index=[year])]
        )
        self.net_income = pd.concat(
            [self.net_income, pd.Series([net_income_t], index=[year])]
        )
        self.next_year_dividends = pd.concat(
            [self.next_year_dividends, pd.Series([next_year_dividends_t], index=[year])]
        )
        self.cre = pd.concat([self.cre, pd.Series([cre_t], index=[year])])
