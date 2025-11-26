from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from expenses import AdminSellingExpenses
    from forecasting import Forecasting
    from income_statement import IncomeStatement
    from input import InputData, PolicyTable
    from loans import LoanSchedules
    from trans import Transactions
    from value import DepreciationSchedule, SalesPurchasesSchedule


@dataclass()
class CashBudget:
    """Cash budget tracking operating, investing, and financing activities."""

    years: pd.Index
    inflows_from_sales: pd.Series
    total_operating_inflows: pd.Series
    payments_for_purchases: pd.Series
    admin_selling_expenses: pd.Series
    income_taxes: pd.Series
    total_operating_outflows: pd.Series
    operating_ncb: pd.Series
    investment_in_fixed_assets: pd.Series
    ncb_investment_assets: pd.Series
    ncb_after_capex: pd.Series
    st_loan_inflow: pd.Series
    lt_loan_inflow: pd.Series
    principal_st_loan: pd.Series
    interest_st_loan: pd.Series
    total_st_loan_payment: pd.Series
    principal_lt_loan: pd.Series
    interest_lt_loan: pd.Series
    total_loan_payment: pd.Series
    ncb_financing_activities: pd.Series

    @classmethod
    def initial(
        cls,
        input_data: "InputData",
        policy: "PolicyTable",
        sales_purch_schedule: "SalesPurchasesSchedule",
        admin_selling_expenses: "AdminSellingExpenses",
        dep_and_investment: "DepreciationSchedule",
        forecast: "Forecasting",
        loans: "LoanSchedules",
    ) -> "CashBudget":
        years_full = sales_purch_schedule.years
        y0 = years_full[0]
        years = pd.Index([y0])

        inflow0 = sales_purch_schedule.total_inflows.loc[y0]
        outflow0 = sales_purch_schedule.total_outflows.loc[y0]
        admin0 = admin_selling_expenses.total_as_expenses.loc[y0]

        inflows_from_sales = pd.Series([inflow0], index=years)
        total_operating_inflows = inflows_from_sales.copy()

        payments_for_purchases = pd.Series([outflow0], index=years)
        admin_selling_expenses = pd.Series([admin0], index=years)
        total_operating_outflows = pd.Series([outflow0 + admin0], index=years)
        oper_ncb = inflow0 - (outflow0 + admin0)
        operating_ncb = pd.Series([oper_ncb], index=years)

        investment_in_fixed_assets_val = float(
            dep_and_investment.new_fixed_assets.iloc[years].values[0]
        )
        investment_in_fixed_assets = pd.Series(
            [investment_in_fixed_assets_val], index=years
        )

        ncb_investment_assets = pd.Series(
            [-investment_in_fixed_assets_val], index=years
        )
        ncb_after_capex = oper_ncb + ncb_investment_assets.iloc[0]
        income_taxes = pd.Series([0.0], index=years)

        st_loan_pp = pd.Series([0.0], index=years)
        st_loan_ip = pd.Series([0.0], index=years)
        st_loan_total = pd.Series([0.0], index=years)
        lt_loan_pp = pd.Series([0.0], index=years)
        lt_loan_ip = pd.Series([0.0], index=years)
        total_loan_payment = pd.Series([0.0], index=years)

        st_loan = 0.0
        if oper_ncb - forecast.minimum_cash_required.loc[0] < 0:
            st_loan = -(oper_ncb - forecast.minimum_cash_required.loc[0])

        if st_loan > 0:
            loans.new_loan(
                category="ST",
                start_year=y0,
                amount=st_loan,
                input_data=input_data,
                forecast=forecast,
            )

        lt_loan = 0.0
        if (ncb_after_capex + st_loan - forecast.minimum_cash_required.loc[0]) < 0:
            lt_loan = (
                -(ncb_after_capex + st_loan - forecast.minimum_cash_required.loc[0])
                * policy.debt_financing_pct
            )
        if lt_loan > 0:
            loans.new_loan(
                category="LT",
                start_year=y0,
                amount=lt_loan,
                input_data=input_data,
                forecast=forecast,
            )

        ncb_financing_activities = pd.Series(
            [st_loan + lt_loan - float(total_loan_payment.iloc[0])], index=years
        )

        return cls(
            years=years,
            inflows_from_sales=inflows_from_sales,
            total_operating_inflows=total_operating_inflows,
            payments_for_purchases=payments_for_purchases,
            admin_selling_expenses=admin_selling_expenses,
            income_taxes=income_taxes,
            total_operating_outflows=total_operating_outflows,
            operating_ncb=operating_ncb,
            investment_in_fixed_assets=investment_in_fixed_assets,
            ncb_investment_assets=ncb_investment_assets,
            ncb_after_capex=pd.Series([ncb_after_capex], index=years),
            st_loan_inflow=pd.Series([st_loan], index=years),
            lt_loan_inflow=pd.Series([lt_loan], index=years),
            principal_st_loan=st_loan_pp,
            interest_st_loan=st_loan_ip,
            total_st_loan_payment=st_loan_total,
            principal_lt_loan=lt_loan_pp,
            interest_lt_loan=lt_loan_ip,
            total_loan_payment=total_loan_payment,
            ncb_financing_activities=ncb_financing_activities,
        )

    def add_year(
        self,
        year: int,
        income_statement: "IncomeStatement",
        sales_purchases: "SalesPurchasesSchedule",
        expenses: "AdminSellingExpenses",
        dep_and_investment: "DepreciationSchedule",
        forecast: "Forecasting",
        transactions: "Transactions",
        loans: "LoanSchedules",
        input_data: "InputData",
        policy: "PolicyTable",
    ) -> None:
        self.years = self.years.append(pd.Index([year]))

        admin_expense = expenses.total_as_expenses.loc[year]
        total_inflow = sales_purchases.total_inflows.loc[year]
        payment_for_purchases = sales_purchases.total_outflows.loc[year]
        income_tax = income_statement.income_taxes.loc[year]

        total_outflow = payment_for_purchases + admin_expense + income_tax
        oper_ncb = total_inflow - total_outflow

        invest = dep_and_investment.new_fixed_assets.loc[year]

        ncb_inv = -invest
        ncb_after_capex = ncb_inv + oper_ncb

        loans_summary = loans.schedule_summary(year)
        st_loan_pp = loans_summary.short_term.principal_payments
        st_loan_ip = loans_summary.short_term.interest_payments
        st_loan_total = st_loan_pp + st_loan_ip

        lt_loan_pp = loans_summary.long_term.principal_payments
        lt_loan_ip = loans_summary.long_term.interest_payments

        total_loan_payment = st_loan_total + lt_loan_pp + lt_loan_ip

        previous_ncb = (
            transactions.discretionary.year_ncb.loc[year - 1] if year > 0 else 0.0
        )

        previous_cash = forecast.minimum_cash_required.loc[year - 1]
        st_loan_check = (
            previous_ncb
            + oper_ncb
            - st_loan_total
            - forecast.minimum_cash_required.loc[year]
        )
        st_loan = 0.0
        if st_loan_check < 0:
            st_loan = -st_loan_check

        if st_loan > 0:
            loans.new_loan(
                category="ST",
                start_year=year,
                amount=st_loan,
                input_data=input_data,
                forecast=forecast,
            )

        self.st_loan_inflow = pd.concat(
            [self.st_loan_inflow, pd.Series([st_loan], index=[year])]
        )

        return_from_st_investment = transactions.discretionary.get_st_returns(
            year=year,
            forecast=forecast,
        )
        payments_to_owners = transactions.owner.owner_payments(
            year, income_statement, dep_and_investment, policy
        )

        lt_loan = 0.0
        lt_check = (
            previous_cash
            + ncb_after_capex
            + st_loan
            - total_loan_payment
            - payments_to_owners
            + return_from_st_investment
            - forecast.minimum_cash_required.loc[year]
        )

        if lt_check < 0:
            lt_loan = (-lt_check) * policy.debt_financing_pct

        if lt_loan > 0:
            loans.new_loan(
                category="LT",
                start_year=year,
                amount=lt_loan,
                input_data=input_data,
                forecast=forecast,
            )

        ncb_financing_activities = st_loan + lt_loan - total_loan_payment

        self.inflows_from_sales = pd.concat(
            [
                self.inflows_from_sales,
                pd.Series([total_inflow], index=[year]),
            ]
        )
        self.total_operating_inflows = pd.concat(
            [self.total_operating_inflows, pd.Series([total_inflow], index=[year])]
        )
        self.payments_for_purchases = pd.concat(
            [
                self.payments_for_purchases,
                pd.Series([payment_for_purchases], index=[year]),
            ]
        )
        self.admin_selling_expenses = pd.concat(
            [self.admin_selling_expenses, pd.Series([admin_expense], index=[year])]
        )
        self.income_taxes = pd.concat(
            [self.income_taxes, pd.Series([income_tax], index=[year])]
        )
        self.total_operating_outflows = pd.concat(
            [self.total_operating_outflows, pd.Series([total_outflow], index=[year])]
        )
        self.operating_ncb = pd.concat(
            [self.operating_ncb, pd.Series([oper_ncb], index=[year])]
        )
        self.investment_in_fixed_assets = pd.concat(
            [self.investment_in_fixed_assets, pd.Series([invest], index=[year])]
        )
        self.ncb_investment_assets = pd.concat(
            [self.ncb_investment_assets, pd.Series([ncb_inv], index=[year])]
        )
        self.ncb_after_capex = pd.concat(
            [self.ncb_after_capex, pd.Series([ncb_after_capex], index=[year])]
        )

        self.lt_loan_inflow = pd.concat(
            [self.lt_loan_inflow, pd.Series([lt_loan], index=[year])]
        )
        self.principal_st_loan = pd.concat(
            [self.principal_st_loan, pd.Series([st_loan_pp], index=[year])]
        )
        self.interest_st_loan = pd.concat(
            [self.interest_st_loan, pd.Series([st_loan_ip], index=[year])]
        )
        self.total_st_loan_payment = pd.concat(
            [self.total_st_loan_payment, pd.Series([st_loan_total], index=[year])]
        )
        self.principal_lt_loan = pd.concat(
            [self.principal_lt_loan, pd.Series([lt_loan_pp], index=[year])]
        )
        self.interest_lt_loan = pd.concat(
            [self.interest_lt_loan, pd.Series([lt_loan_ip], index=[year])]
        )
        self.total_loan_payment = pd.concat(
            [self.total_loan_payment, pd.Series([total_loan_payment], index=[year])]
        )
        self.ncb_financing_activities = pd.concat(
            [
                self.ncb_financing_activities,
                pd.Series([ncb_financing_activities], index=[year]),
            ]
        )
