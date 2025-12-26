from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from expenses import AdminSellingExpenses
    from forecasting import Forecasting
    from income_statement import IncomeStatement
    from input import InputData, PolicyTable
    from loans import LoanSchedules
    from trans import Transactions
    from value import DepreciationSchedule, SalesPurchasesSchedule

import tensorflow as tf


@dataclass()
class CashBudget:
    """Cash budget tracking operating, investing, and financing activities."""

    year: tf.Tensor
    inflows_from_sales: tf.Tensor
    total_operating_inflows: tf.Tensor
    payments_for_purchases: tf.Tensor
    admin_selling_expenses: tf.Tensor
    income_taxes: tf.Tensor
    total_operating_outflows: tf.Tensor
    operating_ncb: tf.Tensor
    investment_in_fixed_assets: tf.Tensor
    ncb_investment_assets: tf.Tensor
    ncb_after_capex: tf.Tensor
    st_loan_inflow: tf.Tensor
    lt_loan_inflow: tf.Tensor
    principal_st_loan: tf.Tensor
    interest_st_loan: tf.Tensor
    total_st_loan_payment: tf.Tensor
    principal_lt_loan: tf.Tensor
    interest_lt_loan: tf.Tensor
    total_loan_payment: tf.Tensor
    ncb_financing_activities: tf.Tensor

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
        y0 = input_data.years[0]
        year = tf.constant([y0], dtype=tf.int32)

        inflow0 = sales_purch_schedule.total_inflows.loc[y0]
        outflow0 = sales_purch_schedule.total_outflows.loc[y0]
        admin0 = admin_selling_expenses.total_as_expenses.loc[y0]

        inflows_from_sales = tf.constant([inflow0], dtype=tf.float32)
        total_operating_inflows = inflows_from_sales

        payments_for_purchases = tf.constant([outflow0], dtype=tf.float32)
        admin_selling_expense = tf.constant([admin0], dtype=tf.float32)
        total_operating_outflows = tf.constant([outflow0 + admin0], dtype=tf.float32)
        oper_ncb = inflow0 - (outflow0 + admin0)
        operating_ncb = tf.constant([oper_ncb], dtype=tf.float32)

        investment_in_fixed_assets_val = float(
            dep_and_investment.new_fixed_assets.iloc[y0]
        )
        investment_in_fixed_assets = tf.constant(
            [investment_in_fixed_assets_val], dtype=tf.float32
        )

        ncb_investment_assets = tf.constant(
            [-investment_in_fixed_assets_val], dtype=tf.float32
        )
        ncb_after_capex = oper_ncb + ncb_investment_assets
        income_taxes = tf.constant([0.0], dtype=tf.float32)

        st_loan_pp = tf.constant([0.0], dtype=tf.float32)
        st_loan_ip = tf.constant([0.0], dtype=tf.float32)
        st_loan_total = tf.constant([0.0], dtype=tf.float32)
        lt_loan_pp = tf.constant([0.0], dtype=tf.float32)
        lt_loan_ip = tf.constant([0.0], dtype=tf.float32)
        total_loan_payment = tf.constant([0.0], dtype=tf.float32)

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
        st_loan = tf.constant([st_loan], dtype=tf.float32)

        ncb_financing_activities = tf.subtract(
            tf.add(st_loan, lt_loan), total_loan_payment
        )

        return cls(
            year=year,
            inflows_from_sales=inflows_from_sales,
            total_operating_inflows=total_operating_inflows,
            payments_for_purchases=payments_for_purchases,
            admin_selling_expenses=admin_selling_expense,
            income_taxes=income_taxes,
            total_operating_outflows=total_operating_outflows,
            operating_ncb=operating_ncb,
            investment_in_fixed_assets=investment_in_fixed_assets,
            ncb_investment_assets=ncb_investment_assets,
            ncb_after_capex=ncb_after_capex,
            st_loan_inflow=st_loan,
            lt_loan_inflow=lt_loan,
            principal_st_loan=st_loan_pp,
            interest_st_loan=st_loan_ip,
            total_st_loan_payment=st_loan_total,
            principal_lt_loan=lt_loan_pp,
            interest_lt_loan=lt_loan_ip,
            total_loan_payment=total_loan_payment,
            ncb_financing_activities=ncb_financing_activities,
        )

    @classmethod
    def add_year(
        cls,
        year: int,
        income_statement: "IncomeStatement",
        previous_is: "IncomeStatement",
        sales_purchases: "SalesPurchasesSchedule",
        expenses: "AdminSellingExpenses",
        dep_and_investment: "DepreciationSchedule",
        forecast: "Forecasting",
        transactions: "Transactions",
        loans: "LoanSchedules",
        input_data: "InputData",
        policy: "PolicyTable",
        previous_cash: CashBudget,
    ) -> CashBudget:
        admin_expense = expenses.total_as_expenses.loc[year]
        total_inflow = sales_purchases.total_inflows.loc[year]
        payment_for_purchases = sales_purchases.total_outflows.loc[year]

        income_tax = income_statement.income_taxes

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

        return_from_st_investment = transactions.discretionary.get_st_returns(
            year=year,
            forecast=forecast,
        )
        payments_to_owners = transactions.owner.owner_payments(
            year, previous_is, dep_and_investment, policy
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

        return cls(
            year=tf.constant([year], dtype=tf.int32),
            inflows_from_sales=tf.constant([total_inflow], dtype=tf.float32),
            total_operating_inflows=tf.constant(
                [previous_cash + total_inflow], dtype=tf.float32
            ),
            payments_for_purchases=tf.constant(
                [payment_for_purchases], dtype=tf.float32
            ),
            admin_selling_expenses=tf.constant([admin_expense], dtype=tf.float32),
            income_taxes=tf.constant([income_tax], dtype=tf.float32),
            total_operating_outflows=tf.constant([total_outflow], dtype=tf.float32),
            operating_ncb=tf.constant([oper_ncb], dtype=tf.float32),
            investment_in_fixed_assets=tf.constant([invest], dtype=tf.float32),
            ncb_investment_assets=tf.constant([ncb_inv], dtype=tf.float32),
            ncb_after_capex=tf.constant([ncb_after_capex], dtype=tf.float32),
            st_loan_inflow=tf.constant([st_loan], dtype=tf.float32),
            lt_loan_inflow=tf.expand_dims(tf.cast(lt_loan, tf.float32), 0),
            principal_st_loan=tf.constant([st_loan_pp], dtype=tf.float32),
            interest_st_loan=tf.constant([st_loan_ip], dtype=tf.float32),
            total_st_loan_payment=tf.constant([st_loan_total], dtype=tf.float32),
            principal_lt_loan=tf.constant([lt_loan_pp], dtype=tf.float32),
            interest_lt_loan=tf.constant([lt_loan_ip], dtype=tf.float32),
            total_loan_payment=tf.constant([total_loan_payment], dtype=tf.float32),
            ncb_financing_activities=tf.expand_dims(
                tf.cast(st_loan + lt_loan - total_loan_payment, tf.float32), 0
            ),
        )
