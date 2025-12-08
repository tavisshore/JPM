from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from expenses import AdminSellingExpenses
    from forecasting import Forecasting
    from input import InputData, PolicyTable
    from loans import LoanSchedules
    from value import DepreciationSchedule, InventorySchedule

import tensorflow as tf


@dataclass
class IncomeStatement:
    """Income statement calculations over the forecast horizon."""

    years: tf.Tensor

    sales_revenues: tf.Tensor
    cogs: tf.Tensor
    gross_income: tf.Tensor
    admin_selling_expenses: tf.Tensor
    depreciation: tf.Tensor
    ebit: tf.Tensor
    interest_payments: tf.Tensor
    return_from_st_investment: tf.Tensor
    ebt: tf.Tensor
    income_taxes: tf.Tensor
    net_income: tf.Tensor
    next_year_dividends: tf.Tensor
    cre: tf.Tensor

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
        y0 = input_data.years[0]
        year = tf.constant([y0], dtype=tf.int32)

        sales_revenues = tf.constant(
            forecast_sales.total_sales.loc[y0], dtype=tf.float32
        )
        cogs = tf.constant([inv_fifo.cogs.loc[y0]], dtype=tf.float32)
        gross_income = tf.subtract(sales_revenues, cogs)
        admin_selling_expenses = tf.constant(
            [admin_selling.total_as_expenses.loc[y0]], dtype=tf.float32
        )
        depreciation = tf.constant(
            [depr_sched.annual_depreciation.loc[y0]], dtype=tf.float32
        )
        ebit = gross_income - admin_selling_expenses - depreciation
        dp = loan_schedules.debt_payments(y0)
        interest_payments = tf.constant(
            [dp.st_interest + dp.lt_interest], dtype=tf.float32
        )
        return_from_st_investment = tf.constant([0.0], dtype=tf.float32)
        ebt = ebit + return_from_st_investment - interest_payments
        tax_rate = input_data.corporate_tax_rate

        # income_taxes = ebt.clip(lower=0.0) * tax_rate
        income_taxes = tf.clip_by_value(
            ebt, clip_value_min=0.0, clip_value_max=tf.float32.max
        ) * tf.cast(tax_rate, ebt.dtype)

        net_income = ebt - income_taxes
        payout_ratio = float(policy.payout_ratio.iloc[1])
        next_year_dividends = net_income * payout_ratio
        cre = net_income - next_year_dividends

        return cls(
            years=year,
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

    @classmethod
    def add_year(
        cls,
        year,
        input_data: "InputData",
        policy: "PolicyTable",
        forecast_sales: "Forecasting",
        inv_fifo: "InventorySchedule",
        admin_selling: "AdminSellingExpenses",
        depr_sched: "DepreciationSchedule",
        loan_schedules: "LoanSchedules",
        prev_is: IncomeStatement,
    ) -> IncomeStatement:
        yr = tf.constant([year], dtype=tf.int32)

        sales_revenues = forecast_sales.total_sales.loc[year]
        cogs = inv_fifo.cogs.loc[year]
        gross_income = sales_revenues - cogs
        admin_selling_expenses = admin_selling.total_as_expenses.loc[year]
        depreciation = depr_sched.annual_depreciation.loc[year]
        ebit = gross_income - admin_selling_expenses - depreciation
        dp = loan_schedules.debt_payments(year)
        interest_payments = dp.st_interest + dp.lt_interest
        return_from_st_investment = 0.0
        ebt = ebit + return_from_st_investment - interest_payments
        tax_rate = input_data.corporate_tax_rate
        income_taxes = max(0.0, ebt) * tax_rate
        net_income = ebt - income_taxes
        payout_ratio = float(policy.payout_ratio.iloc[1])
        next_year_dividends = net_income * payout_ratio

        cre_prev = prev_is.cre
        net_income_prev = prev_is.net_income
        dividends_prev = prev_is.next_year_dividends
        cre = tf.subtract(tf.add(cre_prev, net_income_prev), dividends_prev)

        return cls(
            years=yr,
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
