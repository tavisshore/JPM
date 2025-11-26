from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from forecasting import Forecasting
    from input import InputData, PolicyTable


@dataclass(frozen=True)
class AdminSellingExpenses:
    """Admin and selling expense schedule derived from policy and forecasts."""

    years: pd.Index

    sales_commissions: pd.Series
    overhead_expenses: pd.Series
    payroll_expenses: pd.Series
    advertising_expenses: pd.Series
    total_as_expenses: pd.Series

    @classmethod
    def from_inputs(
        cls,
        input_data: "InputData",
        policy: "PolicyTable",
        forecasts: "Forecasting",
    ) -> "AdminSellingExpenses":
        years = input_data.years
        T = len(years)

        commission_rate = float(policy.selling_commission_pct.iloc[1])
        sales_commissions = forecasts.total_sales * commission_rate

        overhead_vals = [0.0] * T
        overhead_vals[0] = input_data.estimated_overhead_expenses
        for t in range(1, T):
            g_nom_ovh = float(forecasts.nominal_overhead.iloc[t])
            overhead_vals[t] = overhead_vals[t - 1] * (1.0 + g_nom_ovh)
        overhead_expenses = pd.Series(overhead_vals, index=years)

        payroll_vals = [0.0] * T
        payroll_vals[0] = input_data.admin_and_sales_payroll
        for t in range(1, T):
            g_nom_pay = float(forecasts.nominal_payroll.iloc[t])
            payroll_vals[t] = payroll_vals[t - 1] * (1.0 + g_nom_pay)
        payroll_expenses = pd.Series(payroll_vals, index=years)

        promo_rate = float(policy.promo_ad.iloc[1])
        advertising_expenses = forecasts.total_sales * promo_rate

        total_as_expenses = (
            sales_commissions
            + overhead_expenses
            + payroll_expenses
            + advertising_expenses
        )
        total_as_expenses.iloc[0] = 0.0

        return cls(
            years=years,
            sales_commissions=sales_commissions,
            overhead_expenses=overhead_expenses,
            payroll_expenses=payroll_expenses,
            advertising_expenses=advertising_expenses,
            total_as_expenses=total_as_expenses,
        )
