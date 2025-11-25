from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class AdminSellingExpenses:
    years: pd.Index

    sales_commissions: pd.Series  # Row 96
    overhead_expenses: pd.Series  # Row 97
    payroll_expenses: pd.Series  # Row 98
    advertising_expenses: pd.Series  # Row 99
    total_as_expenses: pd.Series  # Row 100

    @classmethod
    def from_inputs(
        cls,
        input_data,
        policy,
        forecasts,
    ) -> "AdminSellingExpenses":
        years = input_data.years
        T = len(years)

        # -----------------------------
        # Row 96: Sales commissions
        # Excel: =D57 * $E$38
        #  -> total_sales_t * selling_commission_rate (constant)
        # -----------------------------
        # Commission rate entered only once in the policy table (cell E38),
        # so treat it as a constant = policy.selling_commission_pct at year 1.
        commission_rate = float(policy.selling_commission_pct.iloc[1])
        sales_commissions = forecasts.total_sales * commission_rate

        # -----------------------------
        # Row 97: Overhead expenses
        # Excel:
        #   Year 1: =D12           (base overhead)
        #   Later:  =D97 * (1+E48) (grow with nominal overhead increase)
        # -----------------------------
        overhead_vals = [0.0] * T
        overhead_vals[0] = input_data.estimated_overhead_expenses
        for t in range(1, T):
            g_nom_ovh = float(forecasts.nominal_overhead.iloc[t])
            overhead_vals[t] = overhead_vals[t - 1] * (1.0 + g_nom_ovh)
        overhead_expenses = pd.Series(overhead_vals, index=years)

        # -----------------------------
        # Row 98: Payroll expenses
        # Excel:
        #   Year 1: =D13           (base payroll)
        #   Later:  =D98 * (1+E49) (grow with nominal payroll increase)
        # -----------------------------
        payroll_vals = [0.0] * T
        payroll_vals[0] = input_data.admin_and_sales_payroll
        for t in range(1, T):
            g_nom_pay = float(forecasts.nominal_payroll.iloc[t])
            payroll_vals[t] = payroll_vals[t - 1] * (1.0 + g_nom_pay)
        payroll_expenses = pd.Series(payroll_vals, index=years)

        # -----------------------------
        # Row 99: Advertising expenses
        # Excel: =D57 * $E$28
        #  -> total_sales_t * promo_ad_rate (constant)
        # -----------------------------
        promo_rate = float(policy.promo_ad.iloc[1])
        advertising_expenses = forecasts.total_sales * promo_rate

        # -----------------------------
        # Row 100: A&S expenses
        # Excel: =SUM(E96:E99)
        #  -> row-wise sum of the four components
        # -----------------------------
        total_as_expenses = (
            sales_commissions
            + overhead_expenses
            + payroll_expenses
            + advertising_expenses
        )

        return cls(
            years=years,
            sales_commissions=sales_commissions,
            overhead_expenses=overhead_expenses,
            payroll_expenses=payroll_expenses,
            advertising_expenses=advertising_expenses,
            total_as_expenses=total_as_expenses,
        )
