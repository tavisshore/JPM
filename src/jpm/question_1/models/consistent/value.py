from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from forecasting import Forecasting
    from input import InputData, PolicyTable


def _investment_to_keep_constant(
    t: int, initial_nfa: float, annual_dep: pd.Series
) -> float:
    return initial_nfa if t == 0 else annual_dep[t]


def _investment_for_growth(t: int, T: int, net_fa, increase_sales_volume) -> float:
    if t == 0:
        return 0.0
    if t < T - 1:
        return net_fa[t - 1] * increase_sales_volume.iloc[t + 1]
    return 0.0


def _annual_depreciation_for_year(t: int, life: int, new_fa: pd.Series) -> float:
    dep_t = 0.0
    for tau, value in enumerate(new_fa):
        if (t >= tau + 1) and (t <= tau + life):
            dep_t += value / life
    return dep_t


def _compute_depreciation_schedule(
    years: pd.Index, life: int, initial_nfa: float, increase_sales_volume: pd.Series
) -> pd.DataFrame:
    T = len(years)
    beginning_nfa = [0.0] * T
    new_fa = [0.0] * T
    inv_const = [0.0] * T
    inv_growth = [0.0] * T
    annual_dep = [0.0] * T
    net_fa = [0.0] * T

    for t in range(T):
        beginning_nfa[t] = 0.0 if t == 0 else net_fa[t - 1]
        annual_dep[t] = _annual_depreciation_for_year(t, life, new_fa)
        inv_const[t] = _investment_to_keep_constant(t, initial_nfa, annual_dep)
        inv_growth[t] = _investment_for_growth(t, T, net_fa, increase_sales_volume)
        new_fa[t] = inv_const[t] + inv_growth[t]
        net_fa[t] = beginning_nfa[t] + new_fa[t] - annual_dep[t]

    return {
        "beginning_nfa": pd.Series(beginning_nfa, index=years),
        "new_fixed_assets": pd.Series(new_fa, index=years),
        "investment_keep_constant": pd.Series(inv_const, index=years),
        "investment_for_growth": pd.Series(inv_growth, index=years),
        "annual_depreciation": pd.Series(annual_dep, index=years),
        "net_fixed_assets": pd.Series(net_fa, index=years),
    }


def _cohort_depreciation(
    invest_year: int, life: int, years: pd.Index, new_fa: pd.Series
) -> pd.Series:
    vals = []
    T = len(years)
    for t in range(T):
        if (t >= invest_year + 1) and (t <= invest_year + life):
            vals.append(new_fa.iloc[invest_year] / life)
        else:
            vals.append(0.0)
    return pd.Series(vals, index=years)


@dataclass(frozen=True)
class DepreciationSchedule:
    """Fixed asset depreciation schedule and related investments."""

    years: pd.Index

    beginning_nfa: pd.Series

    dep_invest_year0: pd.Series
    dep_invest_year1: pd.Series
    dep_invest_year2: pd.Series
    dep_invest_year3: pd.Series

    annual_depreciation: pd.Series
    cumulated_depreciation: pd.Series

    investment_keep_constant: pd.Series
    investment_for_growth: pd.Series

    new_fixed_assets: pd.Series
    net_fixed_assets: pd.Series

    @classmethod
    def from_inputs(cls, input_data: "InputData") -> "DepreciationSchedule":
        years = input_data.years
        life = int(round(input_data.lineal_depreciation.iloc[0]))
        initial_nfa = float(input_data.net_fixed_assets)

        depreciation = _compute_depreciation_schedule(
            years, life, initial_nfa, input_data.increase_sales_volume
        )
        annual_dep = depreciation["annual_depreciation"]
        cum_dep = annual_dep.cumsum()

        dep_y0 = _cohort_depreciation(0, life, years, depreciation["new_fixed_assets"])
        dep_y1 = _cohort_depreciation(1, life, years, depreciation["new_fixed_assets"])
        dep_y2 = _cohort_depreciation(2, life, years, depreciation["new_fixed_assets"])
        dep_y3 = _cohort_depreciation(3, life, years, depreciation["new_fixed_assets"])

        return cls(
            years=years,
            beginning_nfa=depreciation["beginning_nfa"],
            dep_invest_year0=dep_y0,
            dep_invest_year1=dep_y1,
            dep_invest_year2=dep_y2,
            dep_invest_year3=dep_y3,
            annual_depreciation=annual_dep,
            cumulated_depreciation=cum_dep,
            investment_keep_constant=depreciation["investment_keep_constant"],
            investment_for_growth=depreciation["investment_for_growth"],
            new_fixed_assets=depreciation["new_fixed_assets"],
            net_fixed_assets=depreciation["net_fixed_assets"],
        )


@dataclass(frozen=True)
class InventorySchedule:
    """FIFO inventory calculations and valuation across years."""

    years: pd.Index

    units_sold: pd.Series
    final_inventory: pd.Series
    initial_inventory: pd.Series
    purchases: pd.Series

    unit_cost: pd.Series
    initial_inventory_value: pd.Series
    purchases_value: pd.Series
    final_inventory_value: pd.Series
    cogs: pd.Series

    @classmethod
    def from_inputs(
        cls,
        input_data: "InputData",
        policy: "PolicyTable",
        forecast: "Forecasting",
    ) -> "InventorySchedule":
        years = input_data.years
        T = len(years)

        units_sold = forecast.sales_units.copy()

        final_inv_vals = [0.0] * T
        for t, _y in enumerate(years):
            if t == 0:
                final_inv_vals[t] = input_data.initial_inventory
            else:
                final_inv_vals[t] = units_sold.iloc[t] * policy.inventory_pct.iloc[t]
        final_inventory = pd.Series(final_inv_vals, index=years)

        init_inv_vals = [0.0] * T
        for t in range(1, T):
            init_inv_vals[t] = final_inventory.iloc[t - 1]
        initial_inventory = pd.Series(init_inv_vals, index=years)

        purchases = units_sold + final_inventory - initial_inventory

        unit_cost_vals = [0.0] * T
        unit_cost_vals[0] = input_data.initial_purchase_price
        for t in range(1, T):
            g_nom = forecast.nominal_purchasing.iloc[t]
            unit_cost_vals[t] = unit_cost_vals[t - 1] * (1.0 + g_nom)
        unit_cost = pd.Series(unit_cost_vals, index=years)

        purchases_value = purchases * unit_cost

        final_inv_val = final_inventory * unit_cost

        init_inv_val = [0.0] * T
        for t in range(1, T):
            init_inv_val[t] = final_inv_val.iloc[t - 1]
        initial_inventory_value = pd.Series(init_inv_val, index=years)

        cogs = initial_inventory_value + purchases_value - final_inv_val

        return cls(
            years=years,
            units_sold=units_sold,
            final_inventory=final_inventory,
            initial_inventory=initial_inventory,
            purchases=purchases,
            unit_cost=unit_cost,
            initial_inventory_value=initial_inventory_value,
            purchases_value=purchases_value,
            final_inventory_value=final_inv_val,
            cogs=cogs,
        )


@dataclass(frozen=True)
class SalesPurchasesSchedule:
    """Sales and purchases schedule including credit and cash flows."""

    years: pd.Index

    total_sales_revenues: pd.Series

    inflow_from_current_year: pd.Series

    credit_sales: pd.Series

    advance_from_customers: pd.Series

    total_purchases: pd.Series

    purchases_paid_same_year: pd.Series

    purchases_on_credit: pd.Series

    advance_to_suppliers: pd.Series

    sales_revenues_current_year: pd.Series

    accounts_receivable_flow: pd.Series

    advance_payments_from_customers: pd.Series

    total_inflows: pd.Series

    purchases_paid_current_year: pd.Series

    payment_accounts_payable: pd.Series

    advance_payment_to_suppliers: pd.Series

    total_outflows: pd.Series

    @classmethod
    def from_inputs(
        cls,
        policy: "PolicyTable",
        forecast: "Forecasting",
        inv: "InventorySchedule",
    ) -> "SalesPurchasesSchedule":
        years = forecast.years

        ar_rate = float(policy.ar_pct.iloc[1])
        adv_cust_rate = float(policy.adv_from_cust_pct.iloc[1])
        ap_rate = float(policy.ap_pct.iloc[1])
        adv_sup_rate = float(policy.adv_to_suppliers_pct.iloc[1])

        total_sales = forecast.total_sales.copy()

        inflow_current = total_sales * (1.0 - ar_rate - adv_cust_rate)

        credit_sales = total_sales * ar_rate

        payments_in_advance = total_sales * adv_cust_rate

        total_purchases = inv.purchases_value.copy()

        purchases_paid_same_year = total_purchases * (1.0 - ap_rate - adv_sup_rate)
        purchases_paid_same_year.iloc[0] = total_purchases.iloc[0]

        purchases_on_credit = total_purchases * ap_rate
        year0_credit = total_purchases.iloc[0] - purchases_paid_same_year.iloc[0]
        purchases_on_credit.iloc[0] = year0_credit

        adv_to_suppliers = total_purchases * adv_sup_rate

        adv_to_suppliers = adv_to_suppliers.shift(-1).fillna(0.0)

        sales_rev_current = inflow_current.copy()

        ar_vals = [0.0] * len(years)
        for t in range(1, len(years)):
            ar_vals[t] = credit_sales.iloc[t - 1]
        accounts_receivable_flow = pd.Series(ar_vals, index=years)

        advance_payments_from_customers = payments_in_advance.copy()

        advance_payments_from_customers = advance_payments_from_customers.shift(
            -1
        ).fillna(0.0)

        total_inflows = (
            accounts_receivable_flow
            + sales_rev_current
            + advance_payments_from_customers
        )

        purchases_paid_current_year = purchases_paid_same_year.copy()

        ap_vals = [0.0] * len(years)
        for t in range(1, len(years)):
            ap_vals[t] = purchases_on_credit.iloc[t - 1]
        payment_accounts_payable = pd.Series(ap_vals, index=years)

        advance_payment_to_suppliers = adv_to_suppliers.copy()

        total_outflows = (
            purchases_paid_current_year
            + payment_accounts_payable
            + advance_payment_to_suppliers
        )

        return cls(
            years=years,
            total_sales_revenues=total_sales,
            inflow_from_current_year=inflow_current,
            credit_sales=credit_sales,
            advance_from_customers=payments_in_advance,
            total_purchases=total_purchases,
            purchases_paid_same_year=purchases_paid_same_year,
            purchases_on_credit=purchases_on_credit,
            advance_to_suppliers=adv_to_suppliers,
            sales_revenues_current_year=sales_rev_current,
            accounts_receivable_flow=accounts_receivable_flow,
            advance_payments_from_customers=advance_payments_from_customers,
            total_inflows=total_inflows,
            purchases_paid_current_year=purchases_paid_current_year,
            payment_accounts_payable=payment_accounts_payable,
            advance_payment_to_suppliers=advance_payment_to_suppliers,
            total_outflows=total_outflows,
        )
