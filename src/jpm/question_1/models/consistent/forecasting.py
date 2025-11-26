from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from input import InputData, MarketResearchInput, PolicyTable


@dataclass(frozen=True)
class Forecasting:
    """Forecasts nominal rates, sales, and cash requirements across years."""

    years: pd.Index

    nominal_selling: pd.Series
    nominal_purchasing: pd.Series
    nominal_overhead: pd.Series
    nominal_payroll: pd.Series
    minimum_cash_required: pd.Series

    increase_factor_volume: pd.Series
    sales_units: pd.Series
    selling_price: pd.Series
    total_sales: pd.Series

    risk_free_rate: pd.Series
    return_st_investment: pd.Series
    cost_of_debt: pd.Series

    @classmethod
    def from_inputs(
        cls,
        input_data: "InputData",
        policy: "PolicyTable",
        market_research: "MarketResearchInput",
    ) -> "Forecasting":
        years = input_data.years

        def nominal_rate(infl: pd.Series, real: pd.Series) -> pd.Series:

            rate = (1.0 + infl) * (1.0 + real) - 1.0
            rate.iloc[0] = 0.0
            return rate

        nominal_selling = nominal_rate(
            input_data.inflation_rate, input_data.real_increase_selling_price
        )
        nominal_purchasing = nominal_rate(
            input_data.inflation_rate, input_data.real_increase_purchase_price
        )
        nominal_overhead = nominal_rate(
            input_data.inflation_rate, input_data.real_increase_overheads
        )
        nominal_payroll = nominal_rate(
            input_data.inflation_rate, input_data.real_increase_payroll
        )

        increase_factor_volume = 1.0 + input_data.increase_sales_volume
        increase_factor_volume.iloc[0] = 0.0

        P0 = market_research.selling_price
        b = market_research.elasticity_b
        b0 = market_research.elasticity_coef
        base_units = b0 * (P0**b)

        units = [0.0, base_units]
        for t in range(2, len(years)):
            if t != 1:
                units.append(units[t - 1] * increase_factor_volume.iloc[t])
        sales_units = pd.Series(units, index=years)

        price = [P0]
        for t in range(1, len(years)):
            price.append(price[t - 1] * (1.0 + nominal_selling.iloc[t]))
        selling_price = pd.Series(price, index=years)

        total_sales = selling_price * sales_units
        total_sales.iloc[0] = 0.0

        minimum_cash_required = policy.cash_pct_of_sales * total_sales
        minimum_cash_required.index = years
        minimum_cash_required.iloc[0] = policy.minimum_initial_cash

        rf_vals = []
        for i, y in enumerate(years):
            if i == 0:
                rf_vals.append(0.0)
            else:
                π = input_data.inflation_rate.loc[y]
                r_real = input_data.real_interest_rate.loc[y]
                rf_vals.append((1.0 + π) * (1.0 + r_real) - 1.0)

        risk_free_rate = pd.Series(rf_vals, index=years)

        return_st_investment = risk_free_rate + input_data.risk_premium_return_st_inv
        return_st_investment.iloc[0] = 0.0

        cost_of_debt = risk_free_rate + input_data.risk_premium_debt_cost
        cost_of_debt.iloc[0] = 0.0

        return cls(
            years=years,
            nominal_selling=nominal_selling,
            nominal_purchasing=nominal_purchasing,
            nominal_overhead=nominal_overhead,
            nominal_payroll=nominal_payroll,
            minimum_cash_required=minimum_cash_required,
            increase_factor_volume=increase_factor_volume,
            sales_units=sales_units,
            selling_price=selling_price,
            total_sales=total_sales,
            risk_free_rate=risk_free_rate,
            return_st_investment=return_st_investment,
            cost_of_debt=cost_of_debt,
        )
