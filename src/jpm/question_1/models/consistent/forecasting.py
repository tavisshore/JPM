from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Forecasting:
    years: pd.Index

    # -------- Nominal increases --------
    nominal_selling: pd.Series
    nominal_purchasing: pd.Series
    nominal_overhead: pd.Series
    nominal_payroll: pd.Series
    minimum_cash_required: pd.Series

    # -------- Forecast sales (Table 3) --------
    increase_factor_volume: pd.Series
    sales_units: pd.Series
    selling_price: pd.Series
    total_sales: pd.Series

    # ------- Rates forecast (Table 4) --------
    risk_free_rate: pd.Series  # Rf
    return_st_investment: pd.Series  # Rf + premium_ST
    cost_of_debt: pd.Series  # Rf + premium_debt

    @classmethod
    def from_inputs(
        cls,
        input_data,
        policy,
        market_research,
    ):
        years = input_data.years

        # ============================================================
        # 1) NOMINAL INCREASES (Table 2 logic)
        # ============================================================
        def nominal_rate(infl: pd.Series, real: pd.Series) -> pd.Series:
            # year 0 = 0
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

        # ============================================================
        # 2) FORECAST SALES (Table 3 logic)
        # ============================================================

        # 2.1 Increase factor volume
        increase_factor_volume = 1.0 + input_data.increase_sales_volume

        # 2.2 Base units from elasticity: Q0 = b0 * P0^b
        P0 = market_research.selling_price
        b = market_research.elasticity_b
        b0 = market_research.elasticity_coef
        base_units = b0 * (P0**b)

        # 2.3 Units each year
        units = [base_units]
        for t in range(1, len(years)):
            units.append(units[t - 1] * increase_factor_volume.iloc[t])
        sales_units = pd.Series(units, index=years)

        # 2.4 Selling price
        price = [P0]
        for t in range(1, len(years)):
            price.append(price[t - 1] * (1.0 + nominal_selling.iloc[t]))
        selling_price = pd.Series(price, index=years)

        # 2.5 Total sales
        total_sales = selling_price * sales_units

        # minimum cash requirement
        minimum_cash_required = policy.minimum_initial_cash * total_sales
        minimum_cash_required.index = years

        # ============================================================
        # 2) RatesForecast (Table 4 logic)
        # ============================================================

        # Risk-free rate: Rf_t = (1 + π_t) * (1 + r_real_t) - 1
        rf_vals = []
        for i, y in enumerate(years):
            if i == 0:
                rf_vals.append(0.0)  # year 0 left as 0 in the spreadsheet
            else:
                π = input_data.inflation_rate.loc[y]
                r_real = input_data.real_interest_rate.loc[y]
                rf_vals.append((1.0 + π) * (1.0 + r_real) - 1.0)

        risk_free_rate = pd.Series(rf_vals, index=years)

        # Return on short-term investment: Rf + risk premium of ST return
        return_st_investment = risk_free_rate + input_data.risk_premium_return_st_inv

        # Cost of debt: Rf + risk premium in cost of debt
        cost_of_debt = risk_free_rate + input_data.risk_premium_debt_cost

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
