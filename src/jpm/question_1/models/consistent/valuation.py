from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .input import InputData


@dataclass()
class ValuationInputs:
    real_kk: pd.Series
    inflation_rate: pd.Series
    nominal_kk: pd.Series

    tax_rate: float
    real_growth_rate: float
    real_interest_rate: float
    risk_premium_debt_cost: float
    kd_real_cost_debt: float
    ku_perpetual: float
    wacc: float

    @classmethod
    def from_inputs(
        cls,
        input_data: InputData,
    ) -> ValuationInputs:
        """Create ValuationInputs from core InputData."""

        real_kk = (1 + input_data.observed_kk) / (1 + input_data.inflation_rate) - 1
        real_kk.iloc[1:] = real_kk.iat[0]
        inflation_rate = input_data.inflation_rate

        nominal_kk = (1 + real_kk) * (1 + inflation_rate) - 1

        tax_rate = input_data.corporate_tax_rate
        real_growth = input_data.real_growth_rate
        real_interest_rate = input_data.real_interest_rate
        risk_premium_debt_cost = input_data.risk_premium_debt_cost

        kd_real_cost_debt = real_interest_rate + risk_premium_debt_cost
        ku_perpetual = real_kk.iat[0]
        wacc = (
            ku_perpetual - tax_rate * input_data.perpetual_leverage * kd_real_cost_debt
        )

        return cls(
            real_kk=real_kk,
            inflation_rate=inflation_rate,
            nominal_kk=nominal_kk,
            tax_rate=tax_rate,
            real_growth_rate=real_growth,
            real_interest_rate=real_interest_rate,
            risk_premium_debt_cost=risk_premium_debt_cost,
            kd_real_cost_debt=kd_real_cost_debt,
            ku_perpetual=ku_perpetual,
            wacc=wacc,
        )

    def tv_and_liquidate(
        self,
        balance_sheet,
        income_statement,
        cash_flow,
        year: int = -1,
    ) -> float:
        noplat = income_statement.ebit.iloc[year] * (1 - self.tax_rate)
        tv = noplat / self.wacc
        # Cash
        cash = balance_sheet.cash_cb.iloc[year]
        ar_discounted_wacc = balance_sheet.ar_it.iloc[year] / (1 + self.wacc)
        market_securities = balance_sheet.st_investments_cb.iloc[year]
        ap_discounted_wacc = -balance_sheet.ap_it.iloc[year] / (1 + self.wacc)
        liquidation_assets = (
            cash + ar_discounted_wacc + market_securities + ap_discounted_wacc
        )
        adjusted_tv = tv + liquidation_assets

        # calculating value
        ccf = cash_flow.ccf
        years_ext = balance_sheet.years.append(pd.Index([balance_sheet.years[-1] + 1]))

        # reindex, filling missing rows
        ccf_ext = ccf.reindex(years_ext, fill_value=0.0)
        nominal_ext = self.nominal_kk.reindex(years_ext, method="ffill")
        ccf_ext = ccf_ext.shift(1)

        v = pd.Series(dtype=float, index=years_ext)
        v.iloc[-1] = adjusted_tv  # terminal value at the extra year

        # work backwards from second-last element down to 0
        for i in range(len(years_ext) - 2, -1, -1):
            next_i = i + 1
            v.iloc[i] = (v.iloc[next_i] + ccf_ext.iloc[next_i]) / (
                1.0 + nominal_ext.iloc[next_i]
            )

        debt = balance_sheet.short_term_debt_cb + balance_sheet.long_term_debt_cb
        ic = (
            balance_sheet.total_assets
            - balance_sheet.current_liabilities
            + balance_sheet.short_term_debt_cb
        )
        debt = debt.reindex(years_ext)
        debt = debt.shift(1)
        # equity = v - debt

        ic = ic.reindex(years_ext)
        ic = ic.shift(1)

        # npv_firm = v - ic
        # npv_equity = equity - balance_sheet.equity_investment_cb

        return adjusted_tv
