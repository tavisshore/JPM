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
        year: int,
        input_data: InputData,
    ) -> ValuationInputs:
        """Create ValuationInputs from core InputData."""

        inflation_rate = input_data.inflation_rate

        real_kk = (1 + input_data.observed_kk) / (1 + inflation_rate) - 1
        real_kk.iloc[1:] = real_kk.iat[0]

        nominal_kk = (1 + real_kk) * (1 + inflation_rate) - 1

        tax_rate = input_data.corporate_tax_rate
        real_growth = input_data.real_growth_rate
        real_interest_rate = input_data.real_interest_rate[year]
        risk_premium_debt_cost = input_data.risk_premium_debt_cost

        kd_real_cost_debt = real_interest_rate + risk_premium_debt_cost
        # ku_perpetual = real_kk.iat[0]
        wacc = real_kk - tax_rate * input_data.perpetual_leverage * kd_real_cost_debt

        return cls(
            real_kk=real_kk,
            inflation_rate=inflation_rate,
            nominal_kk=nominal_kk,
            tax_rate=tax_rate,
            real_growth_rate=real_growth,
            real_interest_rate=real_interest_rate,
            risk_premium_debt_cost=risk_premium_debt_cost,
            kd_real_cost_debt=kd_real_cost_debt,
            ku_perpetual=real_kk.iloc[-1],
            wacc=wacc.iloc[-1],
        )

    def tv_and_liquidate(
        self,
        balance_sheet,
        income_statement,
    ) -> float:
        noplat = income_statement.ebit * (1 - self.tax_rate)
        tv = noplat / self.wacc
        # Cash
        cash = balance_sheet.cash_cb
        ar_discounted_wacc = balance_sheet.ar_it / (1 + self.wacc)
        market_securities = balance_sheet.st_investments_cb
        ap_discounted_wacc = -balance_sheet.ap_it / (1 + self.wacc)
        liquidation_assets = (
            cash + ar_discounted_wacc + market_securities + ap_discounted_wacc
        )
        adjusted_tv = tv + liquidation_assets

        return adjusted_tv

    def pretty_print(self) -> None:
        """Pretty print ValuationInputs data."""
        lines = ["=" * 60, "Valuation Inputs", "=" * 60]

        lines.append("\nCOST OF CAPITAL:")
        lines.append("-" * 60)
        lines.append("Real KK (Cost of Equity):")
        lines.append(str(self.real_kk))
        lines.append("\nInflation Rate:")
        lines.append(str(self.inflation_rate))
        lines.append("\nNominal KK:")
        lines.append(str(self.nominal_kk))

        lines.append("\n" + "=" * 60)
        lines.append("RATES AND RATIOS:")
        lines.append("-" * 60)
        lines.append(f"Tax Rate: {self.tax_rate:.4f}")
        lines.append(f"Real Growth Rate: {self.real_growth_rate:.4f}")
        lines.append(f"Real Interest Rate: {self.real_interest_rate:.4f}")
        lines.append(f"Risk Premium (Debt Cost): {self.risk_premium_debt_cost:.4f}")

        lines.append("\n" + "=" * 60)
        lines.append("CALCULATED METRICS:")
        lines.append("-" * 60)
        lines.append(f"KD (Real Cost of Debt): {self.kd_real_cost_debt:.4f}")

        # Handle ku_perpetual and wacc which might be Series
        if isinstance(self.ku_perpetual, pd.Series):
            lines.append("KU (Perpetual Cost of Equity):")
            lines.append(str(self.ku_perpetual))
        else:
            lines.append(f"KU (Perpetual Cost of Equity): {self.ku_perpetual:.4f}")

        if isinstance(self.wacc, pd.Series):
            lines.append("\nWACC (Weighted Avg Cost of Capital):")
            lines.append(str(self.wacc))
        else:
            lines.append(f"\nWACC (Weighted Avg Cost of Capital): {self.wacc:.4f}")

        lines.append("=" * 60)

        print("\n".join(lines))
