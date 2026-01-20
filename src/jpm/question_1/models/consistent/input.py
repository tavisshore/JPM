from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd


def series(defaults: Iterable[float]) -> field:
    return field(default_factory=lambda: list(defaults))


@dataclass()
class InputData:
    """Core input parameters used across financial schedules."""

    years: pd.Index

    net_fixed_assets: float
    lineal_depreciation: pd.Series

    corporate_tax_rate: float
    initial_inventory: float
    initial_purchase_price: float
    estimated_overhead_expenses: float
    admin_and_sales_payroll: float

    lt_years_loan_3: int
    st_years_loan_2: int

    inflation_rate: pd.Series
    real_increase_selling_price: pd.Series
    real_increase_purchase_price: pd.Series
    real_increase_overheads: pd.Series
    real_increase_payroll: pd.Series
    increase_sales_volume: pd.Series
    real_interest_rate: float
    risk_premium_debt_cost: float
    risk_premium_return_st_inv: float

    # Valuation purposes
    observed_kk: float
    perpetual_leverage: float
    expected_inflation_rate: float
    real_growth_rate: float


@dataclass()
class MarketResearchInput:
    """Market research assumptions for pricing and elasticity."""

    selling_price: float
    elasticity_b: float
    elasticity_coef: float


@dataclass
class PolicyTable:
    """Operational policies and percentages applied to model calculations."""

    years: pd.Index

    promo_ad: pd.Series = series([0.0, 0.03, 0.0, 0.0, 0.0])
    inventory_pct: pd.Series = series([0.0, 1 / 12, 1 / 12, 1 / 12, 1 / 12])
    ar_pct: pd.Series = series([0.0, 0.05, 0.05, 0.05, 0.05])
    adv_from_cust_pct: pd.Series = series([0.0, 0.10, 0.10, 0.10, 0.10])
    ap_pct: pd.Series = series([0.0, 0.10, 0.10, 0.10, 0.10])
    adv_to_suppliers_pct: pd.Series = series([0.0, 0.10, 0.10, 0.10, 0.10])
    payout_ratio: pd.Series = series([0.0, 0.70, 0.70, 0.70, 0.70])
    cash_pct_of_sales: pd.Series = series([0.0, 0.04, 0.04, 0.04, 0.04])
    debt_financing_pct: float = 0.70
    minimum_initial_cash: float = 13.0
    selling_commission_pct: pd.Series = series([0.0, 0.04, 0.04, 0.04, 0.04])
    stock_repurchase_pct: pd.Series = series([0.0, 0.0, 0.0, 0.0, 0.0])

    def __post_init__(self):
        def to_series(value, name):
            vals = value.values if isinstance(value, pd.Series) else list(value)
            if len(vals) != len(self.years):
                raise ValueError(
                    f"{name} length {len(vals)} does not match length {len(self.years)}"
                )
            return pd.Series(vals, index=self.years)

        self.promo_ad = to_series(self.promo_ad, "promo_ad")
        self.inventory_pct = to_series(self.inventory_pct, "inventory_pct")
        self.ar_pct = to_series(self.ar_pct, "ar_pct")
        self.adv_from_cust_pct = to_series(self.adv_from_cust_pct, "adv_from_cust_pct")
        self.ap_pct = to_series(self.ap_pct, "ap_pct")
        self.adv_to_suppliers_pct = to_series(
            self.adv_to_suppliers_pct, "adv_to_suppliers_pct"
        )
        self.payout_ratio = to_series(self.payout_ratio, "payout_ratio")
        self.cash_pct_of_sales = to_series(self.cash_pct_of_sales, "cash_pct_of_sales")
        self.selling_commission_pct = to_series(
            self.selling_commission_pct, "selling_commission_pct"
        )
        self.stock_repurchase_pct = to_series(
            self.stock_repurchase_pct, "stock_repurchase_pct"
        )
