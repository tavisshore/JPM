from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class InputData:
    years: pd.Index
    ebit: pd.Series
    depreciation: pd.Series
    net_fixed_assets: pd.Series
    min_cash: pd.Series
    kd: pd.Series
    rtn_st_inv: pd.Series
    equity_investment: float
    st_loan_term: int
    lt_loan_term: int
    ebitda: pd.Series = field(init=False, repr=True)

    def __post_init__(self):
        object.__setattr__(self, "ebit", self.ebit.fillna(0))
        object.__setattr__(self, "depreciation", self.depreciation.fillna(0))
        object.__setattr__(self, "net_fixed_assets", self.net_fixed_assets.fillna(0))
        object.__setattr__(self, "min_cash", self.min_cash.fillna(0))
        object.__setattr__(self, "kd", self.kd.fillna(0))
        object.__setattr__(self, "rtn_st_inv", self.rtn_st_inv.fillna(0))
        object.__setattr__(self, "ebitda", (self.ebit + self.depreciation).fillna(0.0))
