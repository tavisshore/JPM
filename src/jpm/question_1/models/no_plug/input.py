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
    equity_investment: pd.Series
    st_loan_term: int
    lt_loan_term: int
    ebitda: pd.Series = field(init=False, repr=True)

    def __post_init__(self):
        years_idx = pd.Index(self.years)
        if years_idx.empty:
            raise ValueError("years index must contain at least one period")
        if self.st_loan_term <= 0 or self.lt_loan_term <= 0:
            raise ValueError("loan terms must be positive integers")

        def _clean_series(name: str, value: pd.Series) -> pd.Series:
            if not isinstance(value, pd.Series):
                raise TypeError(f"{name} must be a pandas Series")
            missing_years = years_idx.difference(value.index)
            if not missing_years.empty:
                raise ValueError(
                    f"{name} missing data for years: {missing_years.tolist()}"
                )
            return value.reindex(years_idx).fillna(0)

        cleaned_ebit = _clean_series("ebit", self.ebit)
        cleaned_dep = _clean_series("depreciation", self.depreciation)
        cleaned_nfa = _clean_series("net_fixed_assets", self.net_fixed_assets)
        cleaned_min_cash = _clean_series("min_cash", self.min_cash)
        cleaned_kd = _clean_series("kd", self.kd)
        cleaned_rtn_inv = _clean_series("rtn_st_inv", self.rtn_st_inv)
        cleaned_equity = _clean_series("equity_investment", self.equity_investment)

        object.__setattr__(self, "ebit", cleaned_ebit)
        object.__setattr__(self, "depreciation", cleaned_dep)
        object.__setattr__(self, "net_fixed_assets", cleaned_nfa)
        object.__setattr__(self, "min_cash", cleaned_min_cash)
        object.__setattr__(self, "kd", cleaned_kd)
        object.__setattr__(self, "rtn_st_inv", cleaned_rtn_inv)
        object.__setattr__(self, "equity_investment", cleaned_equity)

        object.__setattr__(self, "ebitda", (cleaned_ebit + cleaned_dep).fillna(0.0))
