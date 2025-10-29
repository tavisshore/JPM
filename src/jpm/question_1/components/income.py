from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class IncomeStatement:
    years: pd.Index
    ebit: pd.Series  # InputData.ebit
    rtn_rate_st: pd.Series  # InputData.rtn_st_inv
    st_invest_end_prev: pd.Series  # EoY ST investments from CashBudget
    st_interest: pd.Series  # "Interest payment ST loan" from STLoanSchedule
    lt_interest: pd.Series  # "Interest payment LT loan" from LTLoanSchedule
    dividends: pd.Series | float = (
        0.0  # cash dividends decided this year (paid next year?)
    )

    def compute(self) -> pd.DataFrame:
        y = self.years

        # normalise inputs to the index
        ebit = self.ebit.reindex(y, fill_value=0.0)
        rtn_rate = self.rtn_rate_st.reindex(y, fill_value=0.0)
        st_inv_prev = self.st_invest_end_prev.reindex(y, fill_value=0.0)
        st_int = self.st_interest.reindex(y, fill_value=0.0)
        lt_int = self.lt_interest.reindex(y, fill_value=0.0)

        # Return from ST investments in year t is earned on prior-year invested cash
        # i.e., use previous year's ending ST-investments
        st_return = (st_inv_prev.shift(1).fillna(0.0)) * rtn_rate

        interest_total = st_int + lt_int
        net_income = ebit + st_return - interest_total

        if isinstance(self.dividends, pd.Series):
            dividends = self.dividends.reindex(y, fill_value=0.0)
        else:
            dividends = pd.Series(float(self.dividends), index=y)

        # Cumulated retained earnings (starting from 0)
        retained = (net_income - dividends).cumsum()

        df = pd.DataFrame(
            {
                "EBIT": ebit,
                "Return from ST investment": st_return,
                "Interest payments (ST+LT)": interest_total,
                "Net income": net_income,
                "Dividends (declared this year)": dividends,
                "Cumulated retained earnings": retained,
            }
        )
        return df
