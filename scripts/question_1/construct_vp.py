from __future__ import annotations

import pandas as pd

from jpm.question_1.misc import as_series
from jpm.question_1.models.consistent import InputData

years = pd.Index([0, 1, 2, 3], name="year")

input_data = InputData(
    years=years,
    ebit=as_series({0: 0, 1: 5, 2: 9, 3: 12.0}, years),
    depreciation=as_series({0: 0, 1: 9, 2: 9, 3: 9}, years),
    net_fixed_assets=as_series({0: 45, 1: 36, 2: 27, 3: 18}, years),
    min_cash=as_series({0: 10, 1: 10, 2: 10, 3: 10}, years),
    kd=as_series({0: 0, 1: 0.13, 2: 0.13, 3: 0.13}, years),
    rtn_st_inv=as_series({0: 0.08, 1: 0.08, 2: 0.08, 3: 0.08}, years),
    equity_investment=as_series({0: 25, 1: 0, 2: 0, 3: 0}, years),
    st_loan_term=1,
    lt_loan_term=5,
)
