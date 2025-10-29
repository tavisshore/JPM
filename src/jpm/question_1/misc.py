import math

import pandas as pd


def as_series(mapping, years):
    return pd.Series(
        [mapping.get(y, math.nan) for y in years], index=years, dtype="float64"
    )
