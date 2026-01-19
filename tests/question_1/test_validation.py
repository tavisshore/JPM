import numpy as np
import pandas as pd
import pytest

from jpm.question_1.config import DataConfig
from jpm.question_1.data import utils as data_utils


def test_dataconfig_rejects_empty_ticker():
    with pytest.raises(ValueError):
        DataConfig(ticker="", cache_dir="/tmp")


def test_build_windows_rejects_wrong_shape():
    bad = np.arange(10)  # 1D instead of 2D
    with pytest.raises(ValueError):
        data_utils.build_windows(bad, lookback=2, horizon=1)


def test_build_windows_rejects_oob_indices():
    X = np.zeros((5, 3))
    with pytest.raises(IndexError):
        data_utils.build_windows(X, lookback=2, horizon=1, tgt_indices=[5])


def test_bs_identity_missing_columns():
    df = pd.DataFrame({"some_col": [1.0, 2.0]})
    with pytest.raises(ValueError):
        data_utils.bs_identity(df, ticker="AAPL")


def test_xbrl_to_snake_rejects_empty():
    with pytest.raises(ValueError):
        data_utils.xbrl_to_snake("")


def test_get_bs_structure_unsupported_ticker():
    with pytest.raises(ValueError):
        data_utils.get_bs_structure(ticker="UNKNOWN")


def test_get_targets_invalid_mode():
    with pytest.raises(ValueError):
        data_utils.get_targets(mode="invalid", ticker="AAPL")
