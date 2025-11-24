import numpy as np
import pandas as pd
import pytest

from jpm.question_1.data import utils

unit = pytest.mark.unit
integration = pytest.mark.integration


@unit
def test_xbrl_to_snake_strips_prefix_and_snake_cases():
    """xbrl_to_snake should remove prefixes and insert underscores."""
    assert utils.xbrl_to_snake("us-gaap_RevenueFromSales") == "revenue_from_sales"
    assert utils.xbrl_to_snake("custom-metric_CostOfGoods") == "cost_of_goods"


@unit
def test_get_targets_returns_structure_leaves(monkeypatch):
    """get_targets should return BS leaves when mode=bs."""
    fake_structure = {
        "assets": {"current_assets": ["cash"], "non_current_assets": []},
        "liabilities": {"current_liabilities": [], "non_current_liabilities": []},
        "equity": ["retained_earnings"],
    }
    monkeypatch.setattr(utils, "get_bs_structure", lambda ticker: fake_structure)
    assert utils.get_targets(mode="bs", ticker="AAPL") == [
        "cash",
        "retained_earnings",
    ]


@unit
def test_get_bs_structure_returns_defaults_for_unknown_ticker():
    """Unsupported tickers should raise to surface missing mappings."""
    with pytest.raises(ValueError):
        utils.get_bs_structure(ticker="UNKNOWN")


@unit
def test_get_cf_structure_flatten(monkeypatch):
    """get_cf_structure(flatten=True) should return flattened list of leaves."""
    monkeypatch.setattr(
        utils, "get_leaf_values", lambda d, sub_key=None: ["net_income"]
    )
    flattened = utils.get_cf_structure(ticker="AAPL", flatten=True)
    assert flattened == ["net_income"]


@unit
def test_build_windows_splits_train_and_test():
    """build_windows should split sliding windows into train/test
    respecting the withhold."""
    X = np.arange(12, dtype=float).reshape(6, 2)
    X_train, y_train, X_test, y_test = utils.build_windows(
        X, lookback=2, horizon=1, tgt_indices=[0], withhold=1
    )
    assert X_train.shape == (3, 2, 2)
    assert y_train.shape == (3, 1)
    assert X_test.shape == (1, 2, 2)
    assert y_test.shape == (1, 1)
    np.testing.assert_allclose(y_test[0], X[-1, 0])


@unit
def test_build_windows_raises_on_invalid_params():
    """build_windows should raise ValueError for impossible configurations."""
    X = np.arange(6, dtype=float).reshape(3, 2)
    with pytest.raises(ValueError):
        utils.build_windows(X, lookback=3, horizon=1, withhold=0)
    with pytest.raises(ValueError):
        utils.build_windows(X, lookback=1, horizon=1, withhold=-1)


@integration
def test_bs_identity_adds_validation_columns(capsys):
    """bs_identity should compute sums and print validity counts."""
    structure = utils.get_bs_structure("AAPL")
    row = {}
    for col in structure["assets"]["current_assets"]:
        row[col] = 10
    for col in structure["assets"]["non_current_assets"]:
        row[col] = 10
    for col in structure["liabilities"]["current_liabilities"]:
        row[col] = 5
    for col in structure["liabilities"]["non_current_liabilities"]:
        row[col] = 5
    for col in structure["equity"]:
        row[col] = 10

    df = pd.DataFrame([row])
    utils.bs_identity(df, ticker="AAPL", tol=1e-3)
    captured = capsys.readouterr().out
    assert "Accounting Identity" in captured
