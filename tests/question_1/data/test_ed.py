from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from jpm.question_1.config import Config, DataConfig, LLMConfig, LSTMConfig, XGBConfig
from jpm.question_1.data import ed

unit = pytest.mark.unit
integration = pytest.mark.integration


# Tests for refactored helper functions in ed.py


@unit
def test_extract_leaf_mappings_flat():
    """_extract_leaf_mappings should extract leaf lists from flat dict."""
    mapping = {
        "Cash": ["cash_col1", "cash_col2"],
        "Inventory": ["inv_col"],
    }
    result = ed._extract_leaf_mappings(mapping)
    assert result == {
        "Cash": ["cash_col1", "cash_col2"],
        "Inventory": ["inv_col"],
    }


@unit
def test_extract_leaf_mappings_nested():
    """_extract_leaf_mappings should extract leaf lists from nested dict."""
    mapping = {
        "assets": {
            "current": {
                "Cash": ["cash_col"],
                "Inventory": ["inv_col"],
            },
            "non_current": {
                "PPE": ["ppe_col"],
            },
        },
        "__unmapped__": ["should_be_ignored"],
    }
    result = ed._extract_leaf_mappings(mapping)
    assert result == {
        "Cash": ["cash_col"],
        "Inventory": ["inv_col"],
        "PPE": ["ppe_col"],
    }


@unit
def test_extract_leaf_mappings_ignores_unmapped():
    """_extract_leaf_mappings should skip __unmapped__ key."""
    mapping = {
        "Cash": ["cash_col"],
        "__unmapped__": ["some_unmapped_col"],
    }
    result = ed._extract_leaf_mappings(mapping)
    assert "__unmapped__" not in result
    assert result == {"Cash": ["cash_col"]}


@unit
def test_is_mutually_exclusive_true():
    """_is_mutually_exclusive should return True when columns don't overlap."""
    df = pd.DataFrame(
        {
            "col_a": [100, 0, 0, 0],
            "col_b": [0, 200, 0, 0],
            "col_c": [0, 0, 300, 0],
        }
    )
    assert ed._is_mutually_exclusive(df, ["col_a", "col_b", "col_c"])


@unit
def test_is_mutually_exclusive_false():
    """_is_mutually_exclusive should return False when columns overlap."""
    df = pd.DataFrame(
        {
            "col_a": [100, 100, 100, 100],
            "col_b": [200, 200, 200, 200],
        }
    )
    assert not ed._is_mutually_exclusive(df, ["col_a", "col_b"])


@unit
def test_is_mutually_exclusive_single_col():
    """_is_mutually_exclusive should return False for single column."""
    df = pd.DataFrame({"col_a": [100, 200]})
    assert not ed._is_mutually_exclusive(df, ["col_a"])


@unit
def test_map_single_column_empty_list():
    """_map_single_column should return NaN for empty source list."""
    df = pd.DataFrame({"col": [1, 2]})
    result = ed._map_single_column(df, "new_col", [])
    assert np.isnan(result)


@unit
def test_map_single_column_no_existing():
    """_map_single_column should return NaN when no source columns exist."""
    df = pd.DataFrame({"col_a": [1, 2]})
    result = ed._map_single_column(df, "new_col", ["nonexistent1", "nonexistent2"])
    assert np.isnan(result)


@unit
def test_map_single_column_single_source():
    """_map_single_column should return source column directly for single match."""
    df = pd.DataFrame({"source_col": [100, 200, 300]})
    result = ed._map_single_column(df, "new_col", ["source_col"])
    assert list(result) == [100, 200, 300]


@unit
def test_map_single_column_coalesce_mutually_exclusive():
    """_map_single_column should coalesce mutually exclusive columns."""
    df = pd.DataFrame(
        {
            "col_a": [100, 0, 0],
            "col_b": [0, 200, 0],
            "col_c": [0, 0, 300],
        }
    )
    result = ed._map_single_column(df, "new_col", ["col_a", "col_b", "col_c"])
    assert list(result) == [100, 200, 300]


@unit
def test_map_single_column_sum_overlapping():
    """_map_single_column should sum overlapping columns."""
    df = pd.DataFrame(
        {
            "col_a": [100, 100, 100],
            "col_b": [50, 50, 50],
        }
    )
    result = ed._map_single_column(df, "new_col", ["col_a", "col_b"])
    assert list(result) == [150, 150, 150]


@unit
def test_remap_financial_dataframe_basic():
    """remap_financial_dataframe should apply column mappings correctly."""
    df = pd.DataFrame(
        {
            "old_cash": [100, 200],
            "old_inventory": [50, 60],
            "unused": [999, 999],
        }
    )
    mapping = {
        "Cash": ["old_cash"],
        "Inventory": ["old_inventory"],
    }
    result = ed.remap_financial_dataframe(df, mapping)

    assert list(result.columns) == ["Cash", "Inventory"]
    assert list(result["Cash"]) == [100, 200]
    assert list(result["Inventory"]) == [50, 60]


@unit
def test_remap_financial_dataframe_nested_mapping():
    """remap_financial_dataframe should handle nested mappings."""
    df = pd.DataFrame(
        {
            "cash_col": [100],
            "ppe_col": [500],
        }
    )
    mapping = {
        "assets": {
            "current": {"Cash": ["cash_col"]},
            "non_current": {"PPE": ["ppe_col"]},
        }
    }
    result = ed.remap_financial_dataframe(df, mapping)

    assert "Cash" in result.columns
    assert "PPE" in result.columns
    assert result["Cash"].iloc[0] == 100
    assert result["PPE"].iloc[0] == 500


@unit
def test_remap_financial_dataframe_missing_source():
    """remap_financial_dataframe should produce NaN for missing sources."""
    df = pd.DataFrame({"existing": [1, 2]})
    mapping = {"NewCol": ["nonexistent"]}
    result = ed.remap_financial_dataframe(df, mapping)

    assert "NewCol" in result.columns
    assert result["NewCol"].isna().all()


def make_config(tmp_path):
    return Config(
        data=DataConfig(
            ticker="AAPL",
            cache_dir=str(tmp_path),
            periods=2,
            lookback=1,
            horizon=1,
            batch_size=2,
            target_type="full",
        ),
        llm=LLMConfig(),
        xgb=XGBConfig(),
        lstm=LSTMConfig(),
    )


@unit
def test_map_features_creates_expected_mappings(tmp_path):
    """map_features should translate structure names into target indices."""
    from jpm.question_1.data.ed import EdgarDataLoader

    config = make_config(tmp_path)
    loader = object.__new__(EdgarDataLoader)
    loader.config = config
    loader.bs_structure = {
        "assets": {
            "current_assets": ["cash", "inventory"],
            "non_current_assets": ["ppe"],
        },
        "liabilities": {
            "current_liabilities": ["payables"],
            "non_current_liabilities": ["lt_debt"],
        },
        "equity": ["equity"],
    }
    loader.targets = ["cash", "inventory", "ppe", "payables", "lt_debt", "equity"]
    loader.map_features()

    assert loader.feature_mappings["assets"] == [0, 1, 2]
    assert loader.feature_mappings["liabilities"] == [3, 4]
    assert loader.feature_mappings["equity"] == [5]


@unit
def test_process_statement_normalizes_and_filters(monkeypatch, tmp_path):
    """_process_statement should lower-case concepts and filter to needed columns."""
    from jpm.question_1.data.ed import EdgarDataLoader

    config = make_config(tmp_path)
    loader = object.__new__(EdgarDataLoader)
    loader.config = config

    df = pd.DataFrame(
        {
            "concept": ["us-gaap_Cash", "us-gaap_Cash"],
            "2022-12-31": [100, 200],
            "2023-12-31": [150, 250],
        }
    )
    stmt = MagicMock()
    stmt.to_dataframe.return_value = df

    processed = loader._process_statement(
        stmt=stmt, kind="balance sheet", needed_cols=["cash"]
    )
    assert list(processed.columns) == ["cash"]
    # Duplicate concept collapsed; latest value retained
    assert processed.iloc[0, 0] == 100


@unit
def test_process_statement_raises_when_missing_statement(tmp_path):
    """_process_statement should raise ValueError when stmt is None."""
    from jpm.question_1.data.ed import EdgarDataLoader

    config = make_config(tmp_path)
    loader = object.__new__(EdgarDataLoader)
    loader.config = config

    with pytest.raises(ValueError):
        loader._process_statement(stmt=None, kind="balance sheet", needed_cols=["cash"])


@unit
def test_map_features_ignores_missing_targets(tmp_path):
    """map_features should skip structure entries that are absent from targets."""
    from jpm.question_1.data.ed import EdgarDataLoader

    config = make_config(tmp_path)
    loader = object.__new__(EdgarDataLoader)
    loader.config = config
    loader.bs_structure = {
        "assets": {
            "current_assets": ["cash", "goodwill_not_tracked"],
            "non_current_assets": [],
        },
        "liabilities": {"current_liabilities": [], "non_current_liabilities": []},
        "equity": [],
    }
    loader.targets = ["cash"]
    loader.map_features()

    assert loader.feature_mappings["current_assets"] == [0]


@unit
def test_get_final_window_uses_original_timestamp_index(tmp_path):
    """get_final_window should restore the original data timestamps on outputs."""
    from jpm.question_1.data.ed import EdgarDataLoader

    lookback = 2
    horizon = 1
    config = Config(
        data=DataConfig(cache_dir=str(tmp_path), lookback=lookback, horizon=horizon)
    )
    loader = object.__new__(EdgarDataLoader)
    loader.config = config

    periods = pd.period_range("2020-03-31", periods=4, freq="Q")
    loader.data = pd.DataFrame(
        {
            "cash": [10.0, 20.0, 30.0, 40.0],
            "inventory": [1.0, 2.0, 3.0, 4.0],
        },
        index=periods,
    )
    loader.targets = list(loader.data.columns)
    loader.feat_to_idx = {name: idx for idx, name in enumerate(loader.targets)}
    loader.tgt_indices = list(range(len(loader.targets)))
    loader.full_mean = np.zeros(len(loader.targets))
    loader.full_std = np.ones(len(loader.targets))
    loader.target_mean = loader.full_mean
    loader.target_std = loader.full_std

    # Final window corresponds to rows 1:3 with the last row as the target
    loader.X_test = np.expand_dims(loader.data.values[1:3], axis=0)
    loader.y_test = np.expand_dims(loader.data.values[-1], axis=0)

    X_named, y_named = loader.get_final_window()

    expected_idx = loader.data.index.to_timestamp()
    start_idx = len(loader.data) - (lookback + horizon)

    assert list(X_named.index) == list(expected_idx[start_idx : start_idx + lookback])
    assert y_named.name == expected_idx[-1]
    assert isinstance(X_named.index[0], pd.Timestamp)


@integration
def test_create_dataset_uses_cache_when_available(monkeypatch, tmp_path):
    """EdgarDataLoader should load cached parquet instead of hitting EDGAR
    when available."""
    from jpm.question_1.data.ed import EdgarDataLoader

    config = make_config(tmp_path)
    fake_df = pd.DataFrame(
        {
            "cash": [100.0, 110.0, 120.0],
            "inventory": [50.0, 55.0, 60.0],
        }
    )
    cache_path = Path(config.data.cache_dir) / f"{config.data.ticker}.parquet"
    fake_df.to_parquet(cache_path)

    with (
        patch("jpm.question_1.data.ed.bs_identity") as bs_identity_mock,
        patch("jpm.question_1.data.ed.get_targets", return_value=list(fake_df.columns)),
    ):
        loader = EdgarDataLoader(config=config)

    assert bs_identity_mock.called
    assert loader.train_dataset is not None
    assert loader.val_dataset is not None
