import numpy as np
import pandas as pd
import pytest

from jpm.question_1.data import utils

unit = pytest.mark.unit
integration = pytest.mark.integration


# Tests for helper functions in utils.py (moved from ed.py)


@unit
def test_extract_leaf_mappings_flat():
    """_extract_leaf_mappings should extract leaf lists from flat dict."""
    mapping = {
        "Cash": ["cash_col1", "cash_col2"],
        "Inventory": ["inv_col"],
    }
    result = utils._extract_leaf_mappings(mapping)
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
    result = utils._extract_leaf_mappings(mapping)
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
    result = utils._extract_leaf_mappings(mapping)
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
    assert utils._is_mutually_exclusive(df, ["col_a", "col_b", "col_c"])


@unit
def test_is_mutually_exclusive_false():
    """_is_mutually_exclusive should return False when columns overlap."""
    df = pd.DataFrame(
        {
            "col_a": [100, 100, 100, 100],
            "col_b": [200, 200, 200, 200],
        }
    )
    assert not utils._is_mutually_exclusive(df, ["col_a", "col_b"])


@unit
def test_is_mutually_exclusive_single_col():
    """_is_mutually_exclusive should return False for single column."""
    df = pd.DataFrame({"col_a": [100, 200]})
    assert not utils._is_mutually_exclusive(df, ["col_a"])


@unit
def test_map_single_column_empty_list():
    """_map_single_column should return NaN for empty source list."""
    df = pd.DataFrame({"col": [1, 2]})
    result = utils._map_single_column(df, "new_col", [])
    assert np.isnan(result)


@unit
def test_map_single_column_no_existing():
    """_map_single_column should return NaN when no source columns exist."""
    df = pd.DataFrame({"col_a": [1, 2]})
    result = utils._map_single_column(df, "new_col", ["nonexistent1", "nonexistent2"])
    assert np.isnan(result)


@unit
def test_map_single_column_single_source():
    """_map_single_column should return source column directly for single match."""
    df = pd.DataFrame({"source_col": [100, 200, 300]})
    result = utils._map_single_column(df, "new_col", ["source_col"])
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
    result = utils._map_single_column(df, "new_col", ["col_a", "col_b", "col_c"])
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
    result = utils._map_single_column(df, "new_col", ["col_a", "col_b"])
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
    result = utils.remap_financial_dataframe(df, mapping)

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
    result = utils.remap_financial_dataframe(df, mapping)

    assert "Cash" in result.columns
    assert "PPE" in result.columns
    assert result["Cash"].iloc[0] == 100
    assert result["PPE"].iloc[0] == 500


@unit
def test_remap_financial_dataframe_missing_source():
    """remap_financial_dataframe should produce NaN for missing sources."""
    df = pd.DataFrame({"existing": [1, 2]})
    mapping = {"NewCol": ["nonexistent"]}
    result = utils.remap_financial_dataframe(df, mapping)

    assert "NewCol" in result.columns
    assert result["NewCol"].isna().all()
