import pytest

from jpm.question_1.config import DataConfig
from jpm.question_1.data import utils as data_utils


def test_dataconfig_rejects_empty_ticker():
    with pytest.raises(ValueError):
        DataConfig(ticker="", cache_dir="/tmp")


def test_xbrl_to_snake_rejects_empty():
    with pytest.raises(ValueError):
        data_utils.xbrl_to_snake("")


def test_xbrl_to_snake_strips_prefix():
    """xbrl_to_snake should strip namespace prefix and convert to snake_case."""
    result = data_utils.xbrl_to_snake("us-gaap_RevenueFromSales")
    assert result == "revenue_from_sales"


def test_add_derived_columns_handles_empty_df():
    """add_derived_columns should handle empty DataFrame gracefully."""
    import pandas as pd

    df = pd.DataFrame()
    result = data_utils.add_derived_columns(df)
    assert result.empty


def test_should_calculate_returns_true_for_missing():
    """_should_calculate should return True for missing columns."""
    import pandas as pd

    df = pd.DataFrame({"existing": [1, 2, 3]})
    assert data_utils._should_calculate(df, "nonexistent")


def test_should_calculate_returns_false_for_valid_data():
    """_should_calculate should return False for columns with valid data."""
    import pandas as pd

    df = pd.DataFrame({"valid": [1, 2, 3]})
    assert not data_utils._should_calculate(df, "valid")


def test_derived_col_returns_zeros_for_missing():
    """_derived_col should return zeros for missing columns."""
    import pandas as pd

    df = pd.DataFrame({"existing": [1, 2, 3]})
    result = data_utils._derived_col(df, "missing")
    assert list(result) == [0, 0, 0]


def test_remap_financial_dataframe_basic():
    """remap_financial_dataframe should apply mappings correctly."""
    import pandas as pd

    df = pd.DataFrame({"old_col": [100, 200]})
    mapping = {"New Column": ["old_col"]}
    result = data_utils.remap_financial_dataframe(df, mapping)

    assert "New Column" in result.columns
    assert list(result["New Column"]) == [100, 200]


def test_extract_leaf_mappings_flat():
    """_extract_leaf_mappings should handle flat dictionaries."""
    mapping = {"Cash": ["cash_col"], "Inventory": ["inv_col"]}
    result = data_utils._extract_leaf_mappings(mapping)
    assert result == {"Cash": ["cash_col"], "Inventory": ["inv_col"]}


def test_extract_leaf_mappings_nested():
    """_extract_leaf_mappings should handle nested dictionaries."""
    mapping = {
        "assets": {
            "current": {"Cash": ["cash_col"]},
        }
    }
    result = data_utils._extract_leaf_mappings(mapping)
    assert result == {"Cash": ["cash_col"]}
