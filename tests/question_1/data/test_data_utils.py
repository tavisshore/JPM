import numpy as np
import pandas as pd
import pytest

from jpm.question_1.data import utils

unit = pytest.mark.unit
integration = pytest.mark.integration


# Tests for refactored helper functions (_bs_*, _derived_*, _should_calculate)


@unit
def test_bs_has_data_returns_true_for_nonempty_column():
    """_bs_has_data should return True when column exists with non-zero values."""
    df = pd.DataFrame({"col_a": [0, 10, 20], "col_b": [0, 0, 0]})
    assert utils._bs_has_data(df, "col_a")
    assert not utils._bs_has_data(df, "col_b")
    assert not utils._bs_has_data(df, "nonexistent")


@unit
def test_bs_get_field_returns_zero_for_missing():
    """_bs_get_field should return 0 when column is missing or empty."""
    df = pd.DataFrame({"col_a": [100, 200], "col_b": [0, 0]})
    result_a = utils._bs_get_field(df, "col_a")
    result_b = utils._bs_get_field(df, "col_b")
    result_c = utils._bs_get_field(df, "nonexistent")

    assert list(result_a) == [100, 200]
    assert result_b == 0
    assert result_c == 0


@unit
def test_bs_get_deferred_tax_components_net_position():
    """_bs_get_deferred_tax_components should split net DT by sign."""
    df = pd.DataFrame({"Deferred Tax Assets": [100, -50, 0]})
    mappings = {"Deferred Tax Assets": ["SomeNetDeferredTaxAsset"]}

    dta, dtl, has_net = utils._bs_get_deferred_tax_components(df, mappings)

    assert has_net is True
    np.testing.assert_array_equal(dta, [100, 0, 0])
    np.testing.assert_array_equal(dtl, [0, 50, 0])


@unit
def test_bs_get_deferred_tax_components_separate():
    """_bs_get_deferred_tax_components should return separate fields when not net."""
    df = pd.DataFrame(
        {"Deferred Tax Assets": [100, 200], "Deferred Tax Liabilities": [50, 75]}
    )
    mappings = {"Deferred Tax Assets": ["SeparateDTA"]}

    dta, dtl, has_net = utils._bs_get_deferred_tax_components(df, mappings)

    assert has_net is False
    assert list(dta) == [100, 200]
    assert list(dtl) == [50, 75]


@unit
def test_bs_reconstruct_assets_sums_components():
    """_bs_reconstruct_assets should sum all asset components."""
    df = pd.DataFrame(
        {
            "Cash and Equivalents": [100],
            "Receivables": [200],
            "Inventory": [50],
        }
    )
    dta_component = pd.Series([10])
    result = utils._bs_reconstruct_assets(df, dta_component)
    # Only the columns present contribute; others are 0
    assert result.iloc[0] == 360  # 100 + 200 + 50 + 10


@unit
def test_bs_reconstruct_liabilities_sums_components():
    """_bs_reconstruct_liabilities should sum all liability components."""
    df = pd.DataFrame(
        {
            "Accounts Payable and Accrued Expenses": [100],
            "Short-term Debt": [50],
            "Long-term Debt": [200],
        }
    )
    dtl_component = pd.Series([25])
    result = utils._bs_reconstruct_liabilities(df, dtl_component)
    assert result.iloc[0] == 375  # 100 + 50 + 200 + 25


@unit
def test_bs_reconstruct_equity_with_treasury():
    """_bs_reconstruct_equity should subtract Treasury Stock when present."""
    df = pd.DataFrame(
        {
            "Common Stock and APIC": [1000],
            "Retained Earnings": [500],
            "Accumulated Other Comprehensive Income": [50],
            "Treasury Stock": [100],
        }
    )
    mappings = {"Treasury Stock": ["TreasuryStockValue"]}

    equity, has_treasury = utils._bs_reconstruct_equity(df, mappings)

    assert has_treasury
    assert equity.iloc[0] == 1450  # 1000 + 500 + 50 - 100


@unit
def test_bs_reconstruct_equity_without_treasury():
    """_bs_reconstruct_equity should not deduct when Treasury Stock missing."""
    df = pd.DataFrame(
        {
            "Common Stock and APIC": [1000],
            "Retained Earnings": [500],
        }
    )
    mappings = {}

    equity, has_treasury = utils._bs_reconstruct_equity(df, mappings)

    assert not has_treasury
    assert equity.iloc[0] == 1500


@unit
def test_derived_col_returns_zero_for_missing():
    """_derived_col should return Series of zeros when column missing."""
    df = pd.DataFrame({"existing": [1, 2, 3]}, index=[0, 1, 2])
    result = utils._derived_col(df, "nonexistent")
    assert list(result) == [0, 0, 0]


@unit
def test_derived_col_fills_na_with_zero():
    """_derived_col should fill NaN values with 0."""
    df = pd.DataFrame({"col": [1.0, np.nan, 3.0]})
    result = utils._derived_col(df, "col")
    assert list(result) == [1.0, 0.0, 3.0]


@unit
def test_should_calculate_true_for_missing():
    """_should_calculate should return True when column is missing."""
    df = pd.DataFrame({"existing": [1, 2]})
    assert utils._should_calculate(df, "nonexistent")


@unit
def test_should_calculate_true_for_all_zero():
    """_should_calculate should return True when column is all zeros."""
    df = pd.DataFrame({"zeros": [0, 0, 0]})
    assert utils._should_calculate(df, "zeros")


@unit
def test_should_calculate_true_for_all_nan():
    """_should_calculate should return True when column is all NaN."""
    df = pd.DataFrame({"nans": [np.nan, np.nan]})
    assert utils._should_calculate(df, "nans")


@unit
def test_should_calculate_false_for_valid_data():
    """_should_calculate should return False when column has valid data."""
    df = pd.DataFrame({"valid": [1, 2, 3]})
    assert not utils._should_calculate(df, "valid")


@unit
def test_add_derived_assets_calculates_totals():
    """_add_derived_assets should compute Total Current/Non-Current/Assets."""
    df = pd.DataFrame(
        {
            "Cash and Equivalents": [100.0],
            "Receivables": [200.0],
            "Property, Plant, and Equipment (net)": [500.0],
        }
    )
    utils._add_derived_assets(df)

    assert "Total Current Assets" in df.columns
    assert "Total Non-Current Assets" in df.columns
    assert df["Total Current Assets"].iloc[0] == 300.0
    assert df["Total Non-Current Assets"].iloc[0] == 500.0


@unit
def test_add_derived_liabilities_calculates_totals():
    """_add_derived_liabilities should compute liability totals."""
    df = pd.DataFrame(
        {
            "Accounts Payable and Accrued Expenses": [100.0],
            "Short-term Debt": [50.0],
            "Long-term Debt": [300.0],
        }
    )
    utils._add_derived_liabilities(df)

    assert "Total Current Liabilities" in df.columns
    assert "Total Non-Current Liabilities" in df.columns
    assert df["Total Current Liabilities"].iloc[0] == 150.0
    assert df["Total Non-Current Liabilities"].iloc[0] == 300.0


@unit
def test_add_derived_equity_calculates_total():
    """_add_derived_equity should compute Total Equity."""
    df = pd.DataFrame(
        {
            "Common Stock and APIC": [1000.0],
            "Retained Earnings": [500.0],
            "Treasury Stock": [100.0],
        }
    )
    utils._add_derived_equity(df)

    assert "Total Equity" in df.columns
    # 1000 + 500 - 100 + 0 (AOCI) = 1400
    assert df["Total Equity"].iloc[0] == 1400.0


@unit
def test_add_derived_income_calculates_gross_profit():
    """_add_derived_income should calculate Gross Profit from revenue - COGS."""
    df = pd.DataFrame(
        {
            "Total Revenues": [1000.0],
            "Total Cost of Revenue": [600.0],
        }
    )
    utils._add_derived_income(df)

    assert "Gross Profit" in df.columns
    assert df["Gross Profit"].iloc[0] == 400.0


@unit
def test_add_derived_income_calculates_total_debt():
    """_add_derived_income should calculate Total Debt."""
    df = pd.DataFrame(
        {
            "Short-term Debt": [100.0],
            "Long-term Debt": [400.0],
        }
    )
    utils._add_derived_income(df)

    assert "Total Debt" in df.columns
    assert df["Total Debt"].iloc[0] == 500.0


@unit
def test_add_derived_columns_preserves_existing():
    """add_derived_columns should not overwrite existing valid data."""
    df = pd.DataFrame(
        {
            "Cash and Equivalents": [100.0],
            "Total Current Assets": [999.0],  # Should NOT be overwritten
        }
    )
    result = utils.add_derived_columns(df)

    assert result["Total Current Assets"].iloc[0] == 999.0


@unit
def test_add_derived_columns_overwrites_zeros():
    """add_derived_columns should overwrite columns that are all zeros."""
    df = pd.DataFrame(
        {
            "Cash and Equivalents": [100.0],
            "Receivables": [200.0],
            "Total Current Assets": [0.0],  # Should be recalculated
        }
    )
    result = utils.add_derived_columns(df)

    assert result["Total Current Assets"].iloc[0] == 300.0


@unit
def test_xbrl_to_snake_strips_prefix_and_snake_cases():
    """xbrl_to_snake should remove prefixes and insert underscores."""
    assert utils.xbrl_to_snake("us-gaap_RevenueFromSales") == "revenue_from_sales"
    assert utils.xbrl_to_snake("custom-metric_CostOfGoods") == "cost_of_goods"


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
