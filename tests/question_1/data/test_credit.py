"""Tests for credit.py - Financial ratio calculations for credit rating prediction."""

import numpy as np
import pandas as pd
import pytest

from jpm.question_1.data import credit

unit = pytest.mark.unit
integration = pytest.mark.integration


# Tests for helper functions


@unit
def test_safe_get_existing_column():
    """_safe_get should return column when it exists."""
    df = pd.DataFrame({"col_a": [1, 2, 3]})
    result = credit._safe_get(df, "col_a")
    assert list(result) == [1, 2, 3]


@unit
def test_safe_get_missing_column():
    """_safe_get should return default for missing column."""
    df = pd.DataFrame({"col_a": [1, 2, 3]})
    result = credit._safe_get(df, "nonexistent", default=0)
    assert list(result) == [0, 0, 0]


@unit
def test_safe_get_nan_default():
    """_safe_get should return NaN series by default for missing column."""
    df = pd.DataFrame({"col_a": [1, 2]})
    result = credit._safe_get(df, "nonexistent")
    assert result.isna().all()


@unit
def test_safe_ratio_normal():
    """_safe_ratio should calculate ratio normally."""
    num = pd.Series([10, 20, 30])
    den = pd.Series([2, 4, 5])
    result = credit._safe_ratio(num, den)
    assert list(result) == [5.0, 5.0, 6.0]


@unit
def test_safe_ratio_division_by_zero():
    """_safe_ratio should return NaN for division by zero."""
    num = pd.Series([10, 20, 30])
    den = pd.Series([2, 0, 5])
    result = credit._safe_ratio(num, den)
    assert result.iloc[0] == 5.0
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == 6.0


@unit
def test_safe_ratio_negative_infinity():
    """_safe_ratio should convert infinities to NaN."""
    num = pd.Series([10, -20])
    den = pd.Series([0, 0])
    result = credit._safe_ratio(num, den)
    assert result.isna().all()


# Tests for derived values calculation


@unit
def test_calculate_derived_values_total_debt():
    """_calculate_derived_values should calculate total debt."""
    df = pd.DataFrame(
        {
            "Short-term Debt": [100, 200],
            "Long-term Debt": [300, 400],
        }
    )

    total_debt, _, _ = credit._calculate_derived_values(df)

    assert list(total_debt) == [400, 600]


@unit
def test_calculate_derived_values_ebitda():
    """_calculate_derived_values should calculate EBITDA."""
    df = pd.DataFrame(
        {
            "Operating Income": [100, 200],
            "Depreciation and Amortization": [50, 75],
        }
    )

    _, ebitda, _ = credit._calculate_derived_values(df)

    assert list(ebitda) == [150, 275]


@unit
def test_calculate_derived_values_fcf():
    """_calculate_derived_values should calculate free cash flow."""
    df = pd.DataFrame(
        {
            "Net Cash from Operating Activities": [200, 300],
            "Capital Expenditures": [50, 100],
        }
    )

    _, _, fcf = credit._calculate_derived_values(df)

    assert list(fcf) == [150, 200]


@unit
def test_calculate_derived_values_missing_columns():
    """_calculate_derived_values should return None for missing columns."""
    df = pd.DataFrame({"Unrelated Column": [1, 2, 3]})

    total_debt, ebitda, fcf = credit._calculate_derived_values(df)

    assert total_debt is None
    assert ebitda is None
    assert fcf is None


@integration
def test_calculate_credit_ratios_comprehensive():
    """calculate_credit_ratios should calculate JPM ratios (quick_ratio, debt ratios)."""
    df = pd.DataFrame(
        {
            "rating": ["A", "BBB"],
            "Total Assets": [1000, 2000],
            "Total Equity": [500, 1000],
            "Total Liabilities": [500, 1000],
            "Total Current Assets": [300, 600],
            "Total Current Liabilities": [150, 300],
            "Cash and Equivalents": [100, 200],
            "Inventory": [50, 100],
            "Short-term Debt": [50, 100],
            "Long-term Debt": [200, 400],
            "Total Revenues": [2000, 4000],
            "Net Income": [100, 200],
            "Operating Income": [150, 300],
            "Gross Profit": [800, 1600],
            "Net Cash from Operating Activities": [200, 400],
            "Capital Expenditures": [50, 100],
        }
    )

    ratios = credit.calculate_credit_ratios(df)

    # Check that rating is preserved
    assert "rating" in ratios.columns
    assert list(ratios["rating"]) == ["A", "BBB"]

    # Check JPM ratios (lowercase names from _calculate_jpm_ratios)
    assert "quick_ratio" in ratios.columns
    assert "debt_to_equity" in ratios.columns
    assert "debt_to_assets" in ratios.columns
    assert "debt_to_capital" in ratios.columns
    assert "debt_to_ebitda" in ratios.columns


@integration
def test_calculate_credit_ratios_missing_features():
    """calculate_credit_ratios should handle missing features gracefully."""
    df = pd.DataFrame(
        {
            "Total Assets": [1000],
            "Total Equity": [500],
        }
    )

    ratios = credit.calculate_credit_ratios(df)

    # With only Total Assets and Total Equity, no JPM ratios can be calculated
    # (they require debt components or current assets/liabilities)
    # The result should be an empty or minimal DataFrame
    assert len(ratios) == 1  # Still has one row


@integration
def test_calculate_credit_ratios_empty_dataframe():
    """calculate_credit_ratios should handle empty DataFrame."""
    df = pd.DataFrame()

    ratios = credit.calculate_credit_ratios(df)

    assert len(ratios) == 0
    assert len(ratios.columns) == 0


@integration
def test_calculate_credit_ratios_preserves_index():
    """calculate_credit_ratios should preserve the original index."""
    idx = pd.date_range("2020-01-01", periods=3, freq="QE")
    df = pd.DataFrame(
        {
            "Total Assets": [1000, 1100, 1200],
            "Total Equity": [500, 550, 600],
        },
        index=idx,
    )

    ratios = credit.calculate_credit_ratios(df)

    assert list(ratios.index) == list(idx)
