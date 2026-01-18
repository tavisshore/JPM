"""Tests for credit.py - Financial ratio calculations for credit rating prediction."""

import numpy as np
import pandas as pd
import pytest

from jpm.question_1.data import credit

unit = pytest.mark.unit
integration = pytest.mark.integration


# =============================================================================
# Tests for helper functions
# =============================================================================


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


# =============================================================================
# Tests for derived values calculation
# =============================================================================


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


# =============================================================================
# Tests for leverage ratios
# =============================================================================


@unit
def test_calculate_leverage_ratios_debt_to_equity():
    """_calculate_leverage_ratios should calculate Debt_to_Equity."""
    df = pd.DataFrame(
        {
            "Total Equity": [500, 1000],
        }
    )
    ratios_df = pd.DataFrame(index=df.index)
    total_debt = pd.Series([250, 500], index=df.index)

    credit._calculate_leverage_ratios(df, ratios_df, total_debt)

    assert "Debt_to_Equity" in ratios_df.columns
    assert ratios_df["Debt_to_Equity"].iloc[0] == 0.5
    assert ratios_df["Debt_to_Equity"].iloc[1] == 0.5


@unit
def test_calculate_leverage_ratios_debt_to_assets():
    """_calculate_leverage_ratios should calculate Debt_to_Assets."""
    df = pd.DataFrame(
        {
            "Total Assets": [1000, 2000],
        }
    )
    ratios_df = pd.DataFrame(index=df.index)
    total_debt = pd.Series([200, 500], index=df.index)

    credit._calculate_leverage_ratios(df, ratios_df, total_debt)

    assert "Debt_to_Assets" in ratios_df.columns
    assert ratios_df["Debt_to_Assets"].iloc[0] == 0.2
    assert ratios_df["Debt_to_Assets"].iloc[1] == 0.25


# =============================================================================
# Tests for profitability ratios
# =============================================================================


@unit
def test_calculate_profitability_ratios_roa():
    """_calculate_profitability_ratios should calculate ROA."""
    df = pd.DataFrame(
        {
            "Net Income": [100, 200],
            "Total Assets": [1000, 2000],
        }
    )
    ratios_df = pd.DataFrame(index=df.index)

    credit._calculate_profitability_ratios(df, ratios_df, ebitda=None)

    assert "ROA" in ratios_df.columns
    assert ratios_df["ROA"].iloc[0] == 0.1
    assert ratios_df["ROA"].iloc[1] == 0.1


@unit
def test_calculate_profitability_ratios_roe():
    """_calculate_profitability_ratios should calculate ROE."""
    df = pd.DataFrame(
        {
            "Net Income": [50, 100],
            "Total Equity": [500, 1000],
        }
    )
    ratios_df = pd.DataFrame(index=df.index)

    credit._calculate_profitability_ratios(df, ratios_df, ebitda=None)

    assert "ROE" in ratios_df.columns
    assert ratios_df["ROE"].iloc[0] == 0.1
    assert ratios_df["ROE"].iloc[1] == 0.1


@unit
def test_calculate_profitability_ratios_margins():
    """_calculate_profitability_ratios should calculate margin ratios."""
    df = pd.DataFrame(
        {
            "Operating Income": [100],
            "Net Income": [80],
            "Gross Profit": [300],
            "Total Revenues": [1000],
        }
    )
    ratios_df = pd.DataFrame(index=df.index)

    credit._calculate_profitability_ratios(df, ratios_df, ebitda=None)

    assert ratios_df["Operating_Margin"].iloc[0] == 0.1
    assert ratios_df["Net_Margin"].iloc[0] == 0.08
    assert ratios_df["Gross_Margin"].iloc[0] == 0.3


# =============================================================================
# Tests for liquidity ratios
# =============================================================================


@unit
def test_calculate_liquidity_ratios_current_ratio():
    """_calculate_liquidity_ratios should calculate Current_Ratio."""
    df = pd.DataFrame(
        {
            "Total Current Assets": [500, 600],
            "Total Current Liabilities": [250, 300],
        }
    )
    ratios_df = pd.DataFrame(index=df.index)

    credit._calculate_liquidity_ratios(df, ratios_df)

    assert "Current_Ratio" in ratios_df.columns
    assert ratios_df["Current_Ratio"].iloc[0] == 2.0
    assert ratios_df["Current_Ratio"].iloc[1] == 2.0


@unit
def test_calculate_liquidity_ratios_quick_ratio():
    """_calculate_liquidity_ratios should calculate Quick_Ratio."""
    df = pd.DataFrame(
        {
            "Total Current Assets": [500],
            "Inventory": [100],
            "Total Current Liabilities": [200],
        }
    )
    ratios_df = pd.DataFrame(index=df.index)

    credit._calculate_liquidity_ratios(df, ratios_df)

    assert "Quick_Ratio" in ratios_df.columns
    # (500 - 100) / 200 = 2.0
    assert ratios_df["Quick_Ratio"].iloc[0] == 2.0


@unit
def test_calculate_liquidity_ratios_cash_ratio():
    """_calculate_liquidity_ratios should calculate Cash_Ratio."""
    df = pd.DataFrame(
        {
            "Cash and Equivalents": [100],
            "Total Current Liabilities": [200],
        }
    )
    ratios_df = pd.DataFrame(index=df.index)

    credit._calculate_liquidity_ratios(df, ratios_df)

    assert "Cash_Ratio" in ratios_df.columns
    assert ratios_df["Cash_Ratio"].iloc[0] == 0.5


# =============================================================================
# Tests for efficiency ratios
# =============================================================================


@unit
def test_calculate_efficiency_ratios_asset_turnover():
    """_calculate_efficiency_ratios should calculate Asset_Turnover."""
    df = pd.DataFrame(
        {
            "Total Revenues": [1000],
            "Total Assets": [500],
        }
    )
    ratios_df = pd.DataFrame(index=df.index)

    credit._calculate_efficiency_ratios(df, ratios_df)

    assert "Asset_Turnover" in ratios_df.columns
    assert ratios_df["Asset_Turnover"].iloc[0] == 2.0


@unit
def test_calculate_efficiency_ratios_inventory_turnover():
    """_calculate_efficiency_ratios should calculate Inventory_Turnover."""
    df = pd.DataFrame(
        {
            "Total Cost of Revenue": [600],
            "Inventory": [100],
        }
    )
    ratios_df = pd.DataFrame(index=df.index)

    credit._calculate_efficiency_ratios(df, ratios_df)

    assert "Inventory_Turnover" in ratios_df.columns
    assert ratios_df["Inventory_Turnover"].iloc[0] == 6.0


# =============================================================================
# Tests for coverage ratios
# =============================================================================


@unit
def test_calculate_coverage_ratios_interest_coverage():
    """_calculate_coverage_ratios should calculate Interest_Coverage."""
    df = pd.DataFrame(
        {
            "Interest Paid": [50],
        }
    )
    ratios_df = pd.DataFrame(index=df.index)
    ebitda = pd.Series([200], index=df.index)

    credit._calculate_coverage_ratios(df, ratios_df, None, ebitda, None)

    assert "Interest_Coverage" in ratios_df.columns
    assert ratios_df["Interest_Coverage"].iloc[0] == 4.0


@unit
def test_calculate_coverage_ratios_debt_to_ebitda():
    """_calculate_coverage_ratios should calculate Debt_to_EBITDA."""
    df = pd.DataFrame(index=[0])
    ratios_df = pd.DataFrame(index=df.index)
    total_debt = pd.Series([600], index=df.index)
    ebitda = pd.Series([200], index=df.index)

    credit._calculate_coverage_ratios(df, ratios_df, total_debt, ebitda, None)

    assert "Debt_to_EBITDA" in ratios_df.columns
    assert ratios_df["Debt_to_EBITDA"].iloc[0] == 3.0


# =============================================================================
# Tests for size metrics
# =============================================================================


@unit
def test_calculate_size_metrics_log_total_assets():
    """_calculate_size_metrics should calculate Log_Total_Assets."""
    df = pd.DataFrame({"Total Assets": [1000000]})
    ratios_df = pd.DataFrame(index=df.index)

    credit._calculate_size_metrics(df, ratios_df)

    assert "Log_Total_Assets" in ratios_df.columns
    assert ratios_df["Log_Total_Assets"].iloc[0] == pytest.approx(np.log1p(1000000))


@unit
def test_calculate_size_metrics_handles_negative():
    """_calculate_size_metrics should handle edge cases."""
    df = pd.DataFrame({"Total Assets": [0]})
    ratios_df = pd.DataFrame(index=df.index)

    credit._calculate_size_metrics(df, ratios_df)

    # log1p(0) = 0
    assert ratios_df["Log_Total_Assets"].iloc[0] == 0.0


# =============================================================================
# Tests for main calculate_credit_ratios function
# =============================================================================


@integration
def test_calculate_credit_ratios_comprehensive():
    """calculate_credit_ratios should calculate all available ratios."""
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

    # Check derived values
    assert "Total_Debt" in ratios.columns
    assert ratios["Total_Debt"].iloc[0] == 250  # 50 + 200

    # Check leverage ratios
    assert "Debt_to_Equity" in ratios.columns
    assert "Debt_to_Assets" in ratios.columns

    # Check profitability ratios
    assert "ROA" in ratios.columns
    assert "ROE" in ratios.columns
    assert "Operating_Margin" in ratios.columns

    # Check liquidity ratios
    assert "Current_Ratio" in ratios.columns
    assert "Quick_Ratio" in ratios.columns

    # Check efficiency ratios
    assert "Asset_Turnover" in ratios.columns

    # Check size metrics
    assert "Log_Total_Assets" in ratios.columns


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

    # Should have some ratios calculated
    assert "Equity_to_Assets" in ratios.columns
    assert ratios["Equity_to_Assets"].iloc[0] == 0.5

    # Should not have ratios requiring missing columns
    assert "Current_Ratio" not in ratios.columns
    assert "Interest_Coverage" not in ratios.columns


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
    idx = pd.date_range("2020-01-01", periods=3, freq="Q")
    df = pd.DataFrame(
        {
            "Total Assets": [1000, 1100, 1200],
            "Total Equity": [500, 550, 600],
        },
        index=idx,
    )

    ratios = credit.calculate_credit_ratios(df)

    assert list(ratios.index) == list(idx)
