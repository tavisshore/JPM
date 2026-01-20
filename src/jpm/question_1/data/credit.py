import numpy as np
import pandas as pd


def _safe_get(df, col_name, default=np.nan):
    return (
        df[col_name] if col_name in df.columns else pd.Series(default, index=df.index)
    )


def _safe_ratio(numerator, denominator):
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def _calculate_derived_values(df):
    # CALCULATE TOTAL DEBT (if components available)
    if "Short-term Debt" in df.columns or "Long-term Debt" in df.columns:
        total_debt = _safe_get(df, "Short-term Debt").fillna(0) + _safe_get(
            df, "Long-term Debt"
        ).fillna(0)
    else:
        total_debt = None

    # CALCULATE EBITDA (if components available)
    if "Operating Income" in df.columns:
        ebitda = _safe_get(df, "Operating Income") + _safe_get(
            df, "Depreciation and Amortization"
        ).fillna(0)
    else:
        ebitda = None

    # CALCULATE FREE CASH FLOW (if components available)
    if (
        "Net Cash from Operating Activities" in df.columns
        and "Capital Expenditures" in df.columns
    ):
        free_cash_flow = _safe_get(
            df, "Net Cash from Operating Activities"
        ) - _safe_get(df, "Capital Expenditures")
    else:
        free_cash_flow = None

    return total_debt, ebitda, free_cash_flow


def _calculate_jpm_ratios(df, ratios_df, total_debt, ebitda):
    """
    Calculate MLCOE question ratios.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing financial statement line items.
    ratios_df : pd.DataFrame
        DataFrame to store calculated ratios (modified in place).
    total_debt : pd.Series or None
        Total debt (short-term + long-term), or None if unavailable.
    ebitda : pd.Series or None
        Earnings before interest, taxes, depreciation, and amortization,
        or None if unavailable.

    Returns
    -------
    None
        Updates ratios_df in place with the following ratios (if calculable):
        - quick_ratio
        - debt_to_equity
        - debt_to_assets
        - debt_to_capital
        - debt_to_ebitda
        - ebit_to_interest
        - cost_to_income
    """
    # Quick Ratio
    if (
        "Total Current Assets" in df.columns
        and "Inventory" in df.columns
        and "Total Current Liabilities" in df.columns
    ):
        quick_assets = _safe_get(df, "Total Current Assets") - _safe_get(
            df, "Inventory"
        ).fillna(0)
        ratios_df["quick_ratio"] = _safe_ratio(
            quick_assets, _safe_get(df, "Total Current Liabilities")
        )
    # Debt to Equity
    if total_debt is not None and "Total Equity" in df.columns:
        ratios_df["debt_to_equity"] = _safe_ratio(
            total_debt, _safe_get(df, "Total Equity")
        )
    # Debt to Assets
    if total_debt is not None and "Total Assets" in df.columns:
        ratios_df["debt_to_assets"] = _safe_ratio(
            total_debt, _safe_get(df, "Total Assets")
        )
    # Debt to Capital
    if total_debt is not None and "Total Equity" in df.columns:
        total_capital = total_debt + _safe_get(df, "Total Equity")
        ratios_df["debt_to_capital"] = _safe_ratio(total_debt, total_capital)
    # Debt to EBITDA
    if total_debt is not None and ebitda is not None:
        ratios_df["debt_to_ebitda"] = _safe_ratio(total_debt, ebitda)
    # Interest Coverage
    if "Operating Income" in df.columns and "Interest Paid" in df.columns:
        ratios_df["ebit_to_interest"] = _safe_ratio(
            _safe_get(df, "Operating Income"), _safe_get(df, "Interest Paid")
        )
    if "Operating Expenses" in df.columns and "Total Revenues" in df.columns:
        ratios_df["cost_to_income"] = _safe_ratio(
            _safe_get(df, "Operating Expenses"), _safe_get(df, "Total Revenues")
        )


def calculate_credit_ratios(df):
    """
    Calculate financial ratios for credit rating \
        prediction from financial statement data.
    Handles missing features gracefully by skipping ratios that can't be calculated.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with financial statement features and ratings

    Returns:
    --------
    pd.DataFrame
        DataFrame with calculated ratios, metadata, and target variable
    """
    ratios_df = pd.DataFrame(index=df.index)

    # TARGET
    if "rating" in df.columns:
        ratios_df["rating"] = df["rating"]

    # Calculate derived values
    total_debt, ebitda, free_cash_flow = _calculate_derived_values(df)

    # Store derived values in ratios_df
    if total_debt is not None:
        ratios_df["total_debt"] = total_debt
    if ebitda is not None:
        ratios_df["ebitda"] = ebitda
    if free_cash_flow is not None:
        ratios_df["free_cash_flow"] = free_cash_flow
    # Ratios in bonus question
    _calculate_jpm_ratios(df, ratios_df, total_debt, ebitda)

    # TODO Add more later - this is a tiny subset for demonstration purposes

    if "total_debt" in ratios_df.columns:
        ratios_df = ratios_df.drop(columns=["total_debt"])
    if "ebitda" in ratios_df.columns:
        ratios_df = ratios_df.drop(columns=["ebitda"])
    if "free_cash_flow" in ratios_df.columns:
        ratios_df = ratios_df.drop(columns=["free_cash_flow"])
    ratios_df = ratios_df.replace([np.inf, -np.inf], np.nan)

    return ratios_df
