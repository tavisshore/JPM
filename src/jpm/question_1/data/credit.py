import numpy as np
import pandas as pd


def _safe_get(df, col_name, default=np.nan):
    """Safely get column from DataFrame, returning default if not present."""
    return (
        df[col_name] if col_name in df.columns else pd.Series(default, index=df.index)
    )


def _safe_ratio(numerator, denominator):
    """Calculate ratio, returning NaN where denominator is 0 or near-zero."""
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def _calculate_derived_values(df):
    """Calculate derived values: total_debt, ebitda, free_cash_flow."""
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


def _calculate_leverage_ratios(df, ratios_df, total_debt):
    """Calculate leverage ratios."""
    if total_debt is not None and "Total Equity" in df.columns:
        ratios_df["Debt_to_Equity"] = _safe_ratio(
            total_debt, _safe_get(df, "Total Equity")
        )

    if total_debt is not None and "Total Assets" in df.columns:
        ratios_df["Debt_to_Assets"] = _safe_ratio(
            total_debt, _safe_get(df, "Total Assets")
        )
        ratios_df["Debt_to_Tangible_Assets"] = _safe_ratio(
            total_debt,
            _safe_get(df, "Total Assets")
            - _safe_get(df, "Total Current Assets").fillna(0),
        )

    if "Total Equity" in df.columns and "Total Assets" in df.columns:
        ratios_df["Equity_to_Assets"] = _safe_ratio(
            _safe_get(df, "Total Equity"), _safe_get(df, "Total Assets")
        )

    if total_debt is not None and "Long-term Debt" in df.columns:
        ratios_df["Long_Term_Debt_to_Total_Debt"] = _safe_ratio(
            _safe_get(df, "Long-term Debt"), total_debt
        )

    if total_debt is not None and "Total Equity" in df.columns:
        total_capital = total_debt + _safe_get(df, "Total Equity")
        ratios_df["Total_Debt_to_Total_Capital"] = _safe_ratio(
            total_debt, total_capital
        )

    if "Long-term Debt" in df.columns and "Total Equity" in df.columns:
        ratios_df["Long_Term_Debt_to_Equity"] = _safe_ratio(
            _safe_get(df, "Long-term Debt"), _safe_get(df, "Total Equity")
        )


def _calculate_coverage_ratios(df, ratios_df, total_debt, ebitda, free_cash_flow):
    """Calculate coverage ratios."""
    if ebitda is not None and "Interest Paid" in df.columns:
        ratios_df["Interest_Coverage"] = _safe_ratio(
            ebitda, _safe_get(df, "Interest Paid")
        )

    if "Operating Income" in df.columns and "Interest Paid" in df.columns:
        ratios_df["EBIT_to_Interest"] = _safe_ratio(
            _safe_get(df, "Operating Income"), _safe_get(df, "Interest Paid")
        )

    if total_debt is not None and ebitda is not None:
        ratios_df["Debt_to_EBITDA"] = _safe_ratio(total_debt, ebitda)

    if total_debt is not None and "Net Cash from Operating Activities" in df.columns:
        ratios_df["Operating_CF_to_Debt"] = _safe_ratio(
            _safe_get(df, "Net Cash from Operating Activities"), total_debt
        )

    if total_debt is not None and free_cash_flow is not None:
        ratios_df["Free_CF_to_Debt"] = _safe_ratio(free_cash_flow, total_debt)


def _calculate_profitability_ratios(df, ratios_df, ebitda):
    """Calculate profitability ratios."""
    if "Net Income" in df.columns and "Total Assets" in df.columns:
        ratios_df["ROA"] = _safe_ratio(
            _safe_get(df, "Net Income"), _safe_get(df, "Total Assets")
        )

    if "Net Income" in df.columns and "Total Equity" in df.columns:
        ratios_df["ROE"] = _safe_ratio(
            _safe_get(df, "Net Income"), _safe_get(df, "Total Equity")
        )

    if "Operating Income" in df.columns and "Total Revenues" in df.columns:
        ratios_df["Operating_Margin"] = _safe_ratio(
            _safe_get(df, "Operating Income"), _safe_get(df, "Total Revenues")
        )

    if "Net Income" in df.columns and "Total Revenues" in df.columns:
        ratios_df["Net_Margin"] = _safe_ratio(
            _safe_get(df, "Net Income"), _safe_get(df, "Total Revenues")
        )

    if "Gross Profit" in df.columns and "Total Revenues" in df.columns:
        ratios_df["Gross_Margin"] = _safe_ratio(
            _safe_get(df, "Gross Profit"), _safe_get(df, "Total Revenues")
        )

    if ebitda is not None and "Total Revenues" in df.columns:
        ratios_df["EBITDA_Margin"] = _safe_ratio(
            ebitda, _safe_get(df, "Total Revenues")
        )

    if "Operating Income" in df.columns and "Total Assets" in df.columns:
        ratios_df["Operating_Income_to_Assets"] = _safe_ratio(
            _safe_get(df, "Operating Income"), _safe_get(df, "Total Assets")
        )


def _calculate_liquidity_ratios(df, ratios_df):
    """Calculate liquidity ratios."""
    if (
        "Total Current Assets" in df.columns
        and "Total Current Liabilities" in df.columns
    ):
        ratios_df["Current_Ratio"] = _safe_ratio(
            _safe_get(df, "Total Current Assets"),
            _safe_get(df, "Total Current Liabilities"),
        )

        working_capital = _safe_get(df, "Total Current Assets") - _safe_get(
            df, "Total Current Liabilities"
        )
        if "Total Assets" in df.columns:
            ratios_df["Working_Capital_to_Assets"] = _safe_ratio(
                working_capital, _safe_get(df, "Total Assets")
            )

    if (
        "Total Current Assets" in df.columns
        and "Inventory" in df.columns
        and "Total Current Liabilities" in df.columns
    ):
        quick_assets = _safe_get(df, "Total Current Assets") - _safe_get(
            df, "Inventory"
        ).fillna(0)
        ratios_df["Quick_Ratio"] = _safe_ratio(
            quick_assets, _safe_get(df, "Total Current Liabilities")
        )

    if (
        "Cash and Equivalents" in df.columns
        and "Total Current Liabilities" in df.columns
    ):
        ratios_df["Cash_Ratio"] = _safe_ratio(
            _safe_get(df, "Cash and Equivalents"),
            _safe_get(df, "Total Current Liabilities"),
        )

    if "Cash and Equivalents" in df.columns and "Total Assets" in df.columns:
        ratios_df["Cash_to_Assets"] = _safe_ratio(
            _safe_get(df, "Cash and Equivalents"), _safe_get(df, "Total Assets")
        )


def _calculate_efficiency_ratios(df, ratios_df):
    """Calculate efficiency ratios."""
    if "Total Revenues" in df.columns and "Total Assets" in df.columns:
        ratios_df["Asset_Turnover"] = _safe_ratio(
            _safe_get(df, "Total Revenues"), _safe_get(df, "Total Assets")
        )

    if "Total Revenues" in df.columns and "Receivables" in df.columns:
        ratios_df["Receivables_Turnover"] = _safe_ratio(
            _safe_get(df, "Total Revenues"), _safe_get(df, "Receivables")
        )

    if "Total Cost of Revenue" in df.columns and "Inventory" in df.columns:
        ratios_df["Inventory_Turnover"] = _safe_ratio(
            _safe_get(df, "Total Cost of Revenue"), _safe_get(df, "Inventory")
        )

    if (
        "Total Revenues" in df.columns
        and "Property, Plant, and Equipment (net)" in df.columns
    ):
        ratios_df["Fixed_Asset_Turnover"] = _safe_ratio(
            _safe_get(df, "Total Revenues"),
            _safe_get(df, "Property, Plant, and Equipment (net)"),
        )


def _calculate_cash_flow_ratios(df, ratios_df, free_cash_flow):
    """Calculate cash flow ratios."""
    if (
        "Net Cash from Operating Activities" in df.columns
        and "Total Revenues" in df.columns
    ):
        ratios_df["Operating_CF_to_Sales"] = _safe_ratio(
            _safe_get(df, "Net Cash from Operating Activities"),
            _safe_get(df, "Total Revenues"),
        )

    if (
        "Net Cash from Operating Activities" in df.columns
        and "Total Current Liabilities" in df.columns
    ):
        ratios_df["Operating_CF_to_Current_Liabilities"] = _safe_ratio(
            _safe_get(df, "Net Cash from Operating Activities"),
            _safe_get(df, "Total Current Liabilities"),
        )

    if free_cash_flow is not None and "Total Revenues" in df.columns:
        ratios_df["Free_CF_to_Sales"] = _safe_ratio(
            free_cash_flow, _safe_get(df, "Total Revenues")
        )

    if (
        "Capital Expenditures" in df.columns
        and "Net Cash from Operating Activities" in df.columns
    ):
        ratios_df["Capex_to_Operating_CF"] = _safe_ratio(
            _safe_get(df, "Capital Expenditures"),
            _safe_get(df, "Net Cash from Operating Activities"),
        )

    if "Capital Expenditures" in df.columns and "Total Revenues" in df.columns:
        ratios_df["Capex_to_Sales"] = _safe_ratio(
            _safe_get(df, "Capital Expenditures"), _safe_get(df, "Total Revenues")
        )


def _calculate_capital_structure_ratios(df, ratios_df):
    """Calculate capital structure and solvency ratios."""
    if "Total Assets" in df.columns and "Total Equity" in df.columns:
        ratios_df["Equity_Multiplier"] = _safe_ratio(
            _safe_get(df, "Total Assets"), _safe_get(df, "Total Equity")
        )

    if "Total Liabilities" in df.columns and "Total Assets" in df.columns:
        ratios_df["Liabilities_to_Assets"] = _safe_ratio(
            _safe_get(df, "Total Liabilities"), _safe_get(df, "Total Assets")
        )


def _calculate_dividend_ratios(df, ratios_df):
    """Calculate dividend and shareholder metrics."""
    if "Dividends Paid" in df.columns and "Net Income" in df.columns:
        ratios_df["Dividend_Payout_Ratio"] = _safe_ratio(
            _safe_get(df, "Dividends Paid"), _safe_get(df, "Net Income")
        )
        ratios_df["Retention_Ratio"] = 1 - _safe_ratio(
            _safe_get(df, "Dividends Paid"), _safe_get(df, "Net Income")
        )

    if (
        "Dividends Paid" in df.columns
        and "Stock Repurchases" in df.columns
        and "Net Income" in df.columns
    ):
        total_payout = _safe_get(df, "Dividends Paid") + _safe_get(
            df, "Stock Repurchases"
        ).fillna(0)
        ratios_df["Total_Payout_Ratio"] = _safe_ratio(
            total_payout, _safe_get(df, "Net Income")
        )


def _calculate_growth_ratios(df, ratios_df):
    """Calculate growth and sustainability indicators."""
    if (
        "Capital Expenditures" in df.columns
        and "Net Cash from Operating Activities" in df.columns
    ):
        reinvestment = _safe_get(df, "Capital Expenditures") + _safe_get(
            df, "Acquisitions (net of cash acquired)"
        ).fillna(0)
        ratios_df["Reinvestment_Rate"] = _safe_ratio(
            reinvestment, _safe_get(df, "Net Cash from Operating Activities")
        )

    if "Research and Development" in df.columns and "Total Revenues" in df.columns:
        ratios_df["RD_to_Sales"] = _safe_ratio(
            _safe_get(df, "Research and Development"), _safe_get(df, "Total Revenues")
        )

    if "Total Operating Expenses" in df.columns and "Total Revenues" in df.columns:
        ratios_df["Operating_Expense_Ratio"] = _safe_ratio(
            _safe_get(df, "Total Operating Expenses"), _safe_get(df, "Total Revenues")
        )


def _calculate_quality_metrics(df, ratios_df):
    """Calculate quality metrics."""
    if (
        "Net Cash from Operating Activities" in df.columns
        and "Net Income" in df.columns
    ):
        ratios_df["Operating_CF_to_Net_Income"] = _safe_ratio(
            _safe_get(df, "Net Cash from Operating Activities"),
            _safe_get(df, "Net Income"),
        )

        if "Total Assets" in df.columns:
            accruals = _safe_get(df, "Net Income") - _safe_get(
                df, "Net Cash from Operating Activities"
            )
            ratios_df["Accruals_Ratio"] = _safe_ratio(
                accruals, _safe_get(df, "Total Assets")
            )


def _calculate_size_metrics(df, ratios_df):
    """Calculate size metrics (log scale for XGBoost)."""
    if "Total Assets" in df.columns:
        ratios_df["Log_Total_Assets"] = np.log1p(
            _safe_get(df, "Total Assets").fillna(0)
        )

    if "Total Revenues" in df.columns:
        ratios_df["Log_Total_Revenues"] = np.log1p(
            _safe_get(df, "Total Revenues").fillna(0)
        )

    if "Total Equity" in df.columns:
        ratios_df["Log_Market_Cap_Proxy"] = np.log1p(
            _safe_get(df, "Total Equity").fillna(0)
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
        ratios_df["Total_Debt"] = total_debt
    if ebitda is not None:
        ratios_df["EBITDA"] = ebitda
    if free_cash_flow is not None:
        ratios_df["Free_Cash_Flow"] = free_cash_flow

    # Calculate all ratio categories
    _calculate_leverage_ratios(df, ratios_df, total_debt)
    _calculate_coverage_ratios(df, ratios_df, total_debt, ebitda, free_cash_flow)
    _calculate_profitability_ratios(df, ratios_df, ebitda)
    _calculate_liquidity_ratios(df, ratios_df)
    _calculate_efficiency_ratios(df, ratios_df)
    _calculate_cash_flow_ratios(df, ratios_df, free_cash_flow)
    _calculate_capital_structure_ratios(df, ratios_df)
    _calculate_dividend_ratios(df, ratios_df)
    _calculate_growth_ratios(df, ratios_df)
    _calculate_quality_metrics(df, ratios_df)
    _calculate_size_metrics(df, ratios_df)

    # REPLACE INFINITIES WITH NaN (safety check)
    ratios_df = ratios_df.replace([np.inf, -np.inf], np.nan)

    return ratios_df


# Usage example:

#
# # Prepare for XGBoost
# feature_cols = [col for col in ratios_df.columns
#                 if col not in ['ticker', 'obligor_name', 'rating']]
# X = ratios_df[feature_cols]
# y = ratios_df['rating']
