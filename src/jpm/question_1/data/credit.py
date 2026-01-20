import numpy as np
import pandas as pd


def _safe_get(df, col_name, default=np.nan):
    """
    Safely retrieve a column from a DataFrame.

    Returns the column if it exists, otherwise returns a Series filled with
    the default value and matching the DataFrame's index.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to retrieve the column from.
    col_name : str
        The name of the column to retrieve.
    default : scalar, optional
        The default value to use if the column doesn't exist (default: np.nan).

    Returns
    -------
    pd.Series
        The requested column if it exists, or a Series filled with the default value.
    """
    return (
        df[col_name] if col_name in df.columns else pd.Series(default, index=df.index)
    )


def _safe_ratio(numerator, denominator):
    """
    Safely calculate a ratio with zero handling.

    Computes the ratio of numerator to denominator, replacing infinite values
    (resulting from division by zero or near-zero) with NaN.

    Parameters
    ----------
    numerator : pd.Series or scalar
        The numerator values for the ratio.
    denominator : pd.Series or scalar
        The denominator values for the ratio.

    Returns
    -------
    pd.Series or scalar
        The calculated ratio with infinite values replaced by NaN.
    """
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def _calculate_derived_values(df):
    """
    Calculate derived financial values from financial statement line items.

    Computes total debt, EBITDA, and free cash flow from their component
    values if available in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing financial statement line items.

    Returns
    -------
    total_debt : pd.Series or None
        Sum of short-term and long-term debt, or None if components unavailable.
    ebitda : pd.Series or None
        Earnings before interest, taxes, depreciation, and amortization
        (Operating Income + Depreciation and Amortization), or None if unavailable.
    free_cash_flow : pd.Series or None
        Operating cash flow minus capital expenditures, or None if unavailable.
    """
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
    """
    Calculate debt and leverage ratios.

    Computes various leverage metrics including debt-to-equity, debt-to-assets,
    and capital structure ratios. Updates the ratios_df DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing financial statement line items.
    ratios_df : pd.DataFrame
        DataFrame to store calculated ratios (modified in place).
    total_debt : pd.Series or None
        Total debt (short-term + long-term), or None if unavailable.

    Returns
    -------
    None
        Updates ratios_df in place with the following ratios (if calculable):
        - Debt_to_Equity
        - Debt_to_Assets
        - Debt_to_Tangible_Assets
        - Equity_to_Assets
        - Long_Term_Debt_to_Total_Debt
        - Total_Debt_to_Total_Capital
        - Long_Term_Debt_to_Equity
    """
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
    """
    Calculate interest and debt coverage ratios.

    Computes metrics that measure a company's ability to service its debt
    obligations. Updates the ratios_df DataFrame in place.

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
    free_cash_flow : pd.Series or None
        Operating cash flow minus capital expenditures, or None if unavailable.

    Returns
    -------
    None
        Updates ratios_df in place with the following ratios (if calculable):
        - Interest_Coverage (EBITDA / Interest Paid)
        - EBIT_to_Interest (Operating Income / Interest Paid)
        - Debt_to_EBITDA
        - Operating_CF_to_Debt
        - Free_CF_to_Debt
    """
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
    """
    Calculate profit margins and return ratios.

    Computes profitability metrics including margins, ROA, and ROE.
    Updates the ratios_df DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing financial statement line items.
    ratios_df : pd.DataFrame
        DataFrame to store calculated ratios (modified in place).
    ebitda : pd.Series or None
        Earnings before interest, taxes, depreciation, and amortization,
        or None if unavailable.

    Returns
    -------
    None
        Updates ratios_df in place with the following ratios (if calculable):
        - ROA (Return on Assets)
        - ROE (Return on Equity)
        - Operating_Margin
        - Net_Margin
        - Gross_Margin
        - EBITDA_Margin
        - Operating_Income_to_Assets
    """
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
    """
    Calculate current and quick ratios.

    Computes liquidity metrics that measure a company's ability to meet
    short-term obligations. Updates the ratios_df DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing financial statement line items.
    ratios_df : pd.DataFrame
        DataFrame to store calculated ratios (modified in place).

    Returns
    -------
    None
        Updates ratios_df in place with the following ratios (if calculable):
        - Current_Ratio
        - Working_Capital_to_Assets
        - Quick_Ratio
        - Cash_Ratio
        - Cash_to_Assets
    """
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
    """
    Calculate asset turnover and efficiency ratios.

    Computes metrics that measure how efficiently a company uses its assets
    to generate revenue. Updates the ratios_df DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing financial statement line items.
    ratios_df : pd.DataFrame
        DataFrame to store calculated ratios (modified in place).

    Returns
    -------
    None
        Updates ratios_df in place with the following ratios (if calculable):
        - Asset_Turnover
        - Receivables_Turnover
        - Inventory_Turnover
        - Fixed_Asset_Turnover
        - Cost_to_Income
    """
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
    if "Operating Expenses" in df.columns and "Total Revenues" in df.columns:
        ratios_df["Cost_to_Income"] = _safe_ratio(
            _safe_get(df, "Operating Expenses"), _safe_get(df, "Total Revenues")
        )


def _calculate_cash_flow_ratios(df, ratios_df, free_cash_flow):
    """
    Calculate cash flow related ratios.

    Computes metrics that measure cash generation and capital allocation
    efficiency. Updates the ratios_df DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing financial statement line items.
    ratios_df : pd.DataFrame
        DataFrame to store calculated ratios (modified in place).
    free_cash_flow : pd.Series or None
        Operating cash flow minus capital expenditures, or None if unavailable.

    Returns
    -------
    None
        Updates ratios_df in place with the following ratios (if calculable):
        - Operating_CF_to_Sales
        - Operating_CF_to_Current_Liabilities
        - Free_CF_to_Sales
        - Capex_to_Operating_CF
        - Capex_to_Sales
    """
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
    """
    Calculate equity and capitalization ratios.

    Computes metrics that measure the company's capital structure and
    solvency. Updates the ratios_df DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing financial statement line items.
    ratios_df : pd.DataFrame
        DataFrame to store calculated ratios (modified in place).

    Returns
    -------
    None
        Updates ratios_df in place with the following ratios (if calculable):
        - Equity_Multiplier
        - Liabilities_to_Assets
    """
    if "Total Assets" in df.columns and "Total Equity" in df.columns:
        ratios_df["Equity_Multiplier"] = _safe_ratio(
            _safe_get(df, "Total Assets"), _safe_get(df, "Total Equity")
        )

    if "Total Liabilities" in df.columns and "Total Assets" in df.columns:
        ratios_df["Liabilities_to_Assets"] = _safe_ratio(
            _safe_get(df, "Total Liabilities"), _safe_get(df, "Total Assets")
        )


def _calculate_dividend_ratios(df, ratios_df):
    """
    Calculate dividend payout and yield ratios.

    Computes metrics related to dividend payments and shareholder returns.
    Updates the ratios_df DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing financial statement line items.
    ratios_df : pd.DataFrame
        DataFrame to store calculated ratios (modified in place).

    Returns
    -------
    None
        Updates ratios_df in place with the following ratios (if calculable):
        - Dividend_Payout_Ratio
        - Retention_Ratio
        - Total_Payout_Ratio
    """
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
    """
    Calculate year-over-year growth rates.

    Computes metrics related to reinvestment and growth sustainability.
    Updates the ratios_df DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing financial statement line items.
    ratios_df : pd.DataFrame
        DataFrame to store calculated ratios (modified in place).

    Returns
    -------
    None
        Updates ratios_df in place with the following ratios (if calculable):
        - Reinvestment_Rate
        - RD_to_Sales (Research & Development to Sales)
        - Operating_Expense_Ratio
    """
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
    """
    Calculate earnings quality metrics.

    Computes metrics that assess the quality and sustainability of earnings,
    particularly the relationship between reported earnings and cash flows.
    Updates the ratios_df DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing financial statement line items.
    ratios_df : pd.DataFrame
        DataFrame to store calculated ratios (modified in place).

    Returns
    -------
    None
        Updates ratios_df in place with the following ratios (if calculable):
        - Operating_CF_to_Net_Income
        - Accruals_Ratio
    """
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
    """
    Calculate company size metrics.

    Computes log-transformed size metrics suitable for tree-based models
    like XGBoost. Updates the ratios_df DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing financial statement line items.
    ratios_df : pd.DataFrame
        DataFrame to store calculated ratios (modified in place).

    Returns
    -------
    None
        Updates ratios_df in place with the following ratios (if calculable):
        - Log_Total_Assets (log1p transformation)
        - Log_Total_Revenues (log1p transformation)
        - Log_Market_Cap_Proxy (log1p of Total Equity)
    """
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


def _calculate_jpm_ratios(df, ratios_df, total_debt, ebitda):
    """
    Calculate JPMorgan-specific ratios.

    Computes a specific set of financial ratios used for JPMorgan credit
    rating analysis. These ratios follow JPMorgan's naming conventions
    (lowercase with underscores). Updates the ratios_df DataFrame in place.

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
        ratios_df["Total_Debt"] = total_debt
    if ebitda is not None:
        ratios_df["EBITDA"] = ebitda
    if free_cash_flow is not None:
        ratios_df["Free_Cash_Flow"] = free_cash_flow

    # Ratios in bonus question
    _calculate_jpm_ratios(df, ratios_df, total_debt, ebitda)

    # Full ratios instead
    # _calculate_liquidity_ratios(df, ratios_df)
    # _calculate_leverage_ratios(df, ratios_df, total_debt)
    # _calculate_coverage_ratios(df, ratios_df, total_debt, ebitda, free_cash_flow)
    # _calculate_profitability_ratios(df, ratios_df, ebitda)
    # _calculate_efficiency_ratios(df, ratios_df)
    # _calculate_cash_flow_ratios(df, ratios_df, free_cash_flow)
    # _calculate_capital_structure_ratios(df, ratios_df)
    # _calculate_dividend_ratios(df, ratios_df)
    # _calculate_growth_ratios(df, ratios_df)
    # _calculate_quality_metrics(df, ratios_df)
    # _calculate_size_metrics(df, ratios_df)

    if "Total_Debt" in ratios_df.columns:
        ratios_df = ratios_df.drop(columns=["Total_Debt"])
    if "EBITDA" in ratios_df.columns:
        ratios_df = ratios_df.drop(columns=["EBITDA"])
    if "Free_Cash_Flow" in ratios_df.columns:
        ratios_df = ratios_df.drop(columns=["Free_Cash_Flow"])

    ratios_df = ratios_df.replace([np.inf, -np.inf], np.nan)

    return ratios_df
