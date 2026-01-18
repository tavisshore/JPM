import json
import re
from typing import List, Optional, Tuple, TypedDict

import numpy as np
import pandas as pd
import requests

# LLMClient and LLMConfig imported lazily in leis_to_ticker to avoid circular import


class AssetSection(TypedDict):
    current_assets: List[str]
    non_current_assets: List[str]


class LiabilitySection(TypedDict):
    current_liabilities: List[str]
    non_current_liabilities: List[str]


class BalanceSheetStructure(TypedDict):
    assets: AssetSection
    liabilities: LiabilitySection
    equity: List[str]


def xbrl_to_snake(name: str) -> str:
    # Drop common XBRL prefixes (e.g., us-gaap_) and normalize to snake_case
    if not isinstance(name, str) or not name:
        raise ValueError("xbrl_to_snake expects a non-empty string")
    name = re.sub(r"^[^-]+-[^_]+_", "", name)
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", name)
    return s.lower()


def xbrl_to_raw(name: str) -> str:
    """
    Remove XBRL prefixes and convert to snake_case.

    Handles patterns like:
    - us-gaap_PropertyPlantAndEquipment -> property plant and equipment
    - amzn_LeaseLiabilityNoncurrent -> lease liability noncurrent
    - dei_EntityRegistrantName -> entity registrant name

    Args:
        name: XBRL field name with prefix

    Returns:
        Normalized snake_case string without prefix
    """
    if not isinstance(name, str) or not name:
        raise ValueError("xbrl_to_raw expects a non-empty string")

    # Remove XBRL namespace prefixes (e.g., us-gaap_, amzn_, dei_)
    # Matches: <prefix>-<anything>_ OR <prefix>_
    # Examples: us-gaap_, amzn_, dei_, ifrs-full_
    name = re.sub(r"^[a-z0-9]+-[^_]+_", "", name)  # Handles us-gaap_, ifrs-full_
    name = re.sub(r"^[a-z0-9]+_", "", name)  # Handles amzn_, dei_, etc.

    # Convert CamelCase to space-separated
    s = re.sub(r"(?<!^)(?=[A-Z])", " ", name)

    return s.lower()


def _is_consecutive_quarters(index, start: int, length: int) -> bool:
    """
    Check if `length` periods starting at `start` are consecutive quarters.
    Handles both ascending (oldest first) and descending (newest first) order.

    Parameters:
    -----------
    index : pd.PeriodIndex or array-like
        The index of the data (must support period arithmetic)
    start : int
        Starting position in the index
    length : int
        Number of periods to check

    Returns:
    --------
    bool
        True if all periods are consecutive quarters
    """
    if index is None:
        return True  # No index provided, assume consecutive

    if length < 2:
        return True

    # Use ordinal values for comparison (works with PeriodIndex)
    first_ord = index[start].ordinal
    second_ord = index[start + 1].ordinal
    expected_diff = second_ord - first_ord  # +1 for ascending, -1 for descending

    if abs(expected_diff) != 1:
        return False  # First two aren't consecutive

    # Check remaining periods maintain same direction
    for i in range(1, length - 1):
        current_ord = index[start + i].ordinal
        next_ord = index[start + i + 1].ordinal
        if next_ord - current_ord != expected_diff:
            return False
    return True


def build_windows(
    config,
    X: np.ndarray,
    tgt_indices: Optional[List[int]] = None,
    index=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding windows over X and split the *last `withhold_periods`
    targets* into a test set. Only includes windows where all periods
    in the window (lookback + horizon) are consecutive quarters.

    Parameters:
    -----------
    X : np.ndarray
        2D array of shape (time, features)
    lookback : int
        Number of historical periods in each window
    horizon : int
        Number of periods ahead to predict
    tgt_indices : Optional[List[int]]
        Indices of target features to predict (None = all features)
    withhold_periods : int
        Number of windows to withhold_periods for test set
    index : pd.PeriodIndex, optional
        Period index of the data. If provided, only windows with
        consecutive quarters will be included.

    Returns:
        X_train: (N_train, lookback, F)
        y_train: (N_train, target_dim)
        X_test:  (N_test, lookback, F)   where N_test == withhold_periods
        y_test:  (N_test, target_dim)
    """
    _validate_window_args(
        config,
        X,
        tgt_indices,
    )
    T, F = X.shape
    max_start = _max_start(T, config.data.lookback, config.data.horizon)

    X_train, y_train, X_test, y_test = _build_window_arrays_with_index(
        config,
        X,
        tgt_indices,
        max_start,
        F,
        index,
    )
    return X_train, y_train, X_test, y_test


def _validate_window_args(
    config,
    X: np.ndarray,
    tgt_indices: Optional[List[int]],
) -> None:
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy ndarray")
    if X.ndim != 2:
        raise ValueError(f"X must be 2D array of shape (time, features); got {X.shape}")
    if config.data.lookback <= 0:
        raise ValueError("lookback must be positive")
    if config.data.horizon <= 0:
        raise ValueError("horizon must be positive")
    if tgt_indices is not None and any(i < 0 or i >= X.shape[1] for i in tgt_indices):
        raise IndexError("tgt_indices contain out-of-bounds indices for X features")
    if not isinstance(config.data.withhold_periods, int):
        raise TypeError("withhold_periods must be an integer")
    if config.data.withhold_periods < 0:
        raise ValueError("withhold_periods must be >= 0")

    if config.data.lookback + config.data.horizon > X.shape[0]:
        raise ValueError("Sequence too short for given lookback and horizon")


def _max_start(T: int, lookback: int, horizon: int) -> int:
    max_start = T - lookback - horizon + 1
    if max_start <= 0:
        raise ValueError("Sequence too short for given lookback and horizon")
    return max_start


def _build_window_arrays_with_index(
    config,
    X: np.ndarray,
    tgt_indices: Optional[List[int]],
    max_start: int,
    num_features: int,
    index=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build window arrays, only including windows with consecutive quarters.
    The last `withhold_periods` valid windows are used for testing.
    """
    all_windows = []
    window_length = config.data.lookback + config.data.horizon

    # Collect all valid windows (those with consecutive quarters)
    for t in range(max_start):
        if _is_consecutive_quarters(index, t, window_length):
            x_win, y_target = _extract_window(
                X, t, config.data.lookback, config.data.horizon, tgt_indices
            )
            all_windows.append((x_win, y_target))

    if len(all_windows) == 0:
        target_dim = len(tgt_indices) if tgt_indices else X.shape[1]
        return (
            np.empty((0, config.data.lookback, num_features)),
            np.empty((0, target_dim)),
            np.empty((0, config.data.lookback, num_features)),
            np.empty((0, target_dim)),
        )

    if config.data.withhold_periods > len(all_windows):
        raise ValueError(
            f"withhold_periods={config.data.withhold_periods} is too large;"
            f"only {len(all_windows)} valid consecutive windows available"
        )

    # Split into train/test - last `withhold_periods` windows go to test
    split_idx = len(all_windows) - config.data.withhold_periods

    X_train = [w[0] for w in all_windows[:split_idx]]
    y_train = [w[1] for w in all_windows[:split_idx]]
    X_test = [w[0] for w in all_windows[split_idx:]]
    y_test = [w[1] for w in all_windows[split_idx:]]

    target_dim = all_windows[0][1].shape[-1] if all_windows[0][1].ndim > 0 else 1

    X_train_arr = (
        np.stack(X_train)
        if X_train
        else np.empty((0, config.data.lookback, num_features))
    )
    y_train_arr = np.stack(y_train) if y_train else np.empty((0, target_dim))
    X_test_arr = (
        np.stack(X_test)
        if X_test
        else np.empty((0, config.data.lookback, num_features))
    )
    y_test_arr = np.stack(y_test) if y_test else np.empty((0, target_dim))

    return X_train_arr, y_train_arr, X_test_arr, y_test_arr


def _extract_window(
    X: np.ndarray,
    start: int,
    lookback: int,
    horizon: int,
    tgt_indices: Optional[List[int]],
) -> tuple[np.ndarray, np.ndarray]:
    x_win = X[start : start + lookback]
    y_target = X[start + lookback + horizon - 1]
    if tgt_indices is not None:
        y_target = y_target[tgt_indices]
    return x_win, y_target


def detect_duplicate_columns(df, tolerance=1e-6, similarity_threshold=0.999):
    """
    Detect columns that are duplicates or derived from other columns.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe to analyze
    tolerance : float
        Numerical tolerance for considering values equal (default 1e-6)
    similarity_threshold : float
        Threshold for considering columns identical (0-1, default 0.9)

    Returns:
    --------
    dict
        Dictionary with duplicate groups and analysis
    """

    results = {"duplicate_groups": [], "columns_to_drop": [], "summary": {}}

    columns = df.columns.tolist()
    checked = set()

    for i, col1 in enumerate(columns):
        if col1 in checked:
            continue

        duplicates = [col1]

        for _, col2 in enumerate(columns[i + 1 :], start=i + 1):
            if col2 in checked:
                continue

            # Check if columns are identical
            if are_columns_duplicate(
                df[col1], df[col2], tolerance, similarity_threshold
            ):
                duplicates.append(col2)
                checked.add(col2)

        # Hard-coded exceptions for key totals
        if "liabilities and stockholders equity" in duplicates:
            duplicates.remove("liabilities and stockholders equity")
            # results['']
        if "assets" in duplicates:
            duplicates.remove("assets")

        if len(duplicates) > 1:
            # Found duplicate group
            results["duplicate_groups"].append(duplicates)

            # Decide which to keep
            column_to_keep = select_column_to_keep(duplicates)
            columns_to_drop = [col for col in duplicates if col != column_to_keep]

            results["columns_to_drop"].extend(columns_to_drop)

    results["summary"] = {
        "total_columns": len(columns),
        "duplicate_groups_found": len(results["duplicate_groups"]),
        "columns_to_drop": len(results["columns_to_drop"]),
        "columns_remaining": len(columns) - len(results["columns_to_drop"]),
    }

    return results


def are_columns_duplicate(col1, col2, tolerance=1e-6, similarity_threshold=0.999):
    """
    Check if two columns are duplicates within tolerance.

    Parameters:
    -----------
    col1, col2 : pd.Series
        Columns to compare
    tolerance : float
        Numerical tolerance
    similarity_threshold : float
        Minimum fraction of matching rows (0-1)

    Returns:
    --------
    bool
        True if columns are considered duplicates
    """

    # Handle NaN values - both NaN is considered a match
    mask_both_nan = col1.isna() & col2.isna()
    mask_both_not_nan = (~col1.isna()) & (~col2.isna())

    # Check numerical similarity for non-NaN values
    if mask_both_not_nan.any():
        try:
            diff = np.abs(col1[mask_both_not_nan] - col2[mask_both_not_nan])
            matches_numeric = (diff <= tolerance).sum()
        except Exception:
            print(col1)
            print(col2)
            breakpoint()
    else:
        matches_numeric = 0

    # Total matches (both NaN + numerical matches)
    total_matches = mask_both_nan.sum() + matches_numeric
    total_rows = len(col1)

    # Calculate similarity ratio
    similarity_ratio = total_matches / total_rows if total_rows > 0 else 0

    return similarity_ratio >= similarity_threshold


def select_column_to_keep(duplicate_columns):
    """
    Select which column to keep from a group of duplicates.
    Prefer to drop columns with 'total' in the name.

    Parameters:
    -----------
    duplicate_columns : list
        List of duplicate column names

    Returns:
    --------
    str
        Column name to keep
    """

    # Keywords that suggest derived/aggregate columns (prefer to drop these)
    drop_keywords = ["total", "sum", "aggregate", "combined", "consolidated"]

    # Score each column (lower score = more likely to drop)
    scores = []
    for col in duplicate_columns:
        col_lower = col.lower()

        # Start with base score
        score = 100

        # Penalize if contains drop keywords
        for keyword in drop_keywords:
            if keyword in col_lower:
                score -= 50

        # Penalize longer names (often more descriptive/derived)
        score -= len(col) * 0.1

        scores.append(score)

    # Keep the column with highest score
    max_score_idx = scores.index(max(scores))
    return duplicate_columns[max_score_idx]


def remove_duplicate_columns(
    df, tolerance=1e-6, similarity_threshold=0.999, verbose=True
):
    """
    Remove duplicate columns from dataframe.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    tolerance : float
        Numerical tolerance for considering values equal
    similarity_threshold : float
        Threshold for considering columns identical (0-1)
    verbose : bool
        Whether to print detailed information

    Returns:
    --------
    tuple
        (cleaned_df, results_dict)
    """

    results = detect_duplicate_columns(df, tolerance, similarity_threshold)

    # Drop duplicate columns
    cleaned_df = df.drop(columns=results["columns_to_drop"])

    return cleaned_df


def bs_identity_checker(df: pd.DataFrame) -> pd.DataFrame:
    df = df.fillna(0)

    # Assets
    assets = (
        df["Cash and Equivalents"]
        + df["Receivables"]
        + df["Inventory"]
        + df["Prepaid and Other Current Assets"]
        + df["Marketable Securities (Short-term)"]
        + df["Property, Plant, and Equipment (net)"]
        + df["Intangible Assets (net)"]
        + df["Goodwill"]
        + df["Long-term Investments"]
        + df["Deferred Tax Assets"]
        + df["Other Non-Current Assets"]
    )

    # Liabilities
    liabilities = (
        df["Accounts Payable and Accrued Expenses"]
        + df["Short-term Debt"]
        + df["Deferred Revenue"]
        + df["Other Current Liabilities"]
        + df["Long-term Debt"]
        + df["Deferred Tax Liabilities"]
        + df["Lease Liabilities"]
        + df["Other Non-Current Liabilities"]
    )

    equity = (
        df["Common Stock and APIC"]
        + df["Retained Earnings"]
        - df["Treasury Stock"]
        + df["Accumulated Other Comprehensive Income"]
    )

    accounting_id = assets - (liabilities + equity)

    if accounting_id.abs().max() > 1e3:
        # Get the windows where discrepancy occurs - remove these
        discrepant_indices = accounting_id[accounting_id.abs() > 1e3].index
        df = df.drop(index=discrepant_indices)

    return df


def drop_constants(df, verbose=False):
    # Drop columns that are all NaN first (avoids warnings from nanstd)
    all_nan_cols = df.columns[df.isna().all()].tolist()
    if all_nan_cols:
        if verbose:
            print(f"Dropping {len(all_nan_cols)} all-NaN columns: {all_nan_cols}")
        df = df.drop(columns=all_nan_cols)

    # Check for constant columns, treating NaN as missing (not as a value)
    stds = df.apply(lambda x: np.nanstd(x.astype(float)))
    constant_cols = stds[stds == 0].index.tolist()
    if constant_cols:
        if verbose:
            print(f"Dropping {len(constant_cols)} constant columns: {constant_cols}")
        df = df.drop(columns=constant_cols)

    # Drop rows that are mostly NaN (should be rare after inner join)
    df = df.dropna(axis=0, thresh=int(0.8 * len(df.columns)))

    # If Income Before Taxes in columns, drop rows where it's 0
    if "Income Before Taxes" in df.columns:
        df = df[df["Income Before Taxes"].fillna(0) != 0]

    # NOTE: fillna(0) moved to after all variance checks in _load_or_fetch_data
    return df


def ytd_to_quarterly(df):
    """
    Convert YTD to quarterly.
    Expects: rows = PeriodIndex (quarterly), columns = metrics
    """
    df = df.copy()
    df = df.sort_index()

    quarterly = pd.DataFrame(index=df.index, columns=df.columns)

    for i, idx in enumerate(df.index):
        # Q1 of fiscal year (quarter 1)
        is_q1 = idx.quarter == 1

        if i == 0 or is_q1:
            quarterly.loc[idx] = df.loc[idx]
        else:
            quarterly.loc[idx] = df.loc[idx] - df.loc[df.index[i - 1]]

    return quarterly


def lei_to_ticker(lei):
    if pd.isna(lei):
        return None

    try:
        # Get company registration info from GLEIF
        gleif = requests.get(
            f"https://api.gleif.org/api/v1/lei-records/{lei}", timeout=10
        )
        gleif.raise_for_status()
        gleif_data = gleif.json()["data"]["attributes"]

        # Get registration details for OpenCorporates
        registration = gleif_data.get("registration", {})
        jurisdiction = registration.get("registrationAuthorityID", "")
        company_number = registration.get("registrationAuthorityEntityID", "")

        # Try OpenCorporates with registration number
        if jurisdiction and company_number:
            oc = requests.get(
                f"https://api.opencorporates.com/v0.4/companies/{jurisdiction}/{company_number}",
                timeout=10,
            )
            if oc.status_code == 200:
                name = oc.json()["results"]["company"]["name"]
            else:
                # Fallback to GLEIF name
                name = gleif_data["entity"]["legalName"]["name"]
        else:
            name = gleif_data["entity"]["legalName"]["name"]

        # Check if name contains non-Latin characters
        if any(ord(char) > 591 for char in name):
            return "non-english"

        figi = requests.post(
            "https://api.openfigi.com/v3/search",
            json={
                "query": name,
                "securityType": "Common Stock",
                "exchCode": "US",
            },
            timeout=10,
        )
        figi.raise_for_status()
        data = figi.json().get("data", [])

        if data:
            return data[0].get("ticker")
    except (requests.RequestException, KeyError, IndexError):
        pass

    return None


def leis_to_ticker(names, ticker_data, llm_client):
    from jpm.question_1.clients.llm_client import LLMConfig

    # if os.path.exists(ticker_data):
    #     with open(ticker_data, "r") as f:
    #         lei_ticker_map = json.load(f)
    # else:

    llm_config = LLMConfig(
        provider="openai",
        model="gpt-5-nano",
        temperature=0.0,
        max_tokens=8192,
        use_llm=False,
        adjust=False,
    )

    # Use LLM
    llm_mapping = llm_client.company_name_to_ticker(names, llm_config, llm_client)

    with open(ticker_data, "w") as f:
        json.dump(llm_mapping, f, indent=2)

    return llm_mapping


def ticker_to_name(names, llm_client):
    from jpm.question_1.clients.llm_client import LLMConfig

    llm_config = LLMConfig(
        provider="openai",
        model="gpt-5-mini",
        temperature=0.0,
        max_tokens=8192,
        use_llm=False,
        adjust=False,
    )

    llm_mapping = llm_client.ticker_to_company_name(names, llm_config)

    return llm_mapping


def add_derived_columns(df):
    d = df.copy()

    def col(name):
        return d[name].fillna(0) if name in d.columns else pd.Series(0, index=d.index)

    # Assets
    d["Total Current Assets"] = (
        col("Cash and Cash Equivalents")
        + col("Accounts Receivable")
        + col("Inventory")
        + col("Prepaid Expenses")
        + col("Marketable Securities (Short-term Investments)")
        + col("Other Current Assets")
    )

    d["Total Non-Current Assets"] = (
        col("Property, Plant, and Equipment (PP&E)")
        + col("Intangible Assets")
        + col("Goodwill")
        + col("Marketable Securities (Non-current)")
        + col("Long-term Investments")
        + col("Deferred Tax Assets")
        + col("Other Non-Current Assets")
    )

    d["Total Assets"] = d["Total Current Assets"] + d["Total Non-Current Assets"]

    # Liabilities
    d["Total Current Liabilities"] = (
        col("Accounts Payable")
        + col("Accrued Expenses")
        + col("Short-term Debt")
        + col("Current Portion of Long-term Debt")
        + col("Unearned Revenue (Deferred Revenue)")
        + col("Income Taxes Payable")
        + col("Other Current Liabilities")
    )

    d["Total Non-Current Liabilities"] = (
        col("Long-term Debt")
        + col("Deferred Tax Liabilities")
        # + col("Pension Liabilities")
        + col("Lease Liabilities")
        + col("Other Non-Current Liabilities")
    )

    d["Total Liabilities"] = (
        d["Total Current Liabilities"] + d["Total Non-Current Liabilities"]
    )

    # Equity
    d["Total Equity"] = (
        col("Additional Paid-in Capital")
        + col("Retained Earnings")
        + col("Treasury Stock")
        + col("Accumulated Other Comprehensive Income (AOCI)")
    )

    # Income Statement
    d["Total Cost of Revenue"] = (
        col("Cost of Goods Sold")
        + col("Cost of Services")
        + col("Other Cost of Revenue")
    )
    d["Gross Profit"] = col("Total Revenues") - d["Total Cost of Revenue"]
    d["Total Operating Expenses"] = (
        col("Selling, General and Administrative")
        + col("Research and Development")
        + col("Other Operating Expenses")
    )
    d["Operating Income (Loss)"] = d["Gross Profit"] - d["Total Operating Expenses"]

    # Cash Flow
    d["Net Cash Provided by (Used in) Operating Activities"] = (
        col("Net Income (Loss)")
        + col("Stock-Based Compensation")
        + col("Changes in Working Capital")
        + col("Other Operating Activities")
    )

    return d


def standardise_rating(rating):
    if pd.isna(rating):
        return None
    return rating.replace("(P)", "").strip()


def drop_non_numeric_columns(df):
    """Drop columns that contain non-numeric, non-NaN values"""
    numeric_cols = []

    for col in df.columns:
        # Convert to numeric, coercing errors to NaN
        converted = pd.to_numeric(df[col], errors="coerce")

        # Check if all values are either numeric or NaN (no strings/objects left)
        if converted.notna().sum() == df[col].notna().sum():
            numeric_cols.append(col)

    return df[numeric_cols]
