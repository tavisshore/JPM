import json
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple, TypedDict

import numpy as np
import pandas as pd
import requests

# LLMClient and LLMConfig imported lazily in leis_to_ticker to avoid circular import


class AssetSection(TypedDict):
    """Balance sheet asset section structure with current and non-current assets."""

    current_assets: List[str]
    non_current_assets: List[str]


class LiabilitySection(TypedDict):
    """Balance sheet liability section structure with current and non-current liabilities."""

    current_liabilities: List[str]
    non_current_liabilities: List[str]


class BalanceSheetStructure(TypedDict):
    """Complete balance sheet structure with assets, liabilities, and equity sections."""

    assets: AssetSection
    liabilities: LiabilitySection
    equity: List[str]


def xbrl_to_snake(name: str) -> str:
    """
    Convert XBRL field names to snake_case by removing prefixes and normalizing.

    Parameters
    ----------
    name : str
        XBRL field name with prefix (e.g., 'us-gaap_PropertyPlantAndEquipment')

    Returns
    -------
    str
        Normalized snake_case string (e.g., 'property_plant_and_equipment')
    """
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
    """
    Validate arguments for window building operations.

    Parameters
    ----------
    config : object
        Configuration object with data.lookback, data.horizon, and data.withhold_periods
    X : np.ndarray
        Input data array of shape (time, features)
    tgt_indices : Optional[List[int]]
        Target feature indices to validate

    Raises
    ------
    TypeError
        If X is not a numpy array or withhold_periods is not an integer
    ValueError
        If arguments are invalid (non-positive lookback/horizon, sequence too short)
    IndexError
        If tgt_indices are out of bounds
    """
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
    """
    Calculate maximum starting index for window construction.

    Parameters
    ----------
    T : int
        Total length of the time series
    lookback : int
        Number of historical periods in each window
    horizon : int
        Number of periods ahead to predict

    Returns
    -------
    int
        Maximum valid starting index for window construction

    Raises
    ------
    ValueError
        If sequence is too short for the given lookback and horizon
    """
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
    Build window arrays with train/test split, filtering for consecutive quarters.

    Only includes windows where all periods are consecutive quarters. The last
    `withhold_periods` valid windows are used for testing.

    Parameters
    ----------
    config : object
        Configuration object with data.lookback, data.horizon, and data.withhold_periods
    X : np.ndarray
        Input data array of shape (time, features)
    tgt_indices : Optional[List[int]]
        Indices of target features to predict (None = all features)
    max_start : int
        Maximum valid starting index for windows
    num_features : int
        Number of features in X
    index : pd.PeriodIndex, optional
        Period index for consecutive quarter validation

    Returns
    -------
    tuple of np.ndarray
        (X_train, y_train, X_test, y_test) where:
        - X_train: (N_train, lookback, F)
        - y_train: (N_train, target_dim)
        - X_test: (N_test, lookback, F)
        - y_test: (N_test, target_dim)

    Raises
    ------
    ValueError
        If withhold_periods exceeds number of valid consecutive windows
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
    """
    Extract a single window from the data for training.

    Parameters
    ----------
    X : np.ndarray
        Input data array of shape (time, features)
    start : int
        Starting index for the window
    lookback : int
        Number of historical periods to include
    horizon : int
        Number of periods ahead to predict
    tgt_indices : Optional[List[int]]
        Indices of target features (None = all features)

    Returns
    -------
    tuple of np.ndarray
        (x_win, y_target) where x_win is the input window and y_target is the prediction target
    """
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


def _bs_has_data(df, col_name):
    """
    Check if a balance sheet field is actually populated.

    Parameters
    ----------
    df : pd.DataFrame
        Balance sheet dataframe
    col_name : str
        Column name to check

    Returns
    -------
    bool
        True if column exists and has non-zero values
    """
    return col_name in df.columns and (df[col_name] != 0).any()


def _bs_get_field(df, col_name):
    """
    Safely retrieve a balance sheet field value.

    Parameters
    ----------
    df : pd.DataFrame
        Balance sheet dataframe
    col_name : str
        Column name to retrieve

    Returns
    -------
    pd.Series or int
        Column values if populated, otherwise 0
    """
    return df[col_name] if _bs_has_data(df, col_name) else 0


def _bs_get_deferred_tax_components(df, mappings):
    """
    Retrieve deferred tax components, handling net vs split positions.

    Parameters
    ----------
    df : pd.DataFrame
        Balance sheet dataframe
    mappings : dict
        Mapping of standardized columns to source columns

    Returns
    -------
    tuple
        (dta, dtl, has_net_dt) where:
        - dta: Deferred tax assets values
        - dtl: Deferred tax liabilities values
        - has_net_dt: True if using net deferred tax position split by sign
    """
    dta_sources = mappings.get("Deferred Tax Assets", [])
    has_net_dt = any("net" in src.lower() for src in dta_sources)

    if has_net_dt and _bs_has_data(df, "Deferred Tax Assets"):
        net_dt = df["Deferred Tax Assets"]
        return net_dt.clip(lower=0), (-net_dt).clip(lower=0), True

    return (
        _bs_get_field(df, "Deferred Tax Assets"),
        _bs_get_field(df, "Deferred Tax Liabilities"),
        False,
    )


def _bs_reconstruct_assets(df, dta_component):
    """
    Reconstruct total assets from component fields.

    Parameters
    ----------
    df : pd.DataFrame
        Balance sheet dataframe
    dta_component : pd.Series or int
        Deferred tax assets component

    Returns
    -------
    pd.Series
        Reconstructed total assets
    """
    return (
        _bs_get_field(df, "Cash and Equivalents")
        + _bs_get_field(df, "Receivables")
        + _bs_get_field(df, "Inventory")
        + _bs_get_field(df, "Prepaid and Other Current Assets")
        + _bs_get_field(df, "Marketable Securities (Short-term)")
        + _bs_get_field(df, "Property, Plant, and Equipment (net)")
        + _bs_get_field(df, "Intangible Assets (net)")
        + _bs_get_field(df, "Goodwill")
        + _bs_get_field(df, "Long-term Investments")
        + dta_component
        + _bs_get_field(df, "Other Non-Current Assets")
    )


def _bs_reconstruct_liabilities(df, dtl_component):
    """
    Reconstruct total liabilities from component fields.

    Parameters
    ----------
    df : pd.DataFrame
        Balance sheet dataframe
    dtl_component : pd.Series or int
        Deferred tax liabilities component

    Returns
    -------
    pd.Series
        Reconstructed total liabilities
    """
    return (
        _bs_get_field(df, "Accounts Payable and Accrued Expenses")
        + _bs_get_field(df, "Short-term Debt")
        + _bs_get_field(df, "Deferred Revenue")
        + _bs_get_field(df, "Other Current Liabilities")
        + _bs_get_field(df, "Long-term Debt")
        + dtl_component
        + _bs_get_field(df, "Lease Liabilities")
        + _bs_get_field(df, "Other Non-Current Liabilities")
    )


def _bs_reconstruct_equity(df, mappings):
    """
    Reconstruct total equity from component fields.

    Parameters
    ----------
    df : pd.DataFrame
        Balance sheet dataframe
    mappings : dict
        Mapping of standardized columns to source columns

    Returns
    -------
    tuple
        (equity, has_treasury) where:
        - equity: Reconstructed total equity
        - has_treasury: True if treasury stock is available and populated
    """
    treasury_sources = mappings.get("Treasury Stock", [])
    has_treasury = len(treasury_sources) > 0 and _bs_has_data(df, "Treasury Stock")

    equity = (
        _bs_get_field(df, "Common Stock and APIC")
        + _bs_get_field(df, "Retained Earnings")
        + _bs_get_field(df, "Accumulated Other Comprehensive Income")
    )

    if has_treasury:
        equity -= _bs_get_field(df, "Treasury Stock")

    return equity, has_treasury


def _bs_print_diagnostics(
    df,
    threshold,
    dta_component,
    dtl_component,
    has_net_dt,
    has_treasury,
    assets_reconstructed,
    liabilities_reconstructed,
    equity_reconstructed,
):
    """
    Print diagnostic information for balance sheet validation.

    Parameters
    ----------
    df : pd.DataFrame
        Balance sheet dataframe
    threshold : float
        Acceptable discrepancy threshold
    dta_component : pd.Series or int
        Deferred tax assets values
    dtl_component : pd.Series or int
        Deferred tax liabilities values
    has_net_dt : bool
        Whether using net deferred tax position
    has_treasury : bool
        Whether treasury stock is available
    assets_reconstructed : pd.Series
        Reconstructed total assets
    liabilities_reconstructed : pd.Series
        Reconstructed total liabilities
    equity_reconstructed : pd.Series
        Reconstructed total equity
    """
    print("\n" + "=" * 80)
    print(" BALANCE SHEET VALIDATION DIAGNOSTICS")
    print("=" * 80)

    if has_net_dt:
        print("ℹ Net Deferred Tax: Split into DTA/DTL by sign")
        print(f"  - Periods with DTA: {(dta_component > 0).sum()}")
        print(f"  - Periods with DTL: {(dtl_component > 0).sum()}")

    if not has_treasury:
        print(
            "⚠ Treasury Stock: Not available - equity reconstruction may be incomplete"
        )

    if not _bs_has_data(df, "Lease Liabilities"):
        print(
            "ℹ Lease Liabilities: Not mapped (likely in Other Non-Current Liabilities)"
        )

    # Compare reconstructed vs reported
    for name, reconstructed, col in [
        ("Assets", assets_reconstructed, "Total Assets"),
        ("Liabilities", liabilities_reconstructed, "Total Liabilities"),
        ("Equity", equity_reconstructed, "Total Equity"),
    ]:
        if _bs_has_data(df, col):
            diff = (df[col] - reconstructed).abs()
            n_mismatch = (diff > threshold).sum()
            if n_mismatch > 0:
                print(f"\n⚠ {name}: Reconstructed != Reported for {n_mismatch} rows")
                print(f"  Median diff: ${diff.median():,.0f}")
                print(f"  Max diff: ${diff.max():,.0f}")
                if name == "Equity" and not has_treasury:
                    print("  (Likely due to missing Treasury Stock)")


def bs_identity_checker(
    df: pd.DataFrame, mappings: dict, threshold: float = 1e6
) -> pd.DataFrame:
    """
    Validate balance sheet accounting identity (Assets = Liabilities + Equity).

    Uses mappings to intelligently handle:
    - Net deferred tax positions
    - Missing fields (Treasury Stock, Lease Liabilities)
    - Composite/rollup features

    Args:
        df: DataFrame with standardized balance sheet columns
        mappings: Dict mapping new features to lists of old features
        threshold: Maximum acceptable identity discrepancy

    Returns:
        DataFrame with rows passing identity validation
    """
    df = df.fillna(0)

    dta_component, dtl_component, has_net_dt = _bs_get_deferred_tax_components(
        df, mappings
    )
    assets_reconstructed = _bs_reconstruct_assets(df, dta_component)
    liabilities_reconstructed = _bs_reconstruct_liabilities(df, dtl_component)
    equity_reconstructed, has_treasury = _bs_reconstruct_equity(df, mappings)

    _bs_print_diagnostics(
        df,
        threshold,
        dta_component,
        dtl_component,
        has_net_dt,
        has_treasury,
        assets_reconstructed,
        liabilities_reconstructed,
        equity_reconstructed,
    )

    # Prefer reported totals for identity check
    assets = (
        df["Total Assets"] if _bs_has_data(df, "Total Assets") else assets_reconstructed
    )
    liabilities = (
        df["Total Liabilities"]
        if _bs_has_data(df, "Total Liabilities")
        else liabilities_reconstructed
    )
    equity = (
        df["Total Equity"] if _bs_has_data(df, "Total Equity") else equity_reconstructed
    )

    accounting_id = assets - (liabilities + equity)
    mask = accounting_id.abs() <= threshold
    discrepancies = accounting_id[~mask]

    if len(discrepancies) > 0:
        print(f"\n{'=' * 80}")
        print(
            f"❌ IDENTITY FAILURES: {len(discrepancies)} rows exceed threshold ${threshold:,.0f}"
        )
        print(f"{'=' * 80}")
    else:
        print(f"\n{'=' * 80}")
        print(
            f"✓ IDENTITY VALIDATED: All {len(df)} rows pass (threshold ${threshold:,.0f})"
        )
        print(f"{'=' * 80}")

    return df[mask]


def drop_constants(df, verbose=False):
    """
    Remove constant columns and sparse rows from dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    verbose : bool, optional
        Whether to print information about dropped columns

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with constant columns and sparse rows removed
    """
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


def ytd_to_quarterly(df, exclude_columns=None):
    """
    Convert YTD cumulative to quarterly values.

    Args:
        df: DataFrame with YTD values
        exclude_columns: List of columns to NOT convert (stock variables)
    """
    df = df.copy()
    df = df.sort_index(ascending=True)

    if exclude_columns is None:
        exclude_columns = []

    # Separate stock vs flow columns
    flow_cols = [col for col in df.columns if col not in exclude_columns]
    stock_cols = exclude_columns

    # Only convert flow columns
    quarterly = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    # Copy stock columns as-is
    for col in stock_cols:
        quarterly[col] = df[col]

    # Convert flow columns
    for i, idx in enumerate(df.index):
        if i == 0:
            quarterly.loc[idx, flow_cols] = df.loc[idx, flow_cols]
        else:
            prev_idx = df.index[i - 1]
            current = df.loc[idx, flow_cols]
            prev = df.loc[prev_idx, flow_cols]

            # Detect fiscal year reset
            is_fy_reset = (current < prev).sum() > len(flow_cols) * 0.5

            if is_fy_reset:
                quarterly.loc[idx, flow_cols] = current
            else:
                quarterly.loc[idx, flow_cols] = current - prev

    return quarterly


def lei_to_ticker(lei):
    """
    Convert Legal Entity Identifier (LEI) to stock ticker symbol.

    Parameters
    ----------
    lei : str
        Legal Entity Identifier

    Returns
    -------
    str or None
        Stock ticker symbol if found, 'non-english' for non-Latin company names,
        or None if lookup fails
    """
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
    """
    Convert multiple LEIs to ticker symbols using LLM-based lookup.

    Parameters
    ----------
    names : list
        List of company names or LEIs
    ticker_data : str
        Path to save/load ticker mapping JSON file
    llm_client : LLMClient
        LLM client for company name to ticker conversion

    Returns
    -------
    dict
        Mapping of company names to ticker symbols
    """
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
    """
    Convert ticker symbols to company names using LLM-based lookup.

    Parameters
    ----------
    names : list
        List of ticker symbols
    llm_client : LLMClient
        LLM client for ticker to company name conversion

    Returns
    -------
    dict
        Mapping of ticker symbols to company names
    """
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


def _derived_col(d, name):
    """
    Safely retrieve column for derived calculations.

    Parameters
    ----------
    d : pd.DataFrame
        Dataframe to retrieve column from
    name : str
        Column name

    Returns
    -------
    pd.Series
        Column values with NaN filled as 0, or Series of 0 if missing
    """
    return d[name].fillna(0) if name in d.columns else pd.Series(0, index=d.index)


def _should_calculate(d, col_name):
    """
    Check if a derived column needs to be calculated.

    Parameters
    ----------
    d : pd.DataFrame
        Dataframe to check
    col_name : str
        Column name to check

    Returns
    -------
    bool
        True if column is missing or contains only zero/NaN values
    """
    if col_name not in d.columns:
        return True
    col_data = d[col_name]
    return col_data.isna().all() or (col_data == 0).all()


def _add_derived_assets(d):
    """
    Add derived asset columns if they don't already exist.

    Calculates Total Current Assets and Total Non-Current Assets from their
    components, and Total Assets from the sum of current and non-current.

    Parameters
    ----------
    d : pd.DataFrame
        Balance sheet dataframe (modified in-place)
    """
    if _should_calculate(d, "Total Current Assets"):
        d["Total Current Assets"] = (
            _derived_col(d, "Cash and Equivalents")
            + _derived_col(d, "Receivables")
            + _derived_col(d, "Inventory")
            + _derived_col(d, "Prepaid and Other Current Assets")
            + _derived_col(d, "Marketable Securities (Short-term)")
        )

    if _should_calculate(d, "Total Non-Current Assets"):
        d["Total Non-Current Assets"] = (
            _derived_col(d, "Property, Plant, and Equipment (net)")
            + _derived_col(d, "Intangible Assets (net)")
            + _derived_col(d, "Goodwill")
            + _derived_col(d, "Long-term Investments")
            + _derived_col(d, "Deferred Tax Assets")
            + _derived_col(d, "Other Non-Current Assets")
        )

    if _should_calculate(d, "Total Assets"):
        has_current = not _should_calculate(d, "Total Current Assets")
        has_noncurrent = not _should_calculate(d, "Total Non-Current Assets")
        if has_current and has_noncurrent:
            d["Total Assets"] = (
                d["Total Current Assets"] + d["Total Non-Current Assets"]
            )
        elif "Total Assets" not in d.columns:
            d["Total Assets"] = d.get("Total Current Assets", 0) + d.get(
                "Total Non-Current Assets", 0
            )


def _add_derived_liabilities(d):
    """
    Add derived liability columns if they don't already exist.

    Calculates Total Current Liabilities and Total Non-Current Liabilities from their
    components, and Total Liabilities from the sum of current and non-current.

    Parameters
    ----------
    d : pd.DataFrame
        Balance sheet dataframe (modified in-place)
    """
    if _should_calculate(d, "Total Current Liabilities"):
        d["Total Current Liabilities"] = (
            _derived_col(d, "Accounts Payable and Accrued Expenses")
            + _derived_col(d, "Short-term Debt")
            + _derived_col(d, "Deferred Revenue")
            + _derived_col(d, "Other Current Liabilities")
        )

    if _should_calculate(d, "Total Non-Current Liabilities"):
        d["Total Non-Current Liabilities"] = (
            _derived_col(d, "Long-term Debt")
            + _derived_col(d, "Deferred Tax Liabilities")
            + _derived_col(d, "Lease Liabilities")
            + _derived_col(d, "Other Non-Current Liabilities")
        )

    if _should_calculate(d, "Total Liabilities"):
        has_current = not _should_calculate(d, "Total Current Liabilities")
        has_noncurrent = not _should_calculate(d, "Total Non-Current Liabilities")
        if has_current and has_noncurrent:
            d["Total Liabilities"] = (
                d["Total Current Liabilities"] + d["Total Non-Current Liabilities"]
            )
        elif "Total Liabilities" not in d.columns:
            d["Total Liabilities"] = d.get("Total Current Liabilities", 0) + d.get(
                "Total Non-Current Liabilities", 0
            )


def _add_derived_equity(d):
    """
    Add derived equity column if it doesn't already exist.

    Calculates Total Equity from common stock, retained earnings, treasury stock,
    and accumulated other comprehensive income.

    Parameters
    ----------
    d : pd.DataFrame
        Balance sheet dataframe (modified in-place)
    """
    if _should_calculate(d, "Total Equity"):
        d["Total Equity"] = (
            _derived_col(d, "Common Stock and APIC")
            + _derived_col(d, "Retained Earnings")
            - _derived_col(d, "Treasury Stock")
            + _derived_col(d, "Accumulated Other Comprehensive Income")
        )


def _add_derived_income(d):
    """
    Add derived income statement columns if they don't already exist.

    Calculates Gross Profit, Operating Income, and Total Debt from their components.

    Parameters
    ----------
    d : pd.DataFrame
        Financial statement dataframe (modified in-place)
    """
    if _should_calculate(d, "Gross Profit"):
        if (
            "Gross Profit" not in d.columns
            and "Total Revenues" in d.columns
            and "Total Cost of Revenue" in d.columns
        ):
            d["Gross Profit"] = _derived_col(d, "Total Revenues") - _derived_col(
                d, "Total Cost of Revenue"
            )

    if _should_calculate(d, "Operating Income") and "Operating Income" not in d.columns:
        if "Gross Profit" in d.columns and "Total Operating Expenses" in d.columns:
            d["Operating Income"] = _derived_col(d, "Gross Profit") - _derived_col(
                d, "Total Operating Expenses"
            )
        elif (
            "Total Revenues" in d.columns
            and "Total Cost of Revenue" in d.columns
            and "Total Operating Expenses" in d.columns
        ):
            d["Operating Income"] = (
                _derived_col(d, "Total Revenues")
                - _derived_col(d, "Total Cost of Revenue")
                - _derived_col(d, "Total Operating Expenses")
            )

    if _should_calculate(d, "Total Debt"):
        d["Total Debt"] = _derived_col(d, "Short-term Debt") + _derived_col(
            d, "Long-term Debt"
        )


def add_derived_columns(df):
    """
    Add derived columns only if they don't already exist or are all zero/NaN.
    Preserves mapped values from standardized financial statements.
    """
    d = df.copy()

    _add_derived_assets(d)
    _add_derived_liabilities(d)
    _add_derived_equity(d)
    _add_derived_income(d)

    return d


def standardise_rating(rating):
    """
    Standardize credit rating format by removing provisional markers.

    Parameters
    ----------
    rating : str
        Credit rating string (e.g., 'BBB(P)')

    Returns
    -------
    str or None
        Standardized rating (e.g., 'BBB') or None if input is NaN
    """
    if pd.isna(rating):
        return None
    return rating.replace("(P)", "").strip()


def drop_non_numeric_columns(df):
    """
    Remove columns that contain non-numeric, non-NaN values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    Returns
    -------
    pd.DataFrame
        Dataframe with only numeric columns retained
    """
    numeric_cols = []

    for col in df.columns:
        # Convert to numeric, coercing errors to NaN
        converted = pd.to_numeric(df[col], errors="coerce")

        # Check if all values are either numeric or NaN (no strings/objects left)
        if converted.notna().sum() == df[col].notna().sum():
            numeric_cols.append(col)

    return df[numeric_cols]


def remap_financial_dataframe(df, column_mapping):
    """
    Remap and aggregate dataframe columns according to mapping structure.

    Handles two types of multi-source mappings:
    1. Alternate taxonomies (mutually exclusive) - coalesce (take first non-null)
    2. True aggregations (can coexist) - sum

    Parameters:
    -----------
    df : pd.DataFrame
        Source dataframe with original column names
    column_mapping : dict
        Nested dictionary mapping new column names to lists of existing column names

    Returns:
    --------
    pd.DataFrame
        New dataframe with remapped columns
    """
    flat_mapping = _extract_leaf_mappings(column_mapping)
    new_df = pd.DataFrame(index=df.index)

    for new_col, old_cols in flat_mapping.items():
        new_df[new_col] = _map_single_column(df, new_col, old_cols)

    return new_df


def _extract_leaf_mappings(mapping):
    """
    Extract leaf-level column mappings from nested mapping structure.

    Parameters
    ----------
    mapping : dict
        Nested dictionary mapping structure

    Returns
    -------
    dict
        Flattened dictionary with only leaf-level mappings
    """
    leaf_map = {}

    for key, value in mapping.items():
        if key == "__unmapped__":
            continue

        if isinstance(value, dict):
            nested_maps = _extract_leaf_mappings(value)
            leaf_map.update(nested_maps)
        elif isinstance(value, list):
            leaf_map[key] = value

    return leaf_map


def _map_single_column(df, new_col, old_cols):
    """
    Map a single column from one or more source columns.

    Handles both mutually exclusive sources (alternate taxonomies) and
    true aggregations based on data patterns.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe
    new_col : str
        Name of the new column (unused but kept for API consistency)
    old_cols : list
        List of source column names

    Returns
    -------
    pd.Series or np.nan
        Mapped column values (coalesced if mutually exclusive, summed if aggregate)
    """
    if not old_cols:
        return np.nan

    existing_cols = [col for col in old_cols if col in df.columns]

    if not existing_cols:
        return np.nan
    if len(existing_cols) == 1:
        return df[existing_cols[0]]

    # Multiple sources - determine if mutually exclusive or aggregate
    if _is_mutually_exclusive(df, existing_cols):
        # Alternate taxonomies - coalesce (prefer non-null/non-zero values)
        result = df[existing_cols].replace(0, np.nan).bfill(axis=1).iloc[:, 0]
        return result.fillna(0)

    # True aggregation - sum all sources
    return df[existing_cols].sum(axis=1)


def _is_mutually_exclusive(df, cols):
    """
    Check if columns are mutually exclusive (alternate taxonomies).

    Columns are considered mutually exclusive if they rarely have non-zero
    values simultaneously (less than 5% of rows).

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe
    cols : list
        List of column names to check

    Returns
    -------
    bool
        True if columns never have non-zero values simultaneously
    """
    if len(cols) <= 1:
        return False

    # Count rows where multiple columns are non-zero
    non_zero = (df[cols] != 0) & (df[cols].notna())
    simultaneous_nonzero = (non_zero.sum(axis=1) > 1).sum()

    # If <5% of rows have multiple non-zero values, treat as mutually exclusive
    threshold = len(df) * 0.05
    return simultaneous_nonzero < threshold


def load_cached_features(cache_path: Path) -> dict[str, Any]:
    """Load cached parsed features from JSON file."""
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)
