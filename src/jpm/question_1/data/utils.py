import re
from typing import List, Optional, Tuple, TypedDict

import numpy as np


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
        diff = np.abs(col1[mask_both_not_nan] - col2[mask_both_not_nan])
        matches_numeric = (diff <= tolerance).sum()
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
    df, kind="balance sheet", tolerance=1e-6, similarity_threshold=0.999, verbose=True
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

    if verbose:
        print_duplicate_analysis(results, kind)

    # Drop duplicate columns
    cleaned_df = df.drop(columns=results["columns_to_drop"])

    return cleaned_df, results


def print_duplicate_analysis(results, kind="balance sheet"):
    """
    Pretty print the duplicate column analysis with colors.
    """

    try:
        from colorama import Fore, Style, init

        init(autoreset=True)
    except ImportError:
        # Fallback to no colors
        class Fore:
            CYAN = GREEN = RED = YELLOW = MAGENTA = WHITE = ""

        class Style:
            RESET_ALL = BRIGHT = ""

    print(f"\n{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
    print(
        f"{Fore.CYAN}{Style.BRIGHT} {kind.capitalize()} "
        f"DUPLICATE COLUMN ANALYSIS {Style.RESET_ALL}"
    )
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")

    summary = results["summary"]
    print(
        f"{Fore.WHITE}Total Columns: "
        f"{Fore.CYAN}{summary['total_columns']}{Style.RESET_ALL}"
    )
    print(
        f"{Fore.YELLOW}Duplicate Groups Found: "
        f"{Fore.CYAN}{summary['duplicate_groups_found']}{Style.RESET_ALL}"
    )
    print(
        f"{Fore.RED}Columns to Drop: "
        f"{Fore.CYAN}{summary['columns_to_drop']}{Style.RESET_ALL}"
    )
    print(
        f"{Fore.GREEN}Columns Remaining: "
        f"{Fore.CYAN}{summary['columns_remaining']}{Style.RESET_ALL}"
    )

    if results["duplicate_groups"]:
        print(f"\n{Fore.MAGENTA}{'─' * 80}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}DUPLICATE GROUPS:{Style.RESET_ALL}\n")

        for i, group in enumerate(results["duplicate_groups"], 1):
            print(f"{Fore.YELLOW}Group {i}:{Style.RESET_ALL}")

            # Determine which to keep
            to_keep = select_column_to_keep(group)

            for col in group:
                if col == to_keep:
                    print(f"  {Fore.GREEN}✓ KEEP: {col}{Style.RESET_ALL}")
                else:
                    print(f"  {Fore.RED}✗ DROP: {col}{Style.RESET_ALL}")
            print()
    else:
        print(f"\n{Fore.GREEN}✓ No duplicate columns found!{Style.RESET_ALL}")

    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")


# New LLM parsed verifications


# def verify_bs_totals(df, balance_sheet_structure, tolerance=0.01):

#     structure = balance_sheet_structure["structure"]
#     results = {"passed": True, "mismatches": [], "summary": {}}

#     def verify_section(section_data, section_name, parent_name=""):
#         """Recursively verify totals within a section."""

#         for key, value in section_data.items():
#             if key == "__unmapped__":
#                 continue

#             full_path = f"{parent_name}.{key}" if parent_name else key

#             if isinstance(value, dict):
#                 # Check if this dict has a "Total" key
#                 total_keys = [k for k in value.keys() if k.startswith("Total")]

#                 for total_key in total_keys:
#                     # Get all component keys (excluding the total and nested dicts)
#                     component_keys = [
#                         k
#                         for k in value.keys()
#                         if k != total_key
#                         and k != "__unmapped__"
#                         and not isinstance(value[k], dict)
#                     ]

#                     if component_keys and total_key in df.columns:
#                         # Calculate sum of components
#                         existing_components = [
#                             k for k in component_keys if k in df.columns
#                         ]

#                         if existing_components:
#                             calculated_total = df[existing_components].sum(axis=1)
#                             reported_total = df[total_key]

#                             # Check for mismatches
#                             difference = (calculated_total - reported_total).abs()
#                             max_diff = difference.max()

#                             if max_diff > tolerance:
#                                 mismatch_rows = difference[
#                                     difference > tolerance
#                                 ].index.tolist()
#                                 results["passed"] = False
#                                 results["mismatches"].append(
#                                     {
#                                         "section": full_path,
#                                         "total_field": total_key,
#                                         "components": existing_components,
#                                         "max_difference": max_diff,
#                                         "rows_with_mismatch": mismatch_rows[
#                                             :10
#                                         ],  # Limit to first 10
#                                         "num_mismatched_rows": len(mismatch_rows),
#                                     }
#                                 )
#                             else:
#                                 results["summary"][total_key] = "PASS"

#                 # Recurse into nested structure
#                 verify_section(value, key, full_path)

#     # Verify Assets
#     if "Assets" in structure:
#         verify_section(structure["Assets"], "Assets")

#         # Special check: Total Current + Total Non-Current = Total Assets
#         if all(
#             col in df.columns
#             for col in [
#                 "Total Current Assets",
#                 "Total Non-Current Assets",
#                 "Total Assets",
#             ]
#         ):
#             calc_total = df["Total Current Assets"] + df["Total Non-Current Assets"]
#             reported_total = df["Total Assets"]
#             difference = (calc_total - reported_total).abs()
#             max_diff = difference.max()

#             if max_diff > tolerance:
#                 mismatch_rows = difference[difference > tolerance].index.tolist()
#                 results["passed"] = False
#                 results["mismatches"].append(
#                     {
#                         "section": "Assets",
#                         "total_field": "Total Assets",
#                         "components": [
#                             "Total Current Assets",
#                             "Total Non-Current Assets",
#                         ],
#                         "max_difference": max_diff,
#                         "rows_with_mismatch": mismatch_rows[:10],
#                         "num_mismatched_rows": len(mismatch_rows),
#                     }
#                 )
#             else:
#                 results["summary"]["Total Assets"] = "PASS"

#     # Verify Liabilities
#     if "Liabilities" in structure:
#         verify_section(structure["Liabilities"], "Liabilities")

#         # Special check: Total Current + Total Non-Current = Total Liabilities
#         if all(
#             col in df.columns
#             for col in [
#                 "Total Current Liabilities",
#                 "Total Non-Current Liabilities",
#                 "Total Liabilities",
#             ]
#         ):
#             calc_total = (
#                 df["Total Current Liabilities"] + df["Total Non-Current Liabilities"]
#             )
#             reported_total = df["Total Liabilities"]
#             difference = (calc_total - reported_total).abs()
#             max_diff = difference.max()

#             if max_diff > tolerance:
#                 mismatch_rows = difference[difference > tolerance].index.tolist()
#                 results["passed"] = False
#                 results["mismatches"].append(
#                     {
#                         "section": "Liabilities",
#                         "total_field": "Total Liabilities",
#                         "components": [
#                             "Total Current Liabilities",
#                             "Total Non-Current Liabilities",
#                         ],
#                         "max_difference": max_diff,
#                         "rows_with_mismatch": mismatch_rows[:10],
#                         "num_mismatched_rows": len(mismatch_rows),
#                     }
#                 )
#             else:
#                 results["summary"]["Total Liabilities"] = "PASS"

#     # Verify Equity (no subcategories to sum, but included for completeness)
#     if "Equity" in structure:
#         verify_section(structure["Equity"], "Equity")

#     # Verify fundamental accounting equation:
#     # Total Assets = Total Liabilities + Total Equity
#     if all(
#         col in df.columns
#         for col in ["Total Assets", "Total Liabilities", "Total Equity"]
#     ):
#         calc_total = df["Total Liabilities"] + df["Total Equity"]
#         reported_total = df["Total Assets"]
#         difference = (calc_total - reported_total).abs()
#         max_diff = difference.max()

#         if max_diff > tolerance:
#             mismatch_rows = difference[difference > tolerance].index.tolist()
#             results["passed"] = False
#             results["mismatches"].append(
#                 {
#                     "section": "Accounting Equation",
#                     "total_field": "Total Assets",
#                     "components": ["Total Liabilities", "Total Equity"],
#                     "max_difference": max_diff,
#                     "rows_with_mismatch": mismatch_rows[:10],
#                     "num_mismatched_rows": len(mismatch_rows),
#                 }
#             )
#         else:
#             results["summary"]["Accounting Equation
#               (Assets = Liab + Equity)"] = "PASS"

#     # Check Totals section
#     if "Totals" in structure and "Total Liabilities and Equity" in df.columns:
#         if all(
#             col in df.columns
#             for col in [
#                 "Total Liabilities",
#                 "Total Equity",
#                 "Total Liabilities and Equity",
#             ]
#         ):
#             calc_total = df["Total Liabilities"] + df["Total Equity"]
#             reported_total = df["Total Liabilities and Equity"]
#             difference = (calc_total - reported_total).abs()
#             max_diff = difference.max()

#             if max_diff > tolerance:
#                 mismatch_rows = difference[difference > tolerance].index.tolist()
#                 results["passed"] = False
#                 results["mismatches"].append(
#                     {
#                         "section": "Totals",
#                         "total_field": "Total Liabilities and Equity",
#                         "components": ["Total Liabilities", "Total Equity"],
#                         "max_difference": max_diff,
#                         "rows_with_mismatch": mismatch_rows[:10],
#                         "num_mismatched_rows": len(mismatch_rows),
#                     }
#                 )
#             else:
#                 results["summary"]["Total Liabilities and Equity"] = "PASS"

#     return results


# def print_verification_results(results):
#     """Pretty print the verification results."""

#     print("=" * 80)
#     print("BALANCE SHEET VERIFICATION RESULTS")
#     print("=" * 80)

#     if results["passed"]:
#         print("\n✓ ALL CHECKS PASSED")
#         print("\nVerified totals:")
#         for total, status in results["summary"].items():
#             print(f"  • {total}: {status}")
#     else:
#         print("\n✗ VERIFICATION FAILED")
#         print(f"\nFound {len(results['mismatches'])} mismatch(es):\n")

#         for i, mismatch in enumerate(results["mismatches"], 1):
#             print(f"{i}. {mismatch['section']} - {mismatch['total_field']}")
#             print(f"   Components: {', '.join(mismatch['components'])}")
#             print(f"   Max difference: ${mismatch['max_difference']:,.2f}")
#             print(f"   Rows affected: {mismatch['num_mismatched_rows']}")
#             print()

#         if results["summary"]:
#             print("Passed checks:")
#             for total, status in results["summary"].items():
#                 print(f"  • {total}: {status}")

#     print("=" * 80)
