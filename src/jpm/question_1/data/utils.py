import re
from typing import Dict, List, Optional, Tuple, TypedDict, Union

import numpy as np

from jpm.question_1.misc import get_leaf_values


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


def get_targets(mode: str = "net_income", ticker: str = "AAPL") -> List[str]:
    if mode not in {"net_income", "bs", "full"}:
        raise ValueError(f"Unsupported target mode '{mode}'")
    if ticker == "AAPL":
        if mode == "net_income":
            return ["net_income_loss"]
        if mode == "bs":
            return get_leaf_values(get_bs_structure(ticker=ticker))
        if mode == "full":
            return get_leaf_values(get_bs_structure(ticker=ticker))
    raise ValueError(f"Unsupported target mode '{mode}' for ticker '{ticker}'")


# Getting the correct columns from dataframes for BS -> later IS + CF
def get_bs_structure(ticker: str = "AAPL") -> BalanceSheetStructure:
    if ticker in ["AAPL"]:
        structure: BalanceSheetStructure = {
            "assets": {
                "current_assets": [
                    "cash_and_cash_equivalents_at_carrying_value",
                    "marketable_securities_current",
                    "accounts_receivable_net_current",
                    "nontrade_receivables_current",
                    "inventory_net",
                    "other_assets_current",
                ],
                "non_current_assets": [
                    "marketable_securities_noncurrent",
                    "other_assets_noncurrent",
                    "property_plant_and_equipment_net",
                ],
            },
            "liabilities": {
                "current_liabilities": [
                    "accounts_payable_current",
                    "contract_with_customer_liability_current",
                    "other_liabilities_current",
                    "commercial_paper",
                    "other_short_term_borrowings",
                    "long_term_debt_current",
                ],
                "non_current_liabilities": [
                    "other_liabilities_noncurrent",
                    "long_term_debt_noncurrent",
                ],
            },
            "equity": [
                "retained_earnings_accumulated_deficit",
                "common_stocks_including_additional_paid_in_capital",
                "accumulated_other_comprehensive_income_loss_net_of_tax",
            ],
        }
        return structure
    raise ValueError(f"Unsupported ticker '{ticker}' for balance sheet structure")


def get_is_structure(ticker="AAPL") -> Dict[str, list[str]]:
    if ticker in ["AAPL"]:
        structure = {
            "Revenues": [
                "revenue_from_contract_with_customer_excluding_assessed_tax",
            ],
            "Expenses": [
                "cost_of_goods_and_services_sold",
                "operating_expenses",
                "selling_general_and_administrative_expense",
                "research_and_development_expense",
                # "interest_expense",
                "income_tax_expense_benefit",
            ],
        }
        return structure
    raise ValueError(f"Unsupported ticker '{ticker}' for income statement structure")


def get_cf_structure(
    ticker="AAPL", flatten=False
) -> Union[Dict[str, list[str]], list[str]]:
    if ticker in ["AAPL"]:
        structure = {
            "operating_cash_flow": [
                "net_income_loss",
                "depreciation_depletion_and_amortization",
                "share_based_compensation",
                "other_noncash_income_expense",
                # Working capital (only the core components)
                "increase_decrease_in_accounts_receivable",
                "increase_decrease_in_accounts_payable",
                "increase_decrease_in_inventories",
            ],
            "investing_cash_flow": [
                # Core investing activity = capex + investment changes
                "payments_to_acquire_property_plant_and_equipment",
                "payments_to_acquire_other_investments",
                "proceeds_from_sale_and_maturity_of_other_investments",
            ],
            "financing_cash_flow": [
                # Equity
                "proceeds_from_issuance_of_common_stock",
                "payments_for_repurchase_of_common_stock",
                # Debt
                "proceeds_from_issuance_of_long_term_debt",
                "repayments_of_long_term_debt",
                # Dividends
                "payments_of_dividends",
            ],
        }

        if flatten:
            return get_leaf_values(structure)
        return structure
    raise ValueError(f"Unsupported ticker '{ticker}' for cash flow structure")


def build_windows(
    X: np.ndarray,
    lookback: int = 3,
    horizon: int = 1,
    tgt_indices: Optional[List[int]] = None,
    withhold: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding windows over X and split the *last `withhold`
    targets* into a test set.

    Returns:
        X_train: (N_train, lookback, F)
        y_train: (N_train, target_dim)
        X_test:  (N_test, lookback, F)   where N_test == withhold
        y_test:  (N_test, target_dim)
    """
    _validate_window_args(X, lookback, horizon, tgt_indices, withhold)
    T, F = X.shape
    max_start = _max_start(T, lookback, horizon)
    split_idx = _split_index(max_start, withhold)

    X_train, y_train, X_test, y_test = _build_window_arrays(
        X, lookback, horizon, tgt_indices, max_start, split_idx, F
    )
    return X_train, y_train, X_test, y_test


def _validate_window_args(
    X: np.ndarray,
    lookback: int,
    horizon: int,
    tgt_indices: Optional[List[int]],
    withhold: int,
) -> None:
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy ndarray")
    if X.ndim != 2:
        raise ValueError(f"X must be 2D array of shape (time, features); got {X.shape}")
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if tgt_indices is not None and any(i < 0 or i >= X.shape[1] for i in tgt_indices):
        raise IndexError("tgt_indices contain out-of-bounds indices for X features")
    if not isinstance(withhold, int):
        raise TypeError("withhold must be an integer")
    if withhold < 0:
        raise ValueError("withhold must be >= 0")

    if lookback + horizon > X.shape[0]:
        raise ValueError("Sequence too short for given lookback and horizon")


def _max_start(T: int, lookback: int, horizon: int) -> int:
    max_start = T - lookback - horizon + 1
    if max_start <= 0:
        raise ValueError("Sequence too short for given lookback and horizon")
    return max_start


def _split_index(max_start: int, withhold: int) -> int:
    if withhold > max_start:
        raise ValueError(
            f"withhold={withhold} is too large; max possible windows is {max_start}"
        )
    return max_start - withhold


def _build_window_arrays(
    X: np.ndarray,
    lookback: int,
    horizon: int,
    tgt_indices: Optional[List[int]],
    max_start: int,
    split_idx: int,
    num_features: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, y_train = [], []
    X_test, y_test = [], []

    for t in range(max_start):
        x_win, y_target = _extract_window(X, t, lookback, horizon, tgt_indices)
        if t < split_idx:
            X_train.append(x_win)
            y_train.append(y_target)
        else:
            X_test.append(x_win)
            y_test.append(y_target)

    X_train_arr = (
        np.stack(X_train) if X_train else np.empty((0, lookback, num_features))
    )
    y_train_arr = np.stack(y_train) if y_train else np.empty((0, y_target.shape[-1]))
    X_test_arr = np.stack(X_test) if X_test else np.empty((0, lookback, num_features))
    y_test_arr = np.stack(y_test) if y_test else np.empty((0, y_target.shape[-1]))

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


def bs_identity(df, ticker, tol=1e1) -> None:
    """
    Checks Assets = Liabilities + Equity for each row.
    Returns a DataFrame with the computed values and the error.
    """
    if tol <= 0:
        raise ValueError("tol must be positive")
    if df is None or df.empty:
        raise ValueError("Input dataframe for bs_identity cannot be empty")
    categories = get_bs_structure(ticker)
    A_cols = (
        categories["assets"]["current_assets"]
        + categories["assets"]["non_current_assets"]
    )
    L_cols = (
        categories["liabilities"]["current_liabilities"]
        + categories["liabilities"]["non_current_liabilities"]
    )
    E_cols = categories["equity"]

    missing_cols = [c for c in A_cols + L_cols + E_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataframe missing required balance sheet columns: {missing_cols}"
        )

    # Ensure missing values don't break the sums
    df_filled = df.fillna(0)

    out = df_filled.assign(
        assets_sum=df_filled[A_cols].sum(axis=1),
        liabilities_sum=df_filled[L_cols].sum(axis=1),
        equity_sum=df_filled[E_cols].sum(axis=1),
    )

    out["identity_error"] = out["assets_sum"] - (
        out["liabilities_sum"] + out["equity_sum"]
    )
    out["valid"] = out["identity_error"].abs() < tol

    print(f"Accounting Identity: {out['valid'].sum()}/{len(out)} valid")
