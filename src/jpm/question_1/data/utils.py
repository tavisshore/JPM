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
    name = re.sub(r"^[^-]+-[^_]+_", "", name)
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", name)
    return s.lower()


def get_targets(mode: str = "net_income", ticker: str = "AAPL") -> List[str]:
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
    else:
        # TODO: extend for other tickers
        structure: BalanceSheetStructure = {
            "assets": {
                "current_assets": [],
                "non_current_assets": [],
            },
            "liabilities": {
                "current_liabilities": [],
                "non_current_liabilities": [],
            },
            "equity": [],
        }

    return structure


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
    T, F = X.shape

    if withhold < 0:
        raise ValueError("withhold must be >= 0")

    # total number of possible window starts
    max_start = T - lookback - horizon + 1
    if max_start <= 0:
        raise ValueError("Sequence too short for given lookback and horizon")

    if withhold > max_start:
        raise ValueError(
            f"withhold={withhold} is too large; max possible windows is {max_start}"
        )

    split_idx = max_start - withhold  # number of train windows

    X_train, y_train = [], []
    X_test, y_test = [], []

    for t in range(max_start):
        x_win = X[t : t + lookback]
        y_target = X[t + lookback + horizon - 1]

        if tgt_indices is not None:
            y_target = y_target[tgt_indices]

        # Partition into train vs withheld windows
        if t < split_idx:
            X_train.append(x_win)
            y_train.append(y_target)
        else:
            X_test.append(x_win)
            y_test.append(y_target)

    X_train = np.stack(X_train) if X_train else np.empty((0, lookback, F))
    y_train = np.stack(y_train) if y_train else np.empty((0, y_target.shape[-1]))
    X_test = np.stack(X_test) if X_test else np.empty((0, lookback, F))
    y_test = np.stack(y_test) if y_test else np.empty((0, y_target.shape[-1]))

    return X_train, y_train, X_test, y_test


def bs_identity(df, ticker, tol=1e1) -> None:
    """
    Checks Assets = Liabilities + Equity for each row.
    Returns a DataFrame with the computed values and the error.
    """
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
