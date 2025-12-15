import os
from pathlib import Path
from typing import Dict

import edgar
import numpy as np
import pandas as pd
import tensorflow as tf
from edgar import Company
from edgar.xbrl import XBRLS
from sklearn.preprocessing import StandardScaler

from jpm.question_1.clients.llm_client import LLMClient
from jpm.question_1.config import Config
from jpm.question_1.data.utils import (
    build_windows,
    remove_duplicate_columns,
    xbrl_to_raw,
)
from jpm.question_1.data.vis import pretty_print_full_mapping
from jpm.question_1.misc import get_leaf_keys

# SEC requires user identification via email
email = os.getenv("EDGAR_EMAIL")
if not email:
    raise ValueError(
        "EDGAR_EMAIL environment variable not set"
        "set with 'export EDGAR_EMAIL='your_email@jpm.com'"
    )

edgar.set_identity(email)
pd.set_option("future.no_silent_downcasting", True)

# Structure for LLM parsing
# Keys are grouped hier for readability; LLM gets full JSON structure
# Totals and __unmapped__ are included to capture any extra/missing items
# These are then used for verification and negated
balance_sheet_structure = {
    "structure": {
        "Assets": {
            "Current": {
                "Cash and Cash Equivalents": [],
                "Accounts Receivable": [],
                "Inventory": [],
                "Prepaid Expenses": [],
                "Marketable Securities (Short-term Investments)": [],
                "Other Current Assets": [],
                "Total Current Assets": [],
            },
            "Non-Current": {
                "Property, Plant, and Equipment (PP&E)": [],
                "Intangible Assets": [],
                "Goodwill": [],
                "Marketable Securities (Non-current)": [],
                "Long-term Investments": [],
                "Deferred Tax Assets": [],
                "Other Non-Current Assets": [],
                "Total Non-Current Assets": [],
            },
            "Total Assets": [],
        },
        "Liabilities": {
            "Current": {
                "Accounts Payable": [],
                "Accrued Expenses": [],
                "Short-term Debt": [],
                "Current Portion of Long-term Debt": [],
                "Unearned Revenue (Deferred Revenue)": [],
                "Income Taxes Payable": [],
                "Other Current Liabilities": [],
                "Total Current Liabilities": [],
            },
            "Non-Current": {
                "Long-term Debt": [],
                "Deferred Tax Liabilities": [],
                "Pension Liabilities": [],
                "Lease Liabilities": [],
                "Other Non-Current Liabilities": [],
                "Total Non-Current Liabilities": [],
            },
            "Total Liabilities": [],
        },
        "Equity": {
            "Common Stock": {
                "Amount": [],
                "Shares Issued": [],
                "Shares Outstanding": [],
            },
            "Preferred Stock": {
                "Amount": [],
                "Shares Issued": [],
                "Shares Outstanding": [],
            },
            "Additional Paid-in Capital": [],
            "Retained Earnings": [],
            "Treasury Stock": [],
            "Accumulated Other Comprehensive Income (AOCI)": [],
            "Total Equity": [],
        },
        "Totals": {
            "Total Liabilities and Equity": [],
        },
        "__unmapped__": [],
    },
    "derived": [
        "Total Current Assets",
        "Total Non-Current Assets",
        "Total Assets",
        "Total Current Liabilities",
        "Total Non-Current Liabilities",
        "Total Liabilities",
        "Common Stock",
        "Amount",
        "Shares Issued",
        "Shares Outstanding",
        "Preferred Stock",
        "Amount",
        "Shares Issued",
        "Shares Outstanding",
        "Total Equity",
        "Total Liabilities and Equity",
        "__unmapped__",
    ],
}


income_statement_structure = {
    "structure": {
        "Revenues": {
            "Product Revenue": [],
            "Service Revenue": [],
            "Other Revenue": [],
            "Total Revenues": [],
        },
        "Cost of Revenue": {
            "Cost of Goods Sold": [],
            "Cost of Services": [],
            "Other Cost of Revenue": [],
            "Total Cost of Revenue": [],
        },
        "Gross Profit": [],
        "Operating Expenses": {
            "Selling, General and Administrative": [],
            "Research and Development": [],
            "Restructuring and Impairment": [],
            "Other Operating Expenses": [],
            "Total Operating Expenses": [],
        },
        "Operating Income (Loss)": [],
        "Non-Operating Income/Expenses": {
            "Interest Income": [],
            "Interest Expense": [],
            "Other Non-Operating Items": [],
            "Total Non-Operating Income (Expense)": [],
        },
        "Income Before Taxes": [],
        "Taxes": {
            "Income Tax Expense": [],
            "Other Taxes": [],
            "Total Taxes": [],
        },
        "Net Income (Loss)": [],
        "Per-Share Metrics": {
            "Basic EPS": [],
            "Diluted EPS": [],
            "Weighted Average Shares Basic": [],
            "Weighted Average Shares Diluted": [],
        },
        "Extraordinary Items": {
            "Discontinued Operations": [],
            "Other Extraordinary Items": [],
        },
        "Totals": {
            "Comprehensive Income": [],
        },
        "__unmapped__": [],
    },
    "derived": [
        "Total Revenues",
        "Total Cost of Revenue",
        "Gross Profit",
        "Total Operating Expenses",
        "Operating Income (Loss)",
        "Total Non-Operating Income (Expense)",
        "Income Before Taxes",
        "Total Taxes",
        "Comprehensive Income",
        "__unmapped__",
    ],
}


cash_flow_structure = {
    "structure": {
        "Operating Activities": {
            "Stock-Based Compensation": [],
            "Changes in Working Capital": [],
            "Other Operating Activities": [],
            "Net Cash Provided by (Used in) Operating Activities": [],
        },
        "Investing Activities": {
            "Capital Expenditures": [],
            "Acquisitions": [],
            "Purchases of Investments": [],
            "Sales/Maturities of Investments": [],
            "Other Investing Activities": [],
            "Net Cash Provided by (Used in) Investing Activities": [],
        },
        "Financing Activities": {
            "Proceeds from Debt Issuance": [],
            "Debt Repayments": [],
            "Proceeds from Stock Issuance": [],
            "Stock Repurchases": [],
            "Dividends Paid": [],
            "Other Financing Activities": [],
            "Net Cash Provided by (Used in) Financing Activities": [],
        },
        "Supplemental Disclosures": {
            "Cash Paid for Income Taxes": [],
            "Cash Paid for Interest": [],
            "Non-Cash Investing and Financing Activities": [],
        },
        "Totals": {
            "Effect of Exchange Rate Changes on Cash": [],
            "Net Change in Cash": [],
        },
        "__unmapped__": [],
    },
    "derived": [
        "Total Changes in Working Capital",
        "Net Cash Provided by (Used in) Operating Activities",
        "Net Cash Provided by (Used in) Investing Activities",
        "Net Cash Provided by (Used in) Financing Activities",
        "Effect of Exchange Rate Changes on Cash",
        "Net Change in Cash",
        "__unmapped__",
    ],
}


def remap_financial_dataframe(df, column_mapping):
    """
    Remap and aggregate dataframe columns according to mapping structure.

    Parameters:
    -----------
    df : pd.DataFrame
        Source dataframe with original column names
    column_mapping : dict
        Nested dictionary mapping new column names to lists of existing column names

    Returns:
    --------
    pd.DataFrame
        New dataframe with remapped columns, values summed where multiple sources exist
    """

    def extract_leaf_mappings(mapping):
        """Extract only leaf-level column mappings."""
        leaf_map = {}

        for key, value in mapping.items():
            if key == "__unmapped__":
                continue

            if isinstance(value, dict):
                # Recurse into nested structure
                nested_maps = extract_leaf_mappings(value)
                leaf_map.update(nested_maps)
            elif isinstance(value, list):
                # Leaf node - direct mapping
                leaf_map[key] = value

        return leaf_map

    # Extract flat mapping
    flat_mapping = extract_leaf_mappings(column_mapping)

    # Create new dataframe
    new_df = pd.DataFrame(index=df.index)

    # Process each mapped column
    for new_col, old_cols in flat_mapping.items():
        if not old_cols:  # Empty list - column will be NaN
            new_df[new_col] = np.nan
        else:
            # Find which columns actually exist in the source dataframe
            existing_cols = [col for col in old_cols if col in df.columns]

            if existing_cols:
                # Sum the existing columns
                if len(existing_cols) == 1:
                    new_df[new_col] = df[existing_cols[0]]
                else:
                    new_df[new_col] = df[existing_cols].sum(axis=1)
            else:
                # None of the source columns exist
                new_df[new_col] = np.nan

    return new_df


class EdgarDataLoader:
    """Load, scale, and window Edgar filings into train/val datasets."""

    def __init__(self, config: Config, overwrite: bool = False) -> None:
        self.config = config
        self.overwrite = overwrite
        self.cache_statement = Path(
            f"{self.config.data.cache_dir}/{self.config.data.ticker}.parquet"
        )
        self.cache_statement.parent.mkdir(parents=True, exist_ok=True)

        self.llm_client = LLMClient()
        # Prefer cached parquet to avoid repeated SEC fetches
        self.create_dataset()

    def create_dataset(self) -> None:
        self._validate_target_type()

        self.data = self._load_or_fetch_data()

        self._set_feature_index()
        self._prepare_targets()

        X_scaled, scaler = self._scale_features()
        self._set_scaler_stats(scaler)

        X_train, y_train, X_test, y_test = build_windows(
            X=X_scaled,
            lookback=self.config.data.lookback,
            horizon=self.config.data.horizon,
            tgt_indices=self.tgt_indices,
            withhold=self.config.data.withhold_periods,
        )

        X_train, X_test = self._apply_seasonal_weight(X_train, X_test)
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test

        self.num_features = X_train.shape[-1]  # Input dim
        self.num_targets = len(self.tgt_indices)  # Output dim
        # Minimal tf.data pipeline with shuffle/prefetch to smooth training
        self.train_dataset = (
            tf.data.Dataset.from_tensor_slices(
                (X_train.astype("float64"), y_train.astype("float64"))
            )
            .shuffle(len(X_train))
            .batch(self.config.data.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        self.val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(
            self.config.data.batch_size
        )

    def get_final_window(self) -> tuple[pd.DataFrame, pd.Series]:
        """Return the most recent lookback window and its target."""
        if hasattr(self, "X_test") and self.X_test.size:
            X_scaled = self.X_test[-1]
            y_scaled = self.y_test[-1]
        # elif hasattr(self, "X_train") and self.X_train.size:
        # X_scaled = self.X_train[-1]
        # y_scaled = self.y_train[-1]
        else:
            raise ValueError("No windowed data available; ensure create_dataset ran")

        X_unscaled = X_scaled * self.full_std + self.full_mean
        y_unscaled = y_scaled * self.target_std + self.target_mean

        X_named = pd.DataFrame(X_unscaled, columns=self.data.columns)
        y_named = pd.Series(y_unscaled, index=self.targets)
        timestamp_index = self._get_timestamp_index()
        start_idx = len(self.data) - (
            self.config.data.lookback + self.config.data.horizon
        )
        X_named.index = timestamp_index[
            start_idx : start_idx + self.config.data.lookback
        ]
        y_named.name = timestamp_index[-1]
        return X_named, y_named

    def _validate_target_type(self) -> None:
        if self.config.data.target_type not in {"full", "bs", "net_income"}:
            raise ValueError(
                f"Unsupported target_type '{self.config.data.target_type}'. "
                "Use 'full', 'bs', or 'net_income'."
            )

    def _load_or_fetch_data(self) -> pd.DataFrame:
        if self.cache_statement.exists() and not self.overwrite:
            try:
                return pd.read_parquet(self.cache_statement)
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    f"Failed to load cached statement at {self.cache_statement}"
                ) from exc
        self.company = Company(self.config.data.ticker)
        self.create_statements()
        return self.data

    def _validate_data(self) -> None:
        if self.data.empty:
            raise ValueError("Loaded financial data is empty; cannot build dataset")

        non_numeric_cols = [
            col
            for col, dtype in self.data.dtypes.items()
            if not np.issubdtype(dtype, np.number)
        ]
        if non_numeric_cols:
            raise TypeError(
                f"All features must be numeric before scaling; "
                f"found non-numeric columns: {non_numeric_cols}"
            )
        # Add new bs_identity check later with new structure
        # bs_identity(self.data, ticker=self.config.data.ticker)

    def _set_feature_index(self) -> None:
        self.feat_to_idx = {n: i for i, n in enumerate(self.data.columns.tolist())}

    def _get_timestamp_index(self) -> pd.DatetimeIndex:
        """Return a datetime index aligned to the original data index order."""
        if isinstance(self.data.index, pd.PeriodIndex):
            return self.data.index.to_timestamp()
        return pd.DatetimeIndex(self.data.index)

    def _prepare_targets(self, tar=None) -> None:
        self.targets = list(self.data.columns)
        self.tgt_indices = list(range(len(self.targets)))

        bs_keys = get_leaf_keys(balance_sheet_structure["structure"])
        bs_final_keys = [
            k for k in bs_keys if k not in balance_sheet_structure["derived"]
        ]
        self.bs_keys = bs_final_keys

        # Store FS structures with leaf keys and no __unmapped__
        self.bs_structure = {
            "Assets": [
                k
                for k in remap_financial_dataframe(
                    self.data, balance_sheet_structure["structure"]["Assets"]
                ).columns.tolist()
                if k not in balance_sheet_structure["derived"]
            ],
            "Liabilities": [
                k
                for k in remap_financial_dataframe(
                    self.data, balance_sheet_structure["structure"]["Liabilities"]
                ).columns.tolist()
                if k not in balance_sheet_structure["derived"]
            ],
            "Equity": [
                k
                for k in remap_financial_dataframe(
                    self.data, balance_sheet_structure["structure"]["Equity"]
                ).columns.tolist()
                if k not in balance_sheet_structure["derived"]
            ],
        }

        self.is_structure = {
            "Revenues": [
                k
                for k in remap_financial_dataframe(
                    self.data, income_statement_structure["structure"]["Revenues"]
                ).columns.tolist()
                if k not in income_statement_structure["derived"]
            ],
            "Expenses": [
                k
                for k in remap_financial_dataframe(
                    self.data,
                    income_statement_structure["structure"]["Cost of Revenue"],
                ).columns.tolist()
                if k not in income_statement_structure["derived"]
            ],
        }

        self.name_to_target_idx = {name: i for i, name in enumerate(self.targets)}

        asset_mappings = [
            self.name_to_target_idx[n]
            for n in get_leaf_keys(balance_sheet_structure["structure"]["Assets"])
            if n in self.name_to_target_idx
        ]
        liability_mappings = [
            self.name_to_target_idx[n]
            for n in get_leaf_keys(balance_sheet_structure["structure"]["Liabilities"])
            if n in self.name_to_target_idx
        ]
        equity_mappings = [
            self.name_to_target_idx[n]
            for n in get_leaf_keys(balance_sheet_structure["structure"]["Equity"])
            if n in self.name_to_target_idx
        ]

        self.feature_mappings = {
            "assets": asset_mappings,
            "liabilities": liability_mappings,
            "equity": equity_mappings,
        }

    def _scale_features(self) -> tuple[np.ndarray, StandardScaler]:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.data.values.astype("float64"))
        return X_scaled, scaler

    def _set_scaler_stats(self, scaler: StandardScaler) -> None:
        self.full_mean = np.asarray(scaler.mean_, dtype="float64")
        self.full_std = np.asarray(scaler.scale_, dtype="float64")
        self.target_mean = self.full_mean[self.tgt_indices]
        self.target_std = self.full_std[self.tgt_indices]

    def _apply_seasonal_weight(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        seasonal_step = self.config.data.seasonal_lag
        if self.config.data.seasonal_weight == 1.0 or seasonal_step <= 0:
            return X_train, X_test

        seasonal_indices = []
        idx = self.config.data.lookback - seasonal_step
        while idx >= 0:
            seasonal_indices.append(idx)
            idx -= seasonal_step

        if not seasonal_indices:
            return X_train, X_test

        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train[:, seasonal_indices, :] *= self.config.data.seasonal_weight
        X_test[:, seasonal_indices, :] *= self.config.data.seasonal_weight
        return X_train, X_test

    def _print_dataset_statistics(self) -> None:
        """Print comprehensive statistics about the financial dataset."""
        try:
            from colorama import Fore, Style, init

            init(autoreset=True)
        except ImportError:

            class Fore:
                CYAN = GREEN = RED = YELLOW = MAGENTA = WHITE = BLUE = ""

            class Style:
                RESET_ALL = BRIGHT = DIM = ""

        print(f"\n{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
        print(
            f"{Fore.CYAN}{Style.BRIGHT} FINANCIAL DATASET STATISTICS - \
                {self.config.data.ticker} {Style.RESET_ALL}"
        )
        print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")

        # Basic shape info
        print(f"{Fore.WHITE}{Style.BRIGHT}Dataset Shape:{Style.RESET_ALL}")
        print(
            f"  Periods (rows):     \
              {Fore.CYAN}{len(self.data)}{Style.RESET_ALL}"
        )
        print(
            f"  Features (columns): \
                {Fore.CYAN}{len(self.data.columns)}{Style.RESET_ALL}"
        )
        print(
            f"  Date range:         {Fore.CYAN}{self.data.index[0]} to \
                {self.data.index[-1]}{Style.RESET_ALL}"
        )
        print()

        # Data quality metrics
        print(f"{Fore.WHITE}{Style.BRIGHT}Data Quality:{Style.RESET_ALL}")
        total_values = self.data.shape[0] * self.data.shape[1]
        nan_count = self.data.isna().sum().sum()
        zero_count = (self.data == 0).sum().sum()
        inf_count = np.isinf(self.data.select_dtypes(include=[np.number])).sum().sum()

        print(
            f"  NaN values:         {Fore.YELLOW}{nan_count:,}{Style.RESET_ALL} \
                ({100 * nan_count / total_values:.2f}%)"
        )
        print(
            f"  Zero values:        {Fore.BLUE}{zero_count:,}{Style.RESET_ALL} \
                ({100 * zero_count / total_values:.2f}%)"
        )
        print(
            f"  Inf values:         {Fore.RED}{inf_count:,}{Style.RESET_ALL} \
                ({100 * inf_count / total_values:.2f}%)"
        )
        print()

        # Variance statistics
        print(f"{Fore.WHITE}{Style.BRIGHT}Variance Analysis:{Style.RESET_ALL}")
        stds = self.data.std()
        low_var = (stds < 1e-6).sum()
        med_var = ((stds >= 1e-6) & (stds < 1e3)).sum()
        high_var = (stds >= 1e3).sum()

        print(
            f"  Very low variance (std < 1e-6):  \
                {Fore.RED}{low_var}{Style.RESET_ALL}"
        )
        print(
            f"  Medium variance (1e-6 ≤ std < 1e3): \
                {Fore.GREEN}{med_var}{Style.RESET_ALL}"
        )
        print(
            f"  High variance (std ≥ 1e3):       \
                {Fore.YELLOW}{high_var}{Style.RESET_ALL}"
        )
        print()

        # Feature breakdown by statement type
        print(f"{Fore.WHITE}{Style.BRIGHT}Features by Statement:{Style.RESET_ALL}")
        print(
            f"  Balance Sheet features: \
                {Fore.CYAN}{len(self.bs_df.columns)}{Style.RESET_ALL}"
        )
        print(
            f"  Income Statement features: \
                {Fore.CYAN}{len(self.is_df.columns)}{Style.RESET_ALL}"
        )
        print(
            f"  Cash Flow features: \
                {Fore.CYAN}{len(self.cf_df.columns)}{Style.RESET_ALL}"
        )
        print()

        # Value ranges
        print(
            f"{Fore.WHITE}{Style.BRIGHT}Value Ranges \
                (across all features):{Style.RESET_ALL}"
        )
        print(
            f"  Global min:  {Fore.CYAN}{self.data.min().min():,.2f}{Style.RESET_ALL}"
        )
        print(
            f"  Global max:  {Fore.CYAN}{self.data.max().max():,.2f}{Style.RESET_ALL}"
        )
        print(
            f"  Global mean: {Fore.CYAN}{self.data.mean().mean():,.2f}{Style.RESET_ALL}"
        )
        print(
            f"  Global std:  {Fore.CYAN}{self.data.std().mean():,.2f}{Style.RESET_ALL}"
        )
        print()

        # Top features by variance
        print(
            f"{Fore.WHITE}{Style.BRIGHT}Top 5 Features by \
                Standard Deviation:{Style.RESET_ALL}"
        )
        top_var_features = stds.nlargest(5)
        for i, (feature, std_val) in enumerate(top_var_features.items(), 1):
            print(
                f"  {i}. {Fore.GREEN}{feature:<50}{Style.RESET_ALL} \
                    (std: {Fore.CYAN}{std_val:,.2f}{Style.RESET_ALL})"
            )
        print()

        # Bottom features by variance (excluding zeros)
        print(
            f"{Fore.WHITE}{Style.BRIGHT}Bottom 5 Features by \
                Standard Deviation (non-zero):{Style.RESET_ALL}"
        )
        bottom_var_features = stds[stds > 0].nsmallest(5)
        for i, (feature, std_val) in enumerate(bottom_var_features.items(), 1):
            print(
                f"  {i}. {Fore.YELLOW}{feature:<50}{Style.RESET_ALL} \
                    (std: {Fore.CYAN}{std_val:,.2e}{Style.RESET_ALL})"
            )
        print()

        print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")

    def create_statements(self) -> None:
        self.filings = self.company.get_filings(form="10-Q")
        self.xbrls = XBRLS.from_filings(self.filings)

        # Process each statement
        self.bs_df = self._process_statement(
            stmt=self.xbrls.statements.balance_sheet(
                max_periods=self.config.data.periods
            ),
            kind="balance sheet",
            required_structure=balance_sheet_structure,
        )
        self.is_df = self._process_statement(
            stmt=self.xbrls.statements.income_statement(
                max_periods=self.config.data.periods
            ),
            kind="income statement",
            required_structure=income_statement_structure,
        )
        self.cf_df = self._process_statement(
            stmt=self.xbrls.statements.cashflow_statement(
                max_periods=self.config.data.periods
            ),
            kind="cash flow statement",
            required_structure=cash_flow_structure,
        )

        # Align all statements on common dates (monthly periods)
        for attr in ("bs_df", "is_df", "cf_df"):
            df = getattr(self, attr)
            df.index = df.index.to_period("M")

        self.data = pd.concat(
            [self.bs_df, self.is_df, self.cf_df], axis=1, join="inner"
        )

        # Remove constant columns (std = 0)
        constant_cols = self.data.columns[self.data.std() == 0].tolist()
        if constant_cols:
            print(f"Dropping {len(constant_cols)} constant columns: {constant_cols}")
            self.data = self.data.drop(columns=constant_cols)

        # Drop rows that are mostly NaN (should be rare after inner join)
        # self.data = self.data.dropna(axis=0, thresh=int(0.8 * len(self.data.columns)))
        # Remove columns with mostly NaN
        # self.data = self.data.dropna(axis=1, thresh=int(0.8 * len(self.data.index)))
        # Replace NaNs with 0 for modelling - Or mean etc?
        self.data = self.data.fillna(0)

        self._print_dataset_statistics()

        self.data.to_parquet(self.cache_statement)

    def _process_statement(
        self,
        stmt,
        kind: str,
        required_structure: Dict,
    ) -> pd.DataFrame:
        """
        Common XBRL → tidy DataFrame pipeline.
        """
        if stmt is None:
            raise ValueError(f"No {kind} found for {self.config.data.ticker}")

        df = stmt.to_dataframe()

        # Split meta vs date cols
        meta_cols = [c for c in ("label", "concept") if c in df.columns]
        date_cols = [c for c in df.columns if c not in meta_cols]

        # concept x dates -> dates x concept
        wide = df.set_index("concept")[date_cols].T

        # Index normalisation
        wide.index = pd.to_datetime(wide.index)
        wide = wide.sort_index()
        wide.index.name = "period_end"

        # Normalise column names, collapse duplicates, clean NaNs
        wide.columns = [xbrl_to_raw(name) for name in wide.columns]
        collapsed = wide.T.groupby(level=0).first().T  # collapse duplicate concepts
        collapsed = collapsed.replace(r"^\s*$", np.nan, regex=True)

        collapsed = collapsed.dropna(axis=1, how="all")
        collapsed = collapsed.fillna(0)

        # Now parse into the defined financial structures using LLM
        # Just have to send the list of column names, the json structure

        cleaned_df, results = remove_duplicate_columns(collapsed, kind, verbose=True)

        input_columns = cleaned_df.columns.tolist()

        organised_features = self.llm_client.parse_financial_features(
            features=input_columns,
            structure_json=required_structure["structure"],
            cfg=self.config.llm,
        )

        pretty_print_full_mapping(organised_features, show_summary=True)

        # Create new DataFrame with organised columns
        mapped_df = remap_financial_dataframe(cleaned_df, organised_features)

        # Remove any leaves from derived structure that are in mapped_df
        mapped_df = mapped_df.drop(
            columns=required_structure["derived"], errors="ignore"
        )
        # # Verify derived totals
        # verification_results = verify_balance_sheet_totals(
        #     mapped_df, balance_sheet_structure
        # )

        # identities = verification_results["summary"]
        # for identity, info in identities.items():
        #     if info != "PASS":
        #         print(f"Verification failed for {identity}: {info}")

        return mapped_df


if __name__ == "__main__":
    config = Config()
    loader = EdgarDataLoader(config=config, overwrite=True)
