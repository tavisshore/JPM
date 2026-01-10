# import os
import math
import os
import re
from pathlib import Path
from urllib.parse import quote, unquote

import edgar
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from edgar import Company
from edgar.xbrl import XBRLS
from sklearn.preprocessing import StandardScaler

from jpm.question_1.config import Config
from jpm.question_1.data.structures import get_fs_struct
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

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

MONTH_ABBR = {
    1: "JAN",
    2: "FEB",
    3: "MAR",
    4: "APR",
    5: "MAY",
    6: "JUN",
    7: "JUL",
    8: "AUG",
    9: "SEP",
    10: "OCT",
    11: "NOV",
    12: "DEC",
}


def calculate_credit_ratios(df):
    def col(name):
        return (
            df[name].fillna(0) if name in df.columns else pd.Series(0, index=df.index)
        )

    ratios = pd.DataFrame(index=df.index)

    # Derived values
    total_debt = (
        col("Short-term Debt")
        + col("Current Portion of Long-term Debt")
        + col("Long-term Debt")
    )
    ebit = col("Operating Income (Loss)")
    ebitda = ebit
    total_capital = total_debt + col("Total Equity")
    cfo = col("Net Cash Provided by (Used in) Operating Activities")
    capex = col("Capital Expenditures").abs()
    fcf = cfo - capex
    quick_assets = col("Total Current Assets") - col("Inventory")

    # Interest proxy - try multiple sources
    # For banks, interest paid/received may be netted or in IS
    interest_paid_cf = col("Cash Paid for Interest")
    interest_net_cf = col("Interest Paid Net of Interest Received")
    interest_is = col("Net Interest Income (Expense)").abs()

    # Priority: 1) Cash paid for interest, 2) Net interest from CF, 3) Interest from IS
    interest_paid = interest_paid_cf.where(
        interest_paid_cf > 0, interest_net_cf.where(interest_net_cf > 0, interest_is)
    ).replace(0, float("nan"))

    # Leverage
    ratios["debt_to_equity"] = total_debt / col("Total Equity").replace(0, float("nan"))
    ratios["debt_to_assets"] = total_debt / col("Total Assets").replace(0, float("nan"))
    ratios["debt_to_capital"] = total_debt / total_capital.replace(0, float("nan"))
    ratios["debt_to_ebitda"] = total_debt / ebitda.replace(0, float("nan"))
    ratios["liabilities_to_assets"] = col("Total Liabilities") / col(
        "Total Assets"
    ).replace(0, float("nan"))

    # Coverage (using Cash Paid for Interest as proxy)
    ratios["interest_coverage"] = ebit / interest_paid
    ratios["ebitda_interest_coverage"] = ebitda / interest_paid
    ratios["cfo_interest_coverage"] = cfo / interest_paid

    # Profitability
    revenue = col("Total Revenues").replace(0, float("nan"))
    ratios["gross_margin"] = col("Gross Profit") / revenue
    ratios["operating_margin"] = ebit / revenue
    ratios["net_margin"] = col("Net Income (Loss)") / revenue
    ratios["roa"] = col("Net Income (Loss)") / col("Total Assets").replace(
        0, float("nan")
    )
    ratios["roe"] = col("Net Income (Loss)") / col("Total Equity").replace(
        0, float("nan")
    )
    ratios["roic"] = ebit / total_capital.replace(0, float("nan"))

    # Liquidity
    current_liab = col("Total Current Liabilities").replace(0, float("nan"))
    ratios["current_ratio"] = col("Total Current Assets") / current_liab
    ratios["quick_ratio"] = quick_assets / current_liab
    ratios["cash_ratio"] = col("Cash and Cash Equivalents") / current_liab
    ratios["cash_to_short_term_debt"] = col("Cash and Cash Equivalents") / col(
        "Short-term Debt"
    ).replace(0, float("nan"))

    # Cash Flow
    ratios["cfo_to_debt"] = cfo / total_debt.replace(0, float("nan"))
    ratios["fcf_to_debt"] = fcf / total_debt.replace(0, float("nan"))
    ratios["cfo_to_liabilities"] = cfo / col("Total Liabilities").replace(
        0, float("nan")
    )

    # Size
    ratios["log_assets"] = col("Total Assets").apply(
        lambda x: math.log(x) if x > 0 else float("nan")
    )
    ratios["log_revenue"] = col("Total Revenues").apply(
        lambda x: math.log(x) if x > 0 else float("nan")
    )

    # Stability
    ratios["retained_earnings_to_assets"] = col("Retained Earnings") / col(
        "Total Assets"
    ).replace(0, float("nan"))

    return ratios


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
        gleif = requests.get(
            f"https://api.gleif.org/api/v1/lei-records/{lei}", timeout=10
        )
        gleif.raise_for_status()
        name = gleif.json()["data"]["attributes"]["entity"]["legalName"]["name"]

        figi = requests.post(
            "https://api.openfigi.com/v3/search",
            json={"query": name, "securityType": "Common Stock"},
            timeout=10,
        )
        figi.raise_for_status()
        data = figi.json().get("data", [])

        if data:
            return data[0].get("ticker")
    except (requests.RequestException, KeyError, IndexError):
        pass

    return None


class RatingsHistoryDownloader:
    BASE_URL = "https://ratingshistory.info"
    API_URL = f"{BASE_URL}/api/public"

    def __init__(self, config: Config) -> None:
        self.config = config
        self.session = requests.Session()

    def _fetch_available_files(self) -> list[str]:
        """Scrape the homepage to get list of available CSV files."""
        resp = self.session.get(self.BASE_URL, timeout=30)
        resp.raise_for_status()
        # Extract CSV filenames from href attributes
        pattern = r'href="[^"]*api/public/([^"]+\.csv)"'
        matches = re.findall(pattern, resp.text)
        # URL decode the filenames
        return [unquote(m) for m in matches]

    def _find_latest_moodys_financial(self, files: list[str]) -> str | None:
        """Find the most recent Moody's Financial CSV (sorted by date prefix)."""
        moodys_financial = [
            f for f in files if "Moody's Investors Service Financial" in f
        ]
        if not moodys_financial:
            return None
        # Files are named with YYYYMMDD prefix, so lexicographic sort works
        return sorted(moodys_financial, reverse=True)[0]

    def download_moodys_financial(self) -> pd.DataFrame:
        """Download the latest Moody's Financial ratings CSV."""
        files = self._fetch_available_files()
        filename = self._find_latest_moodys_financial(files)

        output_file = Path(self.config.data.cache_dir) / filename

        if output_file.exists():
            ratings_df = pd.read_csv(output_file, dtype=str)
            return ratings_df

        if not filename:
            raise ValueError("No Moody's Financial CSV found on ratingshistory.info")

        url = f"{self.API_URL}/{quote(filename)}"
        resp = self.session.get(url, timeout=30, stream=True)
        resp.raise_for_status()

        with open(output_file, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        ratings_df = pd.read_csv(output_file, dtype=str)

        ratings_df = ratings_df[
            ["obligor_name", "rating", "rating_action_date", "legal_entity_identifier"]
        ]

        # Ratings to exclude (withdrawn, not-prime, short-term, etc.)
        EXCLUDE_RATINGS = {"WR", "NR", "NP", "P-1", "P-2", "P-3"}

        # Create numeric mapping (lower = better credit)
        # rating_to_numeric = {r: i for i, r in enumerate(MOODYS_LONG_TERM)}

        # Filter and normalize
        df_clean = ratings_df[~ratings_df["rating"].isin(EXCLUDE_RATINGS)].copy()
        # df_clean["rating_numeric"] = df_clean["rating"].map(rating_to_numeric)

        df_clean["quarter"] = pd.to_datetime(
            df_clean["rating_action_date"]
        ).dt.to_period("Q")
        df = df_clean.drop(columns=["rating_action_date"])
        # Drop any unmapped ratings
        # df_clean = df_clean.dropna(subset=["rating_numeric"])
        # df_clean["rating_numeric"] = df_clean["rating_numeric"].astype(int)

        # Get unique LEIs to minimize API calls
        unique_leis = df["legal_entity_identifier"].dropna().unique()
        lei_ticker_map = {lei: lei_to_ticker(lei) for lei in unique_leis}

        # Map tickers and drop rows without
        df["ticker"] = df["legal_entity_identifier"].map(lei_ticker_map)
        df = df.dropna(subset=["ticker"])

        # overwrite
        df.to_csv(output_file, index=False)

        return df


class EdgarData:
    """Load and process Edgar filings from SEC or cache."""

    def __init__(
        self, config: Config, overwrite: bool = False, verbose: bool = False
    ) -> None:
        self.config = config
        self.overwrite = overwrite
        self.verbose = verbose
        self.cache_statement = Path(
            f"{self.config.data.cache_dir}/{self.config.data.ticker}.parquet"
        )
        self.cache_statement.parent.mkdir(parents=True, exist_ok=True)

        # Lazy import to avoid circular dependency
        from jpm.question_1.clients.llm_client import LLMClient

        self.llm_client = LLMClient()
        # Load and prepare data
        self._load_data()

    def _load_data(self) -> None:
        """Load data from cache or fetch from SEC, then prepare targets."""
        self._validate_target_type()
        self.data = self._load_or_fetch_data()
        self._prepare_targets()

    def _validate_target_type(self) -> None:
        if self.config.data.target_type not in {"full", "bs", "net_income"}:
            raise ValueError(
                f"Unsupported target_type '{self.config.data.target_type}'. "
                "Use 'full', 'bs', or 'net_income'."
            )

    def _load_or_fetch_data(self) -> pd.DataFrame:
        if self.cache_statement.exists() and not self.overwrite:
            try:
                self.data = pd.read_parquet(self.cache_statement)
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    f"Failed to load cached statement at {self.cache_statement}"
                ) from exc
        else:
            self.company = Company(self.config.data.ticker)
            self.fy_end_month = int(self.company.fiscal_year_end[:2])
            self.fy_start_month = self.fy_end_month + 1
            if self.fy_start_month > 12:
                self.fy_start_month = 1

            self.create_statements()

        # Calculate percentage of most common value in each column
        threshold = 0.8
        cols_to_drop = [
            col
            for col in self.data.columns
            if (self.data[col].value_counts(dropna=False).iloc[0] / len(self.data))
            > threshold
        ]
        self.data = self.data.drop(columns=cols_to_drop)

        # Remove columns with very low variance
        stds = self.data.std()
        low_variance_cols = stds[stds < 1e-6].index.tolist()
        if low_variance_cols:
            if self.verbose:
                print(
                    f"Dropping {len(low_variance_cols)} low-variance columns: \
                    {low_variance_cols}"
                )
            self.data = self.data.drop(columns=low_variance_cols)

        return self.data

    def _get_timestamp_index(self) -> pd.DatetimeIndex:
        """Return a datetime index aligned to the original data index order."""
        if isinstance(self.data.index, pd.PeriodIndex):
            return self.data.index.to_timestamp()
        return pd.DatetimeIndex(self.data.index)

    def _prepare_targets(self, tar=None) -> None:
        self.targets = list(self.data.columns)
        self.tgt_indices = list(range(len(self.targets)))

        fs_structure = get_fs_struct("all")
        balance_sheet_structure = fs_structure["balance_sheet"]
        income_statement_structure = fs_structure["income_statement"]

        bs_keys = get_leaf_keys(balance_sheet_structure["prediction_structure"])
        bs_final_keys = [
            k for k in bs_keys if k not in balance_sheet_structure["drop_summations"]
        ]
        self.bs_keys = bs_final_keys

        # Store FS structures with leaf keys and no __unmapped__
        self.bs_structure = {
            "Assets": [
                k
                for k in remap_financial_dataframe(
                    self.data, balance_sheet_structure["prediction_structure"]["Assets"]
                ).columns.tolist()
                if k not in balance_sheet_structure["drop_summations"]
            ],
            "Liabilities": [
                k
                for k in remap_financial_dataframe(
                    self.data,
                    balance_sheet_structure["prediction_structure"]["Liabilities"],
                ).columns.tolist()
                if k not in balance_sheet_structure["drop_summations"]
            ],
            "Equity": [
                k
                for k in remap_financial_dataframe(
                    self.data, balance_sheet_structure["prediction_structure"]["Equity"]
                ).columns.tolist()
                if k not in balance_sheet_structure["drop_summations"]
            ],
        }

        self.is_structure = {
            "Revenues": [
                k
                for k in remap_financial_dataframe(
                    self.data,
                    income_statement_structure["prediction_structure"]["Revenues"],
                ).columns.tolist()
                if k not in income_statement_structure["drop_summations"]
            ],
            "Expenses": [
                k
                for k in remap_financial_dataframe(
                    self.data,
                    income_statement_structure["prediction_structure"][
                        "Cost of Revenue"
                    ],
                ).columns.tolist()
                if k not in income_statement_structure["drop_summations"]
            ],
        }

        self.name_to_target_idx = {name: i for i, name in enumerate(self.targets)}

        asset_mappings = [
            self.name_to_target_idx[n]
            for n in get_leaf_keys(
                balance_sheet_structure["prediction_structure"]["Assets"]
            )
            if n in self.name_to_target_idx
        ]
        liability_mappings = [
            self.name_to_target_idx[n]
            for n in get_leaf_keys(
                balance_sheet_structure["prediction_structure"]["Liabilities"]
            )
            if n in self.name_to_target_idx
        ]
        equity_mappings = [
            self.name_to_target_idx[n]
            for n in get_leaf_keys(
                balance_sheet_structure["prediction_structure"]["Equity"]
            )
            if n in self.name_to_target_idx
        ]

        self.feature_mappings = {
            "assets": asset_mappings,
            "liabilities": liability_mappings,
            "equity": equity_mappings,
        }

    def get_ratings(self):
        """Training Data for Credit Rating Prediction."""
        # Download and process credit ratings
        downloader = RatingsHistoryDownloader(config=self.config)
        _ratings_df = downloader.download_moodys_financial()  # noqa: F841

        # Calculate ratios from self.data and create separate df
        # Usage:
        df = add_derived_columns(self.data)
        # full_df = pd.concat([financial_df, ratios_df], axis=1)

        # 1. Total Revenues should be positive — likely a sign convention issue
        # df["Total Revenues"] = df["Total Revenues"].abs()

        # 2. Convert Interest Expense to numeric (it's stored as object)
        # df["Interest Expense"] = pd.to_numeric(
        #     df["Interest Expense"], errors="coerce"
        # )

        # 3. Check your data pipeline — these columns may not be populating correctly
        # empty_cols = df.columns[df.isna().all()].tolist()
        # print(f"Empty columns: {empty_cols}")

        # print(df["Interest Expense"].describe())
        # print(df["Short-term Debt"].describe())
        # print(df["Total Revenues"].describe())
        # print()
        print("\n=== BALANCE SHEET COLUMNS ===")
        bs_cols = [col for col in df.columns if col in self.bs_df.columns]
        print(f"BS columns: {len(bs_cols)}")

        print("\n=== INCOME STATEMENT COLUMNS ===")
        is_cols = [col for col in df.columns if col in self.is_df.columns]
        print(f"IS columns: {len(is_cols)}")
        print(is_cols)

        print("\n=== CASH FLOW COLUMNS ===")
        cf_cols = [col for col in df.columns if col in self.cf_df.columns]
        print(f"CF columns: {len(cf_cols)}")
        print(cf_cols)

        print("\n=== KEY VALUES FROM LAST QUARTER ===")
        print(
            df[
                [
                    "Operating Income (Loss)",
                    "Cash Paid for Interest",
                    "Net Cash Provided by (Used in) Operating Activities",
                    "Total Revenues",
                    "Cost of Goods Sold",
                ]
            ].iloc[-1]
        )

        print("\n=== CHECKING FOR INTEREST-RELATED COLUMNS ===")
        interest_cols = [col for col in df.columns if "interest" in col.lower()]
        print(f"Interest columns found: {interest_cols}")
        if interest_cols:
            print("\nLatest quarter values:")
            print(df[interest_cols].iloc[-1])

            # Show which source would be used for interest calculation
            cf_interest = df.get(
                "Cash Paid for Interest", pd.Series(0, index=df.index)
            ).iloc[-1]
            net_interest = df.get(
                "Interest Paid Net of Interest Received", pd.Series(0, index=df.index)
            ).iloc[-1]
            is_interest = (
                df.get("Net Interest Income (Expense)", pd.Series(0, index=df.index))
                .abs()
                .iloc[-1]
            )

            print("\nInterest source priority check:")
            print(f"  1. Cash Paid for Interest (CF): {cf_interest}")
            print(f"  2. Interest Paid Net (CF): {net_interest}")
            print(f"  3. Net Interest Income/Expense (IS): {is_interest}")

            if cf_interest > 0:
                print(f"  → Using Cash Flow interest: {cf_interest}")
            elif net_interest > 0:
                print(f"  → Using Net CF interest: {net_interest}")
            elif is_interest > 0:
                print(f"  → Using Income Statement interest: {is_interest}")
            else:
                print("  → WARNING: No interest data available!")

        print()
        ratios_df = calculate_credit_ratios(df)
        print("\n=== CALCULATED RATIOS (Latest Quarter) ===")
        print(ratios_df.head(1).T)
        print()
        breakpoint()

    def create_statements(self) -> None:
        self.filings = self.company.get_filings(form=["10-Q", "10-K"])
        self.xbrls = XBRLS.from_filings(self.filings)

        # Process each statement
        self.bs_df = self._process_statement(
            stmt=self.xbrls.statements.balance_sheet(
                max_periods=self.config.data.periods
            ),
            kind="balance_sheet",
        )

        self.is_df = self._process_statement(
            stmt=self.xbrls.statements.income_statement(
                max_periods=self.config.data.periods
            ),
            kind="income_statement",
        )

        self.cf_df = self._process_statement(
            stmt=self.xbrls.statements.cashflow_statement(
                max_periods=self.config.data.periods
            ),
            kind="cash_flow",
        )

        # Remove duplicate columns with priority: BS > IS > CF
        # TODO: Check this is the ideal order
        bs_cols = set(self.bs_df.columns)
        is_cols = set(self.is_df.columns)
        cf_cols = set(self.cf_df.columns)

        # Remove IS columns that are also in BS
        self.is_df = self.is_df.drop(columns=is_cols.intersection(bs_cols))
        # Remove CF columns that are also in BS or IS
        self.cf_df = self.cf_df.drop(
            columns=cf_cols.intersection(bs_cols.union(is_cols))
        )
        # Combine all statements
        self.data = pd.concat(
            [self.bs_df, self.is_df, self.cf_df], axis=1, join="inner"
        )

        # Get credit ratings, calculate ratios, and add col to self.data
        # self.get_ratings()

        if self.verbose:
            for col in self.data.columns:
                print(col)
            print("\nFiltering empty values:")

        # Remove constant columns (std = 0)
        constant_cols = self.data.columns[self.data.std() == 0].tolist()
        if constant_cols:
            if self.verbose:
                print(
                    f"Dropping {len(constant_cols)} constant columns: {constant_cols}"
                )
            self.data = self.data.drop(columns=constant_cols)

        # Drop rows that are mostly NaN (should be rare after inner join)
        # self.data = self.data.dropna(axis=0, thresh=int(0.8 * len(self.data.columns)))
        # Remove columns with mostly NaN
        # self.data = self.data.dropna(axis=1, thresh=int(0.8 * len(self.data.index)))
        # Replace NaNs with 0 for modelling - Or mean etc?
        self.data = self.data.fillna(0)

        if self.verbose:
            for col in self.data.columns:
                print(col)

        self.data.to_parquet(self.cache_statement)

    def _process_statement(
        self,
        stmt,
        kind: str,
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

        freq = f"Q-{MONTH_ABBR[self.fy_end_month]}"  # 'Q-SEP'
        wide = wide.sort_index(ascending=True)  # oldest first
        start_period = pd.to_datetime(wide.index[0]).to_period(freq)
        wide.index = pd.period_range(start=start_period, periods=len(wide), freq=freq)[
            ::-1
        ]
        wide.index.name = "quarter"

        # Normalise column names, collapse duplicates, clean NaNs
        wide.columns = [xbrl_to_raw(name) for name in wide.columns]

        collapsed = wide.T.groupby(level=0).first().T  # collapse duplicate concepts
        collapsed = collapsed.replace(r"^\s*$", np.nan, regex=True)

        collapsed = collapsed.dropna(axis=1, how="all")
        collapsed = collapsed.fillna(0)

        cleaned_df, results = remove_duplicate_columns(
            collapsed, kind, verbose=self.verbose
        )

        input_columns = cleaned_df.columns.tolist()

        # Cache LLM mapping results
        cache_dir = self.config.data.cache_dir
        ticker = self.config.data.ticker
        kind_slug = kind.replace(" ", "_")
        features_cache_path = Path(f"{cache_dir}/{ticker}_{kind_slug}_features.json")
        if features_cache_path.exists() and not self.overwrite:
            organised_features = self.llm_client.load_cached_features(
                features_cache_path
            )
        else:
            organised_features = self.llm_client.parse_financial_features(
                features=input_columns,
                statement_type=kind,
                cfg=self.config.llm,
            )
            self.llm_client.save_features_to_cache(
                organised_features, features_cache_path
            )

        if self.verbose:
            pretty_print_full_mapping(organised_features, show_summary=True)

        # Create new DataFrame with organised columns
        mapped_df = remap_financial_dataframe(cleaned_df, organised_features)

        # Remove any leaves from derived structure that are in mapped_df
        mapped_df = mapped_df.drop(
            columns=get_fs_struct(kind)["drop_summations"], errors="ignore"
        )

        # For IS and CF, subtract values to obtain quarterly only (from cumulative)
        if kind in {"income statement", "cash flow statement"}:
            mapped_df = ytd_to_quarterly(mapped_df)

        return mapped_df


class EdgarDataset:
    """Prepares train/val datasets from EdgarData for model training."""

    def __init__(
        self, edgar_data: "EdgarData", target: str = "lstm", verbose: bool = False
    ) -> None:
        """
        Initialize dataset from EdgarData.

        Parameters:
        -----------
        edgar_data : EdgarData
            Instance of EdgarData with loaded and processed financial data
        target : str
            Target model type ("lstm" or "xgboost")
        verbose : bool
            Whether to print detailed information during processing
        """
        self.edgar_data = edgar_data
        self.config = edgar_data.config
        self.target = target
        self.verbose = verbose

        # Copy necessary attributes from EdgarData
        self.data = edgar_data.data
        self.tgt_indices = edgar_data.tgt_indices
        self.targets = edgar_data.targets

        # TODO Starting from now, go back quarterly - remove data after a break
        # print(f"Searching for break in data for {self.config.data.ticker}...")
        # self.data = self.data.sort_index(ascending=False)
        # quarterly_index = self.data.index.asfreq("Q")
        # expected = pd.period_range(
        #     end=quarterly_index[0], periods=len(self.data), freq="Q"
        # )[::-1]
        # mask = (self.data.index == expected).cumprod().astype(bool)
        # self.data = self.data[mask]

        # Prepare the dataset
        self._prepare_dataset()

    def _prepare_dataset(self) -> None:
        """Prepare scaled features and create train/val datasets."""
        self._set_feature_index()

        X_scaled, scaler = self._scale_features()
        self._set_scaler_stats(scaler)

        if self.target == "lstm":
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

            self.val_dataset = tf.data.Dataset.from_tensor_slices(
                (X_test, y_test)
            ).batch(self.config.data.withhold_periods)
        elif self.target == "xgboost":
            # TODO: Implement xgboost dataset preparation
            pass

    def _set_feature_index(self) -> None:
        """Create feature name to index mapping."""
        self.feat_to_idx = {n: i for i, n in enumerate(self.data.columns.tolist())}

    def _scale_features(self) -> tuple[np.ndarray, StandardScaler]:
        """Scale features using StandardScaler."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.data.values.astype("float64"))
        return X_scaled, scaler

    def _set_scaler_stats(self, scaler: StandardScaler) -> None:
        """Store scaler statistics for later use."""
        self.full_mean = np.asarray(scaler.mean_, dtype="float64")
        self.full_std = np.asarray(scaler.scale_, dtype="float64")
        self.target_mean = self.full_mean[self.tgt_indices]
        self.target_std = self.full_std[self.tgt_indices]

    def _apply_seasonal_weight(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply seasonal weighting to training and test data."""
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


if __name__ == "__main__":
    config = Config()
    data = EdgarData(config=config, overwrite=False, verbose=False)
    # dataset = EdgarDataset(edgar_data=data, target="lstm", verbose=False)
    # print(data.data.head(1).T)
