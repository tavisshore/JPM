# import os
import os
import re
from pathlib import Path
from urllib.parse import quote, unquote

import edgar
import numpy as np
import pandas as pd
import requests
from edgar import Company
from edgar.xbrl import XBRLS

from jpm.question_1.config import Config
from jpm.question_1.data.credit import calculate_credit_ratios
from jpm.question_1.data.structures import get_fs_struct
from jpm.question_1.data.utils import (
    add_derived_columns,
    bs_identity_checker,
    drop_constants,
    drop_non_numeric_columns,
    remove_duplicate_columns,
    standardise_rating,
    ticker_to_name,
    xbrl_to_raw,
    ytd_to_quarterly,
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


def _extract_leaf_mappings(mapping):
    """Extract only leaf-level column mappings from nested mapping structure."""
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


def _is_mutually_exclusive(df, cols):
    """
    Check if columns are mutually exclusive (alternate taxonomies).
    Returns True if columns never have non-zero values simultaneously.
    """
    if len(cols) <= 1:
        return False

    # Count rows where multiple columns are non-zero
    non_zero = (df[cols] != 0) & (df[cols].notna())
    simultaneous_nonzero = (non_zero.sum(axis=1) > 1).sum()

    # If <5% of rows have multiple non-zero values, treat as mutually exclusive
    threshold = len(df) * 0.05
    return simultaneous_nonzero < threshold


def _map_single_column(df, new_col, old_cols):
    """Map a single column based on source columns."""
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
            f for f in files if "Moody's Investors Service Corporate" in f
        ]
        if not moodys_financial:
            return None
        # Files are named with YYYYMMDD prefix, so lexicographic sort works
        return sorted(moodys_financial, reverse=True)[0]

    def download_moodys_financial(self, llm_client, ticker=None) -> pd.DataFrame:
        """Download the latest Moody's Financial ratings CSV.
        1. Download latest csv from https://ratingshistory.info/
        2. For each record, add the company's corresponding ticker
        3. Make the df a quarter update of companies rating
        """
        files = self._fetch_available_files()
        filename = self._find_latest_moodys_financial(files)

        raw_data = Path(self.config.data.cache_dir) / filename

        # Downloads the latest moodys corporate data
        if not raw_data.exists():
            url = f"{self.API_URL}/{quote(filename)}"
            resp = self.session.get(url, timeout=30, stream=True)
            resp.raise_for_status()
            total_size = int(resp.headers.get("content-length", 0))
            with open(raw_data, "wb") as f:
                if total_size > 0:
                    from tqdm import tqdm

                    with tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc=f"Downloading {filename}",
                    ) as pbar:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
                else:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)

        # Processes moodys data for the particular ticker - returning quarterly ratings
        # if not output_file.exists():
        ratings_df = pd.read_csv(raw_data, dtype=str)
        ratings_df = ratings_df[
            ["obligor_name", "rating", "rating_type", "rating_action_date"]
        ]
        EXCLUDE_RATINGS = {"WR", "NR", "NP", "P-1", "P-2", "P-3"}
        df_clean = ratings_df[~ratings_df["rating"].isin(EXCLUDE_RATINGS)].copy()
        df_clean["rating_action_date"] = pd.to_datetime(df_clean["rating_action_date"])
        df_clean = df_clean.dropna(subset=["rating"])

        name_variations = ticker_to_name(ticker, llm_client)
        df_clean = df_clean[df_clean["obligor_name"].isin(name_variations)]
        df_clean["rating"] = df_clean["rating"].apply(standardise_rating)
        # Try organisational ratings, otherwise fallback to instrument
        df_ratings = df_clean[df_clean["rating_type"] == "Organization"]
        if df_ratings.empty:
            df_ratings = df_clean[df_clean["rating_type"] == "Instrument"]
        df_clean = df_ratings.drop(columns=["rating_type"])

        # Now convert into quarterly ratings to join with FS data
        results = []
        df_clean = df_clean.sort_values("rating_action_date").reset_index(drop=True)
        for i in range(len(df_clean)):
            start_date = df_clean.loc[i, "rating_action_date"]
            rating = df_clean.loc[i, "rating"]

            # End date is either next rating change or today
            if i < len(df_clean) - 1:
                end_date = df_clean.loc[i + 1, "rating_action_date"]
            else:
                end_date = pd.Timestamp.now()

            quarters = pd.date_range(start=start_date, end=end_date, freq="QE")
            for quarter in quarters:
                results.append(
                    {
                        "rating": rating,
                        "quarter": quarter,
                    }
                )

        df = pd.DataFrame(results)
        df["quarter"] = pd.to_datetime(df["quarter"]).dt.to_period("Q")
        df = df[["rating", "quarter"]]
        df = df.set_index("quarter")

        df.index = pd.PeriodIndex(df.index, freq="Q")
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
            f"{self.config.data.cache_dir}/statements/{self.config.data.ticker}.parquet"
        )
        self.ratings_data_path = Path(
            f"{self.config.data.cache_dir}/ratings/{self.config.data.ticker}_ratings.parquet"
        )
        self.cache_statement.parent.mkdir(parents=True, exist_ok=True)
        self.ratings_data_path.parent.mkdir(parents=True, exist_ok=True)

        # Lazy import to avoid circular dependency
        from jpm.question_1.clients.llm_client import LLMClient

        self.llm_client = LLMClient()
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

        # Get credit ratings, calculate ratios -> separate attr and csv
        self.get_ratings()

        if len(self.data) == 0:
            raise ValueError(
                f"No data available for {self.config.data.ticker}. "
                "The data may have been filtered out entirely."
            )

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
        if not self.ratings_data_path.exists() or self.overwrite:
            # Download and process credit ratings
            downloader = RatingsHistoryDownloader(config=self.config)
            _ratings_df = downloader.download_moodys_financial(
                self.llm_client, self.config.data.ticker
            )  # noqa: F841

            # Select only the relevant company's data
            # Calculate ratios from self.data and add to ratings_df
            df = add_derived_columns(self.data)

            df.index = df.index.asfreq("Q-DEC")
            _ratings_df.index = _ratings_df.index.asfreq("Q-DEC")

            df_combined = pd.merge(
                df, _ratings_df, left_index=True, right_index=True, how="inner"
            )

            self.ratings_data = calculate_credit_ratios(df_combined)
            self.ratings_data.to_parquet(self.ratings_data_path)
        else:
            self.ratings_data = pd.read_parquet(self.ratings_data_path)

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
        self.bs_df = drop_constants(self.bs_df, verbose=self.verbose)

        self.is_df = self._process_statement(
            stmt=self.xbrls.statements.income_statement(
                max_periods=self.config.data.periods
            ),
            kind="income_statement",
        )
        self.is_df = drop_constants(self.is_df, verbose=self.verbose)

        self.cf_df = self._process_statement(
            stmt=self.xbrls.statements.cashflow_statement(
                max_periods=self.config.data.periods
            ),
            kind="cash_flow",
        )
        self.cf_df = drop_constants(self.cf_df, verbose=self.verbose)

        self.eq_df = self._process_statement(
            stmt=self.xbrls.statements.statement_of_equity(
                max_periods=self.config.data.periods
            ),
            kind="equity",
        )
        self.eq_df = drop_constants(self.eq_df, verbose=self.verbose)

        # Remove duplicate columns with priority: BS > IS > CF > E
        bs_cols = set(self.bs_df.columns)
        is_cols = set(self.is_df.columns)
        cf_cols = set(self.cf_df.columns)
        eq_cols = set(self.eq_df.columns)

        # Remove IS columns that are also in BS
        self.is_df = self.is_df.drop(columns=is_cols.intersection(bs_cols))
        # Remove CF columns that are also in BS or IS
        self.cf_df = self.cf_df.drop(
            columns=cf_cols.intersection(bs_cols.union(is_cols))
        )
        # Remove E columns that are also in BS, IS, or CF
        self.eq_df = self.eq_df.drop(
            columns=eq_cols.intersection(bs_cols.union(is_cols).union(cf_cols))
        )
        # Combine all statements
        self.data = pd.concat(
            [self.bs_df, self.is_df, self.cf_df, self.eq_df], axis=1, join="inner"
        )

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
        collapsed = drop_non_numeric_columns(collapsed)

        cleaned_df = remove_duplicate_columns(collapsed, verbose=self.verbose)
        input_columns = cleaned_df.columns.tolist()

        # Cache LLM mapping results
        cache_dir = self.config.data.cache_dir
        ticker = self.config.data.ticker
        kind_slug = kind.replace(" ", "_")
        features_cache_path = Path(
            f"{cache_dir}/features/{ticker}_{kind_slug}_features.json"
        )
        features_cache_path.parent.mkdir(parents=True, exist_ok=True)

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
            pretty_print_full_mapping(
                organised_features, show_summary=True, statement_type=kind
            )

        # Create new DataFrame with organised columns
        mapped_df = remap_financial_dataframe(cleaned_df, organised_features)

        if kind == "balance_sheet":
            # Drops columns that violate BS identity
            mapped_df = bs_identity_checker(df=mapped_df, mappings=organised_features)
        else:
            # Subtract values to obtain quarterly only (from cumulative)
            exclude = []
            if kind == "equity":
                exclude = [
                    "Accumulated Other Comprehensive Income",
                    "Additional Paid-in Capital",
                    "Common Stock",
                    "Retained Earnings",
                    "Total Equity",
                    "Treasury Stock",
                ]

            mapped_df = ytd_to_quarterly(mapped_df, exclude_columns=exclude)

        return mapped_df


if __name__ == "__main__":
    config = Config()
    data = EdgarData(config=config, overwrite=False, verbose=True)
    print(data.data.sample(6).T)
    print()
    print(data.ratings_data.sample(6).T)
    # Getting ratings
    # dataset = EdgarDataset(edgar_data=data, target="lstm", verbose=False)
    # print(data.data.head(1).T)
