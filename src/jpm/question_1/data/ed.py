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
from tqdm import tqdm

from jpm.config import Config
from jpm.question_1.data.credit import calculate_credit_ratios
from jpm.question_1.data.utils import (
    add_derived_columns,
    bs_identity_checker,
    drop_constants,
    drop_non_numeric_columns,
    remap_financial_dataframe,
    remove_duplicate_columns,
    standardise_rating,
    ticker_to_name,
    xbrl_to_raw,
    ytd_to_quarterly,
)
from jpm.question_1.data.vis import pretty_print_full_mapping


# SEC requires user identification via email
def _setup_edgar_identity():
    """Initialize EDGAR with user email. Only called when EDGAR is actually used."""
    email = os.getenv("EDGAR_EMAIL")
    if not email:
        raise ValueError(
            "EDGAR_EMAIL environment variable not set. "
            "Set with 'export EDGAR_EMAIL=your_email@jpm.com'"
        )
    edgar.set_identity(email)
    return email


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


class RatingsHistoryDownloader:
    """Download and process credit ratings data from ratingshistory.info.

    This class provides functionality to download Moody's Corporate Financial ratings
    data and process it into quarterly ratings suitable for joining with financial
    statement data.

    Attributes
    ----------
    BASE_URL : str
        Base URL for the ratings history website
    API_URL : str
        API endpoint URL for downloading ratings data
    config : Config
        Configuration object containing data paths and settings
    session : requests.Session
        HTTP session for making requests
    """

    BASE_URL = "https://ratingshistory.info"
    API_URL = f"{BASE_URL}/api/public"

    def __init__(self, config: Config) -> None:
        """Initialize the RatingsHistoryDownloader.

        Parameters
        ----------
        config : Config
            Configuration object containing data paths and settings
        """
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

    def download_moodys_financial(  # noqa: C901
        self,
        llm_client,
        ticker=None,
        ratings_data_path: Path = None,
        overwrite: bool = False,
    ) -> tuple[pd.DataFrame | None, bool]:
        """Download the latest Moody's Financial ratings CSV.
        1. Download latest csv from https://ratingshistory.info/
        2. For each record, add the company's corresponding ticker
        3. Make the df a quarter update of companies rating
        """
        files = self._fetch_available_files()
        filename = self._find_latest_moodys_financial(files)

        raw_data = Path(self.config.data.cache_dir) / filename
        moody_ratings = Path(self.config.data.cache_dir) / "moody_ratings.parquet"

        # Downloads the latest moodys corporate data
        updated = False
        if not raw_data.exists() or overwrite:
            if filename is None:
                print("No Moody's Financial CSV file found on ratingshistory.info.")
                return None, False
            url = f"{self.API_URL}/{quote(filename)}"
            resp = self.session.get(url, timeout=30, stream=True)
            resp.raise_for_status()
            total_size = int(resp.headers.get("content-length", 0))
            with open(raw_data, "wb") as f:
                if total_size > 0:
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
            updated = True

            # Processes moodys data for the particular ticker - returning quarterly ratings
            ratings_df = pd.read_csv(raw_data, dtype=str)
            ratings_df = ratings_df[
                ["obligor_name", "rating", "rating_type", "rating_action_date"]
            ]
            EXCLUDE_RATINGS = {"WR", "NR", "NP", "P-1", "P-2", "P-3"}
            df_clean = ratings_df[~ratings_df["rating"].isin(EXCLUDE_RATINGS)].copy()
            df_clean["rating_action_date"] = pd.to_datetime(
                df_clean["rating_action_date"]
            )
            df_clean = df_clean.dropna(subset=["rating"])
            df_clean.to_parquet(moody_ratings)
            # Ratings downloaded and processed

        if not ratings_data_path.exists() or overwrite or updated:
            name_variations = ticker_to_name(ticker, llm_client)
            df_clean = pd.read_parquet(moody_ratings)
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

            if len(df_clean) == 0:
                return None, updated

            if "quarter" not in df.columns or df.empty:
                return None, updated

            df["quarter"] = pd.to_datetime(df["quarter"]).dt.to_period("Q")
            df = df[["rating", "quarter"]]
            df = df.set_index("quarter")

            df.index = pd.PeriodIndex(df.index, freq="Q")

            df.to_parquet(ratings_data_path)
        else:
            updated = False
        df = pd.read_parquet(ratings_data_path)

        return df, updated


class EdgarData:
    """Load and process Edgar filings from SEC or cache."""

    def __init__(
        self, config: Config, overwrite: bool = False, verbose: bool = True
    ) -> None:
        """Initialize EdgarData with configuration and load/fetch data.

        Sets up cache paths, initializes LLM client, and loads financial data
        either from cache or by fetching from SEC EDGAR database.

        Parameters
        ----------
        config : Config
            Configuration object containing data paths, ticker, and settings
        overwrite : bool, optional
            If True, fetch fresh data even if cache exists, by default False
        verbose : bool, optional
            If True, print detailed processing information, by default True
        """
        self.config = config
        self.overwrite = overwrite
        self.verbose = verbose
        self.cache_statement = Path(
            f"{self.config.data.cache_dir}/statements/{self.config.data.ticker}.parquet"
        )
        self.ratings_data_path = Path(
            f"{self.config.data.cache_dir}/ratings/{self.config.data.ticker}_ratings.parquet"
        )
        self.ratings_partial_data_path = Path(
            f"{self.config.data.cache_dir}/ratings/{self.config.data.ticker}_partial_ratings.parquet"
        )
        self.cache_statement.parent.mkdir(parents=True, exist_ok=True)
        self.ratings_data_path.parent.mkdir(parents=True, exist_ok=True)

        # Lazy import to avoid circular dependency
        from jpm.question_1.clients.llm_client import LLMClient

        self.llm_client = LLMClient()
        self._load_data()

    def _load_data(self) -> None:
        """Load data from cache or fetch from SEC"""
        self._validate_target_type()
        self.data = self._load_or_fetch_data()

    def _validate_target_type(self) -> None:
        """Validate that the target_type configuration parameter is supported.

        Raises
        ------
        ValueError
            If target_type is not one of 'full', 'bs', or 'net_income'
        """
        if self.config.data.target_type not in {"full", "bs", "net_income"}:
            raise ValueError(
                f"Unsupported target_type '{self.config.data.target_type}'. "
                "Use 'full', 'bs', or 'net_income'."
            )

    def _load_or_fetch_data(self) -> pd.DataFrame:
        """Load financial data from cache or fetch fresh data from SEC.

        Attempts to load cached financial statements if available and overwrite
        is False. Otherwise, fetches fresh data from SEC EDGAR, processes it,
        and caches the results. Also retrieves and processes credit ratings.

        Returns
        -------
        pd.DataFrame
            Combined financial statement data indexed by quarter

        Raises
        ------
        RuntimeError
            If cached data exists but cannot be loaded
        ValueError
            If no data is available after filtering
        """
        if self.cache_statement.exists() and not self.overwrite:
            try:
                self.data = pd.read_parquet(self.cache_statement)
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    f"Failed to load cached statement at {self.cache_statement}"
                ) from exc
        else:
            # Setup EDGAR identity before making API calls
            _setup_edgar_identity()
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

    def get_ratings(self):
        """Training Data for Credit Rating Prediction."""
        # Download and process credit ratings
        downloader = RatingsHistoryDownloader(config=self.config)
        _ratings_df, ratings_updated = downloader.download_moodys_financial(
            self.llm_client, self.config.data.ticker, self.ratings_partial_data_path
        )  # noqa: F841

        if _ratings_df is None or _ratings_df.empty:
            print(f"No Moody's ratings found for {self.config.data.ticker}.")
            self.ratings_data = pd.DataFrame()
            return

        if not self.ratings_data_path.exists() or self.overwrite or ratings_updated:
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
        """Create and process financial statements from SEC EDGAR filings.

        Fetches 10-Q and 10-K filings for the company, extracts XBRL data,
        processes balance sheet, income statement, cash flow statement, and
        statement of equity. Removes duplicate columns across statements
        and combines them into a single DataFrame. Saves the result to cache.

        Notes
        -----
        Processing includes:
        - Converting XBRL data to tidy DataFrames
        - Mapping columns using LLM-based feature extraction
        - Dropping constant columns
        - Validating balance sheet identity
        - Converting year-to-date values to quarterly values
        - Removing duplicate columns with priority: BS > IS > CF > E
        """
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
