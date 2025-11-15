"""
File: yf.py
Author: Tavis Shore
Date: 05/10/2025
Description: Data ingestion pipeline for yfinance data to balance sheet database.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# pip install yfinance pandas numpy pyarrow tensorflow
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from sklearn.preprocessing import StandardScaler

from src.jpm.question_1.data.utils import build_windows
from src.jpm.question_1.misc import _snake, get_leaf_values

balance_sheet_structure = {
    "assets": {
        "current": [
            "Cash Financial",
            "Cash Equivalents",
            # Cash And Cash Equivalents = Cash Financial + Cash Equivalents
            "Other Short Term Investments",
            "Accounts Receivable",
            "Other Receivables",
            # Receivables = Accounts Receivable + Other Receivables
            "Inventory",
            "Other Current Assets",
            # Total Current Assets = sum of above
        ],
        "non_current": [
            "Gross PPE",
            "Accumulated Depreciation",
            # Net PPE = Gross PPE - Accumulated Depreciation
            "Investments And Advances",
            "Non Current Deferred Taxes Assets",  # check this
            "Other Non Current Assets",
            # Total Non Current Assets = Net PPE + sum of above
        ],
    },
    "liabilities": {
        "current": [
            "Payables And Accrued Expenses",
            "Commercial Paper",
            "Other Current Borrowings",
            # Current Debt = Commercial Paper + Other Current Borrowings
            "Current Deferred Liabilities",
            "Other Current Liabilities",
            # Total Current Liabilities = sum of above
        ],
        "non_current": [
            "Long Term Debt And Capital Lease Obligation",
            "Tradeand Other Payables Non Current",
            "Other Non Current Liabilities",
            # Total Non Current Liabilities = sum of above
        ],
    },
    "equity": {
        "components": [
            "Common Stock",
            "Retained Earnings",  # Is this from last years cash flow? -> check
            "Other Equity Adjustments",
            # Total Equity Gross Minority Interest = sum of above
        ]
    },
}

income_statement_structure = [
    "Operating Revenue",
    "Cost Of Revenue",
    # Gross Profit = Operating Revenue - Cost Of Revenue
    "Selling General And Administration",
    "Research And Development",
    # Operating Expense = Selling General And Administration + Research And Development
    "Other Non Operating Income Expenses",
    "Tax Provision",
    # Net Income = Operating Revenue - Cost Of Revenue - Operating Expense \
    # + Other Non Operating Income Expenses - Tax Provision
]  # Net income -> Cash Flow

cash_flow_structure = {
    "operating_cash_flow": [
        "Net Income From Continuing Operations",  # IS Net Income
        "Depreciation Amortization Depletion",
        "Stock Based Compensation",
        "Other Non Cash Items",
        "Change In Working Capital",  # Could be broken down?
        # Operating Cash Flow = sum of above
    ],
    "investing_cash_flow": [
        "Net PPE Purchase And Sale",
        "Purchase Of Investment",
        "Sale Of Investment",
        "Net Other Investing Changes",
        # Investing Cash Flow = sum of above
    ],
    "financing_cash_flow": [
        "Net Long Term Debt Issuance",
        "Net Short Term Debt Issuance",
        # Net Issuance Payments Of Debt = sum of above
        "Net Common Stock Issuance",
        "Cash Dividends Paid",
        "Net Other Financing Charges",
        # Financing Cash Flow = sum of above
    ],
    # Beginning Cash Position = Previous Ending Cash Position
    # Ending Cash Position = Balance Sheets 'Cash And Cash Equivalents'
}


def _to_naive_sorted_idx(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.to_datetime(idx)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.sort_values()


def _dedupe_last(df: pd.DataFrame) -> pd.DataFrame:
    """Keep last record per statement date (handles restatements/multiples)."""
    if df.index.has_duplicates:
        return df[~df.index.duplicated(keep="last")]
    return df


def align_quarterly(
    is_df: pd.DataFrame,
    bs_df: pd.DataFrame,
    cf_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align BS + IS + CF quarterly (or yearly) and attach dividends features.

    Returns a DataFrame indexed by statement dates containing:
      - BS primitives in the order of BS_FEATS (snake_case)
      - IS features in the order of IS_FEATS (+ one shares column if present)
      - CF features in the order of CF_FEATS
      - dividends_per_share
      - dividends_paid_total (per-share × chosen shares column, else 0.0)

    Parameters
    ----------
    is_df, bs_df, cf_df : pd.DataFrame
        Quarterly statement dataframes; columns in Yahoo names.
    div_s : pd.Series
        Event series of dividends per share (index = event timestamps).
    quarter : bool
        True -> sum dividends over last 3 months; False -> last 12 months.
    """

    # ---- Normalise indexes (naive, sorted) ----
    bs_df = bs_df.copy()
    is_df = is_df.copy()
    cf_df = cf_df.copy()
    bs_df.index = _to_naive_sorted_idx(bs_df.index)
    is_df.index = _to_naive_sorted_idx(is_df.index)
    cf_df.index = _to_naive_sorted_idx(cf_df.index)

    bs_cols = get_leaf_values(balance_sheet_structure)
    is_base_cols = income_statement_structure
    cf_cols = get_leaf_values(cash_flow_structure)
    bs_cols = [_snake(c) for c in bs_cols]
    is_base_cols = [_snake(c) for c in is_base_cols]
    cf_cols = [_snake(c) for c in cf_cols]

    # ---- Snake-case once, then reindex deterministically ----
    bs_sn = bs_df.rename(columns=_snake)
    is_sn = is_df.rename(columns=_snake)
    cf_sn = cf_df.rename(columns=_snake)

    # Choose exactly ONE shares column (string), diluted -> basic fallback
    if "diluted_average_shares" in is_sn.columns:
        shares_col: Optional[str] = "diluted_average_shares"
    elif "basic_average_shares" in is_sn.columns:
        shares_col = "basic_average_shares"
    else:
        shares_col = None

    # Bit inefficient
    needed_is = list(is_base_cols)
    if shares_col:
        if shares_col not in needed_is:
            needed_is.append(shares_col)
    # needed_is = set(needed_is)

    # CF columns to keep (only those available, preserve order)
    cf_available = set(cf_sn.columns)
    needed_cf: List[str] = [c for c in cf_cols if c in cf_available]

    # Reindex (missing columns will be created and filled with zeros below)
    bs = bs_sn.reindex(columns=bs_cols).fillna(0.0)
    is_ = is_sn.reindex(columns=needed_is).fillna(0.0)
    cf = cf_sn.reindex(columns=needed_cf).fillna(0.0)

    # ---- De-duplicate BEFORE join to avoid cartesian blow-up ----
    bs = _dedupe_last(bs)
    is_ = _dedupe_last(is_)
    cf = _dedupe_last(cf)

    df = bs.join(is_, how="inner").join(cf, how="inner")
    df = _dedupe_last(df)
    if df.empty:
        raise ValueError(
            "No overlapping records between BS, IS, and CF after alignment."
        )

    return df


#  ACCOUNTING CHECKS
def check_balance_identity(
    df: pd.DataFrame, structure: dict, tol: float = 1e-6
) -> tuple[bool, pd.DataFrame]:
    asset_cols = [_snake(c) for c in get_leaf_values(structure["assets"])]
    liability_cols = [_snake(c) for c in get_leaf_values(structure["liabilities"])]
    equity_cols = [_snake(c) for c in get_leaf_values(structure["equity"])]
    A_cols = df.columns.intersection(asset_cols)
    L_cols = df.columns.intersection(liability_cols)
    E_cols = df.columns.intersection(equity_cols)
    assets = df[A_cols].sum(axis=1)
    liabilities = df[L_cols].sum(axis=1)
    equity = df[E_cols].sum(axis=1)

    diff = assets - (liabilities + equity)

    results = pd.DataFrame(
        {
            "assets": assets,
            "liabilities": liabilities,
            "equity": equity,
            "difference": diff,
        }
    )

    return (diff.abs().lt(tol).all(), results)


def check_accounting_identities(df, tol: float = 1e-4) -> None:
    acc = check_balance_identity(df, structure=balance_sheet_structure, tol=tol)

    # Assets = Liabilities + Equity
    if acc[0]:
        print("Balance sheet identity holds.")
    else:
        print("Identity violated: Assets != Liabilities + Equity")
        print(acc[1][acc[1]["difference"] != 0])

    # Add more later


class DatasetCreator:
    def __init__(
        self, tickers: Optional[List[str]] = None, quarterly: bool = True
    ) -> None:
        self.tickers = tickers
        self.quarterly = quarterly
        self.batch_size = 32
        self.lookback = 4
        self.horizon = 1
        self.create_datasets()

    def create_datasets(self):
        company_series = {}
        self.feat_stat = {}
        # Only works with single company for now - mean means and stds later?
        for ticker in self.tickers:
            self.feat_stat[ticker] = {"mean": None, "std": None}
            ing = FinanceIngestor(
                ticker,
                cache_dir="/Users/tavisshore/Desktop/HK/data",
                ttl_days=21,
                quarterly=True,
            )
            is_q = ing.income_statement(self.quarterly)
            bs_q = ing.balance_sheet(self.quarterly)
            cf_q = ing.cashflow(self.quarterly)
            df = align_quarterly(is_q, bs_q, cf_q)  # join

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df.values.astype("float32"))
            self.feat_stat[ticker]["mean"] = scaler.mean_.astype("float32")
            self.feat_stat[ticker]["std"] = scaler.scale_.astype("float32")
            df_scaled = pd.DataFrame(X_scaled, index=df.index, columns=df.columns)

            X_all = df_scaled.values
            company_series[ticker] = X_all

        # Create separate windows for individual companies + concatenate
        X_windows_list = []
        y_windows_list = []
        for _ticker, X in company_series.items():
            Xw, yw = build_windows(X, self.lookback, self.horizon)
            X_windows_list.append(Xw)
            y_windows_list.append(yw)

        X_all = np.concatenate(X_windows_list, axis=0)  # (N, lookback, F)
        y_all = np.concatenate(y_windows_list, axis=0)  # (N, F)

        self.dataset = (
            tf.data.Dataset.from_tensor_slices(
                (X_all.astype("float32"), y_all.astype("float32"))
            )
            .shuffle(len(X_all))
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        self.bs_keys = [_snake(c) for c in get_leaf_values(balance_sheet_structure)]

        name_to_idx = {n: i for i, n in enumerate(self.bs_keys)}
        asset_keys = [
            _snake(c) for c in get_leaf_values(balance_sheet_structure["assets"])
        ]
        liability_keys = [
            _snake(c) for c in get_leaf_values(balance_sheet_structure["liabilities"])
        ]
        equity_keys = [
            _snake(c) for c in get_leaf_values(balance_sheet_structure["equity"])
        ]

        self.asset_idx = [name_to_idx[k] for k in asset_keys]
        self.liability_idx = [name_to_idx[k] for k in liability_keys]
        self.equity_idx = [name_to_idx[k] for k in equity_keys]


class FinanceIngestor:
    """
    Data ingestor for balance-sheet modeling:
      - pulls statements (annual/quarterly) and prices/dividends/splits
      - transposes & sorts statements so dates are the index (ascending)
      - caches each artifact to Parquet to prevent redundant downloading
    """

    def __init__(
        self, ticker: str = "MSFT", cache_dir: str | Path = "cache", ttl_days: int = 7
    ):
        self.ticker = ticker.upper()
        self.t = yf.Ticker(self.ticker)
        self.cache_dir = Path(cache_dir) / self.ticker
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(days=ttl_days)

    def balance_sheet(
        self, quarterly: bool = True, force: bool = False
    ) -> pd.DataFrame:
        name = f"balance_sheet_{'q' if quarterly else 'a'}"
        return self._get_or_fetch(
            name, self._load_bs_q if quarterly else self._load_bs_a, force
        )

    def income_statement(
        self, quarterly: bool = True, force: bool = False
    ) -> pd.DataFrame:
        name = f"income_{'q' if quarterly else 'a'}"
        return self._get_or_fetch(
            name, self._load_is_q if quarterly else self._load_is_a, force
        )

    def cashflow(self, quarterly: bool = True, force: bool = False) -> pd.DataFrame:
        name = f"cashflow_{'q' if quarterly else 'a'}"
        return self._get_or_fetch(
            name, self._load_cf_q if quarterly else self._load_cf_a, force
        )

    def prices(
        self, period: str = "5y", interval: str = "1d", force: bool = False
    ) -> pd.DataFrame:
        name = f"prices_{period}_{interval}"

        def loader() -> pd.DataFrame:
            return self.t.history(period=period, interval=interval).sort_index()

        return self._get_or_fetch(name, loader, force)

    def all(self, force: bool = False) -> Dict[str, pd.DataFrame]:
        return {
            "bs_q": self.balance_sheet(quarterly=True, force=force),
            "bs_a": self.balance_sheet(quarterly=False, force=force),
            "is_q": self.income_statement(quarterly=True, force=force),
            "is_a": self.income_statement(quarterly=False, force=force),
            "cf_q": self.cashflow(quarterly=True, force=force),
            "cf_a": self.cashflow(quarterly=False, force=force),
        }

    # Add more as needed - are the last 3 useful?
    def _load_bs_a(self) -> pd.DataFrame:
        return self._normalise_statement(self.t.balance_sheet)

    def _load_bs_q(self) -> pd.DataFrame:
        return self._normalise_statement(self.t.quarterly_balance_sheet)

    def _load_is_a(self) -> pd.DataFrame:
        return self._normalise_statement(self.t.financials)

    def _load_is_q(self) -> pd.DataFrame:
        return self._normalise_statement(self.t.quarterly_financials)

    def _load_cf_a(self) -> pd.DataFrame:
        return self._normalise_statement(self.t.cashflow)

    def _load_cf_q(self) -> pd.DataFrame:
        return self._normalise_statement(self.t.quarterly_cashflow)

    def _load_dividends(self) -> pd.DataFrame:
        s = self.t.dividends
        df = s.to_frame("dividend") if isinstance(s, pd.Series) else pd.DataFrame(s)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
        return df.dropna(axis=0, how="all").sort_index()

    def dividends(self, force: bool = False) -> pd.Series:
        div = self._get_or_fetch("dividends", self._load_dividends, force)
        div_s = div.copy().rename(columns={"dividend": "dividend"}).squeeze()

        div_s.index = pd.to_datetime(div_s.index)
        if div_s.index.tz is not None:
            div_s.index = div_s.index.tz_convert("UTC").tz_localize(None)

        div_s = div_s.sort_index().astype(float)
        div_s.name = "Dividends Per Share Event"
        return div_s

    def _read_cache(self, path: Path) -> pd.DataFrame | None:
        """Try reading a parquet cache; return None if it fails."""
        try:
            return pd.read_parquet(path)
        except Exception:
            return None

    def _valid_cache(self, path: Path) -> bool:
        """Cache exists and is within TTL."""
        if not path.exists():
            return False
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        return (datetime.now() - mtime) <= self.ttl

    def _apply_schema(self, name: str, df: pd.DataFrame) -> pd.DataFrame:
        if "balance" in name:
            leaf_cols = get_leaf_values(balance_sheet_structure)
        elif "income" in name:
            leaf_cols = income_statement_structure
        elif "cashflow" in name:
            leaf_cols = get_leaf_values(cash_flow_structure)
        df = df[df.columns.intersection(leaf_cols)]
        return df

    def _get_or_fetch(
        self, name: str, loader: Callable[[], pd.DataFrame], force: bool
    ) -> pd.DataFrame:
        path = self.cache_dir / f"{name}.parquet"

        if (not force) and self._valid_cache(path):
            cached = self._read_cache(path)
            if cached is not None:
                return self._apply_schema(name, cached)

        df = loader()
        if df is None:
            df = pd.DataFrame()

        if isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()

        df.to_parquet(path)
        return self._apply_schema(name, df)

    @staticmethod
    def _normalise_statement(obj: Any) -> pd.DataFrame:
        """
        yfinance returns statements with rows=line items, cols=dates
        Transposing this so indices are ascending dates and cols are the items
        """
        df = pd.DataFrame(obj)
        if df.empty:
            return df
        df = df.T

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()].sort_index()

        # numeric - double check
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # drop fully-NaN rows (dates) and fully-NaN columns
        df = df.dropna(how="all")  # rows
        df = df.dropna(how="all", axis=1)  # columns
        return df


if __name__ == "__main__":
    data = DatasetCreator()
