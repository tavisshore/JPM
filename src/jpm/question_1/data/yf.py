"""
File: yf.py
Author: Tavis Shore
Date: 05/10/2025
Description: Data ingestion pipeline for yfinance data to balance sheet database.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List

# pip install yfinance pandas numpy pyarrow tensorflow
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

from src.jpm.question_1.misc import _snake

BS_FEATS = [
    # Assets
    "Cash And Cash Equivalents",
    "Other Short Term Investments",
    "Accounts Receivable",
    "Other Receivables",
    "Inventory",
    "Other Current Assets",
    "Net PPE",
    "Investments And Advances",
    "Other Non Current Assets",
    # Liabilities
    "Accounts Payable",
    "Payables And Accrued Expenses",
    "Current Debt",
    "Other Current Liabilities",
    "Long Term Debt",
    "Other Non Current Liabilities",
    # Equity
    "Common Stock",
    "Retained Earnings",
    "Treasury Stock",
    "Other Equity Adjustments",
]

BS_FALLBACKS = {
    "Payables And Accrued Expenses": ["Payables"],
    "Current Debt": ["Current Debt And Capital Lease Obligation"],
}

IS_FEATS = [
    "Net Income",
    "Reconciled Depreciation",
    "Operating Income",
    "Total Revenue",
    "Cost Of Revenue",
    "Selling General And Administration",
    "Research And Development",
    "Tax Provision",
    "Pretax Income",
    "Other Income Expense",
    "Diluted Average Shares",
]

IS_FALLBACKS = {"Diluted Average Shares": ["Basic Average Shares"]}


def keep_cols(
    df,
    name: str = "balance",
    columns: List = BS_FEATS,
    fallbacks: Dict | None = None,
) -> pd.DataFrame:
    if "balance" in name:
        columns, fallbacks = BS_FEATS, BS_FALLBACKS
    elif "income" in name:
        columns = IS_FEATS

    colmap = {c.lower(): c for c in df.columns}

    # Use fallbacks if primaries missing
    effective_keep = []
    for k in columns:
        k_lower = k.lower()
        if k_lower in colmap:
            effective_keep.append(colmap[k_lower])
            continue

        # if fallbacks:
        for fb in fallbacks.get(k, []):
            if fb.lower() in colmap:
                effective_keep.append(colmap[fb.lower()])
                break

    out = df.loc[:, effective_keep].copy()
    out.columns = [_snake(c) for c in out.columns]
    return out


def align_quarterly(
    bs_df: pd.DataFrame, is_df: pd.DataFrame, div_s: pd.Series, quarter: bool = True
) -> pd.DataFrame:
    # Index must be chronological quarter end; inner-join on index
    bs_cols = [_snake(x) for x in BS_FEATS]
    is_cols = [_snake(x) for x in IS_FEATS]
    # Clean and align BS/IS
    bs = bs_df.sort_index().reindex(columns=bs_cols).fillna(0.0)
    is_cols = list(is_df.columns)

    shares_col = (
        "diluted_average_shares"
        if "diluted_average_shares" in is_cols
        else "basic_average_shares" if "basic_average_shares" in is_cols else None
    )

    needed_is = is_cols.copy()
    if shares_col:
        needed_is.append(shares_col)
    needed_is = set(needed_is)

    is_ = is_df.sort_index().reindex(columns=needed_is).fillna(0.0)
    df = bs.join(is_, how="inner")

    if df.empty:
        raise ValueError("No overlapping records between BS and IS.")

    # Per-share dividends per statement period
    dps_q = dividends_per_statement_period(div_s, df.index, quarter)

    # Attach per-share dividends
    df["dividends_per_share"] = dps_q.reindex(df.index).fillna(0.0)

    # Total dividends (if we have shares)
    if shares_col:
        # Shares is a quarterly average; this is a decent proxy
        df["dividends_paid_total"] = df["dividends_per_share"] * df[shares_col]
    else:
        # If no shares available, keep only per-share (can still use for RE penalty)
        df["dividends_paid_total"] = 0.0  # or np.nan if you prefer explicit missing

    # df["dividends_to_net_income_pct"] = np.where(
    #     df["net_income"] != 0,
    #     df["dividends_paid_total"] / df["net_income"],
    #     0.0
    # )

    return df


def make_sequences(
    df_merged: pd.DataFrame, steps: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    # X = past steps of [BS primitives + IS features], y = next-step BS primitives
    bs_cols = [_snake(x) for x in BS_FEATS]
    is_cols = [_snake(x) for x in IS_FEATS]
    bs_vals = df_merged[bs_cols].values.astype(np.float32)
    X_feats = df_merged[bs_cols + is_cols].values.astype(np.float32)

    X, y = [], []
    for i in range(len(df_merged) - steps):
        X.append(X_feats[i : i + steps])
        y.append(bs_vals[i + steps])  # next quarter BS primitives
    return np.array(X), np.array(y)


def scale_sequences(X, y):
    # Fit scalers on primitives only
    n_features = X.shape[-1]
    sx, sy = StandardScaler(), StandardScaler()
    X2 = X.reshape(-1, n_features)
    Xs = sx.fit_transform(X2).reshape(X.shape)
    ys = sy.fit_transform(y)
    return Xs, ys, sx, sy


def dividends_per_statement_period(
    div_s: pd.Series, stmt_index: pd.DatetimeIndex, quarter: bool = True
) -> pd.Series:
    """ """
    ds = div_s.copy()
    ds.index = pd.to_datetime(ds.index)
    if ds.index.tz is not None:
        ds.index = ds.index.tz_convert("UTC").tz_localize(None)
    ds = ds.sort_index()

    si = pd.to_datetime(stmt_index)
    if getattr(si, "tz", None) is not None:
        si = si.tz_convert("UTC").tz_localize(None)
    si = si.sort_values()

    delta = pd.DateOffset(months=3 if quarter else 12)

    out = []
    prev_end = None
    for end in si:
        start = end - delta
        if prev_end is not None and prev_end > start:
            start = prev_end
        mask = (ds.index > start) & (ds.index <= end)
        out.append(float(ds.loc[mask].sum()) if mask.any() else 0.0)
        prev_end = end

    return pd.Series(out, index=si, name="dividends_per_share")


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

    # def dividends(self, force: bool = False) -> pd.DataFrame:
    #     return self._get_or_fetch("dividends", self._load_dividends, force)

    # def splits(self, force: bool = False) -> pd.DataFrame:
    #     return self._get_or_fetch("splits", self._load_splits, force)

    def all_minimal(self, force: bool = False) -> Dict[str, pd.DataFrame]:
        return {
            "bs_q": self.balance_sheet(quarterly=True, force=force),
            "bs_a": self.balance_sheet(quarterly=False, force=force),
            "is_q": self.income_statement(quarterly=True, force=force),
            "is_a": self.income_statement(quarterly=False, force=force),
            "cf_q": self.cashflow(quarterly=True, force=force),
            "cf_a": self.cashflow(quarterly=False, force=force),
            # "prices": self.prices(force=force),
            "dividends": self.dividends(force=force),
            # "splits": self.splits(force=force),
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

    # TODO - Double check these - ChatGPT solution to ruff issues
    def _schema_for(self, name: str):
        """Return (columns, fallbacks) for known datasets; else None."""
        if "balance" in name:
            return BS_FEATS, BS_FALLBACKS
        if "income" in name:
            return IS_FEATS, IS_FALLBACKS
        if "cashflow" in name:
            return None
        return None

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
        """Keep only expected cols (with fallbacks) for known datasets."""
        schema = self._schema_for(name)
        if schema is None:
            return df
        columns, fallbacks = schema
        return keep_cols(df, name, columns=columns, fallbacks=fallbacks)

    # End TODO

    def _get_or_fetch(
        self, name: str, loader: Callable[[], pd.DataFrame], force: bool
    ) -> pd.DataFrame:
        path = self.cache_dir / f"{name}.parquet"

        if (not force) and self._valid_cache(path):
            cached = self._read_cache(path)
            if cached is not None:
                return self._apply_schema(name, cached)

        df = loader() or pd.DataFrame()

        if isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()

        # write-through cache
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
    ing = FinanceIngestor(
        "AAPL", cache_dir="/Users/tavisshore/Desktop/HK/data", ttl_days=21
    )
    quarterly = True

    all = ing.all_minimal()

    bs_q = ing.balance_sheet(quarterly)
    is_q = ing.income_statement(quarterly)  # , force=True)
    div = ing.dividends()

    # print(div)
    # print(bs_q)
    # print(is_q)
    df_merged = align_quarterly(bs_q, is_q, div, quarterly)
    print(df_merged)

    # X, y = make_sequences(df_merged, steps=4)
    # print(X.shape)
