"""
File: yf.py
Author: Tavis Shore
Date: 05/10/2025
Description: Data ingestion pipeline for yfinance data to balance sheet database.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict

import pandas as pd
import yfinance as yf


class FinanceIngestor:
    """
    Data ingestor for balance-sheet modeling:
      - pulls statements (annual/quarterly) and prices/dividends/splits
      - transposes & sorts statements so dates are the index (ascending)
      - caches each artifact to Parquet to prevent redundant downloading
    """

    def __init__(self, ticker: str, cache_dir: str | Path = "cache", ttl_days: int = 1):
        self.ticker = ticker.upper()
        self.t = yf.Ticker(self.ticker)
        self.cache_dir = Path(cache_dir) / self.ticker
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(days=ttl_days)

    def balance_sheet(
        self, quarterly: bool = False, force: bool = False
    ) -> pd.DataFrame:
        name = f"balance_sheet_{'q' if quarterly else 'a'}"
        return self._get_or_fetch(
            name, self._load_bs_q if quarterly else self._load_bs_a, force
        )

    def income(self, quarterly: bool = False, force: bool = False) -> pd.DataFrame:
        name = f"income_{'q' if quarterly else 'a'}"
        return self._get_or_fetch(
            name, self._load_is_q if quarterly else self._load_is_a, force
        )

    def cashflow(self, quarterly: bool = False, force: bool = False) -> pd.DataFrame:
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

    def dividends(self, force: bool = False) -> pd.DataFrame:
        return self._get_or_fetch("dividends", self._load_dividends, force)

    def splits(self, force: bool = False) -> pd.DataFrame:
        return self._get_or_fetch("splits", self._load_splits, force)

    def all_minimal(self, force: bool = False) -> Dict[str, pd.DataFrame]:
        return {
            "bs_q": self.balance_sheet(quarterly=True, force=force),
            "bs_a": self.balance_sheet(quarterly=False, force=force),
            "is_q": self.income(quarterly=True, force=force),
            "is_a": self.income(quarterly=False, force=force),
            "cf_q": self.cashflow(quarterly=True, force=force),
            "cf_a": self.cashflow(quarterly=False, force=force),
            "prices": self.prices(force=force),
            "dividends": self.dividends(force=force),
            "splits": self.splits(force=force),
        }

    # loaders - Add more as needed
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

    def _load_splits(self) -> pd.DataFrame:
        s = self.t.splits
        df = s.to_frame("split_ratio") if isinstance(s, pd.Series) else pd.DataFrame(s)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
        return df.dropna(axis=0, how="all").sort_index()

    def _get_or_fetch(
        self, name: str, loader: Callable[[], pd.DataFrame], force: bool
    ) -> pd.DataFrame:
        """
        Caches to Parquet in cache_dir/ticker/name.parquet with TTL.
        """
        path = self.cache_dir / f"{name}.parquet"
        if path.exists() and not force:
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            if datetime.now() - mtime <= self.ttl:
                try:
                    return pd.read_parquet(path)
                except Exception:
                    pass  # refetch if cache is corrupt

        df = loader()
        if df is None:
            df = pd.DataFrame()

        if isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()
        df.to_parquet(path)
        return df

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
        "AAPL", cache_dir="/Users/tavisshore/Desktop/HK/data", ttl_days=0
    )
    bs_q = ing.all_minimal()
    print(bs_q["bs_q"].head())
    print(bs_q["bs_q"].columns)
