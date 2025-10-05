"""
File: balance_sheet.py
Author: Tavis Shore
Date: 05/10/2025
Description:
  Takes a yfinance balance-sheet frame, maps it to a canonical schema,
  re-computes totals from components, and enforces accounting identities
  with a deterministic repair order.

Enforcing:
1. Net PPE = Gross PPE - Accumulated Depreciation
2. Current Assets = Cash + ST Investments + AR + Inventory + Other CA
3. Noncurrent Assets = Net PPE + Investments & Advances + Other NCA
4. Total Assets = Current Assets + Noncurrent Assets
5. Current Liabilities = AP + Tax Payable + Current Debt (+ other CL)
6. Noncurrent Liabilities = LT Debt (+ other NCL)
7. Total Liabilities = CL + NCL
8. Total Equity (incl. MI) = Common Equity + Retained Earnings
   + Other Equity Adj (+ MI if provided)
9. Assets = Liabilities + Equity
10. Total Debt = Current Debt + Long-Term Debt (+ Other Borrowings
    + Lease Obligations if present)
11. Net Debt = Total Debt - (Cash & Equivalents + ST Investments)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _nan_sum(values) -> float:
    return np.nansum(list(values))


@dataclass
class BSCheckResult:
    ok: bool
    violations: Dict[str, float]  # name -> absolute error
    computed: Dict[str, float]  # values the engine computed/overrode


class BalanceSheetBaseline:
    """
    Baseline mapper/validator/repairer for yfinance balance sheets.

    - Input: My frame from yf.py
    - Output: canonical schema DataFrame with identities enforced.
    - Strategy: compute totals from components; use residuals only
      for the last identity.
    """

    # numerical tolerance for equality checks - NOTE check what standard is
    eps: float = 1e-6

    CANON_COLS = [
        # Assets
        "Cash",
        "ShortTermInvestments",
        "CashAndSTI",
        "AccountsReceivable",
        "Inventory",
        "OtherCurrentAssets",
        "CurrentAssets",
        "GrossPPE",
        "AccumDepreciation",
        "NetPPE",
        "InvestmentsAndAdvances",
        "OtherNonCurrentAssets",
        "NonCurrentAssets",
        "TotalAssets",
        # Liabilities
        "AccountsPayable",
        "IncomeTaxPayable",
        "TotalTaxPayable",
        "CurrentDebt",
        "OtherCurrentBorrowings",
        "OtherCurrentLiabilities",
        "CurrentLiabilities",
        "LongTermDebt",
        "LTDebtAndLeases",
        "OtherNonCurrentLiabilities",
        "NonCurrentLiabilities",
        "TotalLiabilities",
        # Equity
        "CommonStockEquity",
        "RetainedEarnings",
        "OtherEquityAdjustments",
        "TotalEquityInclMI",
        "StockholdersEquity",
        "TotalCapitalization",
        # Debt-derived
        "TotalDebt",
        "NetDebt",
        # Working capital / tangibles
        "WorkingCapital",
        "TangibleBookValue",
        "NetTangibleAssets",
        # Convenience
        "MinorityInterestEstimated",
    ]

    # yfinance -> canonical names
    YF_TO_CANON = {
        # Assets
        "Cash And Cash Equivalents": "Cash",
        "Cash Equivalents": "Cash",  # if split, both roll into Cash
        "Cash Financial": "Cash",
        "Other Short Term Investments": "ShortTermInvestments",
        "Cash Cash Equivalents And Short Term Investments": "CashAndSTI",
        "Accounts Receivable": "AccountsReceivable",
        "Receivables": "AccountsReceivable",  # broader; used if AR missing
        "Inventory": "Inventory",
        "Other Current Assets": "OtherCurrentAssets",
        "Current Assets": "CurrentAssets",
        "Gross PPE": "GrossPPE",
        "Accumulated Depreciation": "AccumDepreciation",
        "Net PPE": "NetPPE",
        "Investments And Advances": "InvestmentsAndAdvances",
        "Investmentin Financial Assets": "InvestmentsAndAdvances",
        "Other Non Current Assets": "OtherNonCurrentAssets",
        "Total Non Current Assets": "NonCurrentAssets",
        "Total Assets": "TotalAssets",
        # Liabilities
        "Accounts Payable": "AccountsPayable",
        "Income Tax Payable": "IncomeTaxPayable",
        "Total Tax Payable": "TotalTaxPayable",
        "Current Debt": "CurrentDebt",
        "Other Current Borrowings": "OtherCurrentBorrowings",
        "Other Current Liabilities": "OtherCurrentLiabilities",
        "Current Liabilities": "CurrentLiabilities",
        "Long Term Debt": "LongTermDebt",
        "Long Term Debt And Capital Lease Obligation": "LTDebtAndLeases",
        "Other Non Current Liabilities": "OtherNonCurrentLiabilities",
        "Total Non Current Liabilities Net Minority Interest": "NonCurrentLiabilities",
        "Total Liabilities Net Minority Interest": "TotalLiabilities",
        # Equity
        "Common Stock Equity": "CommonStockEquity",
        "Retained Earnings": "RetainedEarnings",
        "Other Equity Adjustments": "OtherEquityAdjustments",
        "Total Equity Gross Minority Interest": "TotalEquityInclMI",
        "Stockholders Equity": "StockholdersEquity",
        "Total Capitalization": "TotalCapitalization",
        # Debt summaries
        "Total Debt": "TotalDebt",
        "Net Debt": "NetDebt",
        # Working capital, tangibles
        "Working Capital": "WorkingCapital",
        "Tangible Book Value": "TangibleBookValue",
        "Net Tangible Assets": "NetTangibleAssets",
    }

    # columns that feed CurrentAssets
    CA_COMPONENTS = [
        "Cash",
        "ShortTermInvestments",
        "AccountsReceivable",
        "Inventory",
        "OtherCurrentAssets",
    ]
    # columns that feed NonCurrentAssets
    NCA_COMPONENTS = ["NetPPE", "InvestmentsAndAdvances", "OtherNonCurrentAssets"]
    # CL components
    CL_COMPONENTS = [
        "AccountsPayable",
        "IncomeTaxPayable",
        "TotalTaxPayable",
        "CurrentDebt",
        "OtherCurrentBorrowings",
        "OtherCurrentLiabilities",
    ]
    # NCL components
    NCL_COMPONENTS = ["LongTermDebt", "LTDebtAndLeases", "OtherNonCurrentLiabilities"]

    def fit_transform(self, yf_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns:
          (canon_df, checks_df)
          - canon_df: canonical, repaired balance sheet
          - checks_df: per-date errors for each identity (abs error)
        """
        df = self._normalize(yf_df)
        rows = []
        checks = []
        for dt, row in df.iterrows():
            canon, chk = self._process_row(row)
            # TODO add datetime to typing Dict
            canon["Date"] = dt
            rows.append(canon)
            chk["Date"] = dt
            checks.append(chk)

        canon_df = pd.DataFrame(rows).set_index("Date").sort_index()
        checks_df = pd.DataFrame(checks).set_index("Date").sort_index()
        return canon_df[self.CANON_COLS], checks_df

    # -----------------------------------------------------------------------

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # Expect yfinance “rows=line items, cols=dates”; handle both orientations
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        # If columns look like dates, transpose so dates are index
        if not isinstance(df.index, pd.DatetimeIndex):
            # heuristic: if columns are datetime-like
            if pd.to_datetime(df.columns, errors="coerce").notna().any():
                df = df.T
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()].sort_index()
        # Coerce numerics
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # Drop fully-NaN dates and fully-NaN line items
        df = df.dropna(how="all").dropna(how="all", axis=1)
        return df

    def _process_row(self, row: pd.Series) -> Tuple[Dict[str, float], Dict[str, float]]:
        d = self._map_to_canonical(row)
        computed: Dict[str, float] = {}
        errs: Dict[str, float] = {}

        self._derive_cash_sti(d, computed)
        self._compute_ppe(d, computed)

        self._compute_current_assets(d, computed)
        self._compute_noncurrent_assets(d, computed)
        self._compute_total_assets(d, computed)

        self._compute_current_liabilities(d, computed)
        self._compute_noncurrent_liabilities(d, computed)
        self._compute_total_liabilities(d, computed)

        self._compute_total_debt(d, computed)
        self._compute_net_debt(d, computed)

        self._compute_equity(d, computed)
        self._enforce_balance_identity(d, computed, errs)

        self._estimate_minority_interest(d, computed)
        self._compute_working_capital(d, computed)
        self._compute_tangible_book(d, computed)

        self._record_errors(d, errs)

        return d, errs

    # ---------- mapping ----------
    def _map_to_canonical(self, row: pd.Series) -> Dict[str, float]:
        d: Dict[str, float] = dict.fromkeys(self.CANON_COLS, np.nan)
        for yf_col, canon_col in self.YF_TO_CANON.items():
            if yf_col in row.index:
                val = row[yf_col]
                if pd.notna(val) and pd.isna(d.get(canon_col, np.nan)):
                    d[canon_col] = float(val)
        return d

    # ---------- derivations: cash / STI ----------
    def _derive_cash_sti(self, d: Dict[str, float], computed: Dict[str, float]) -> None:
        if (
            pd.isna(d["Cash"])
            and pd.notna(d["CashAndSTI"])
            and pd.notna(d["ShortTermInvestments"])
        ):
            d["Cash"] = d["CashAndSTI"] - d["ShortTermInvestments"]
            computed["Cash"] = d["Cash"]

        if (
            pd.isna(d["ShortTermInvestments"])
            and pd.notna(d["CashAndSTI"])
            and pd.notna(d["Cash"])
        ):
            d["ShortTermInvestments"] = d["CashAndSTI"] - d["Cash"]
            computed["ShortTermInvestments"] = d["ShortTermInvestments"]

        if pd.isna(d["CashAndSTI"]) and (
            pd.notna(d["Cash"]) or pd.notna(d["ShortTermInvestments"])
        ):
            d["CashAndSTI"] = _nan_sum([d["Cash"], d["ShortTermInvestments"]])
            computed["CashAndSTI"] = d["CashAndSTI"]

    # ---------- derivations: PPE ----------
    def _compute_ppe(self, d: Dict[str, float], computed: Dict[str, float]) -> None:
        if pd.isna(d["NetPPE"]) and (
            pd.notna(d["GrossPPE"]) or pd.notna(d["AccumDepreciation"])
        ):
            d["NetPPE"] = _nan_sum([d["GrossPPE"], -d["AccumDepreciation"]])
            computed["NetPPE"] = d["NetPPE"]

        if (
            pd.isna(d["GrossPPE"])
            and pd.notna(d["NetPPE"])
            and pd.notna(d["AccumDepreciation"])
        ):
            d["GrossPPE"] = d["NetPPE"] + d["AccumDepreciation"]
            computed["GrossPPE"] = d["GrossPPE"]

    # ---------- assets ----------
    def _compute_current_assets(
        self, d: Dict[str, float], computed: Dict[str, float]
    ) -> None:
        ca_sum = _nan_sum(d[k] for k in self.CA_COMPONENTS)
        if not np.isnan(ca_sum):
            d["CurrentAssets"] = ca_sum
            computed["CurrentAssets"] = d["CurrentAssets"]

    def _compute_noncurrent_assets(
        self, d: Dict[str, float], computed: Dict[str, float]
    ) -> None:
        nca_sum = _nan_sum(d[k] for k in self.NCA_COMPONENTS)
        if not np.isnan(nca_sum):
            d["NonCurrentAssets"] = nca_sum
            computed["NonCurrentAssets"] = d["NonCurrentAssets"]

    def _compute_total_assets(
        self, d: Dict[str, float], computed: Dict[str, float]
    ) -> None:
        ta_sum = _nan_sum([d["CurrentAssets"], d["NonCurrentAssets"]])
        if not np.isnan(ta_sum):
            d["TotalAssets"] = ta_sum
            computed["TotalAssets"] = d["TotalAssets"]

    # ---------- liabilities ----------
    def _compute_current_liabilities(
        self, d: Dict[str, float], computed: Dict[str, float]
    ) -> None:
        cl_sum = _nan_sum(d[k] for k in self.CL_COMPONENTS)
        if not np.isnan(cl_sum):
            d["CurrentLiabilities"] = cl_sum
            computed["CurrentLiabilities"] = d["CurrentLiabilities"]

    def _compute_noncurrent_liabilities(
        self, d: Dict[str, float], computed: Dict[str, float]
    ) -> None:
        ncl_sum = _nan_sum(d[k] for k in self.NCL_COMPONENTS)
        if not np.isnan(ncl_sum):
            d["NonCurrentLiabilities"] = ncl_sum
            computed["NonCurrentLiabilities"] = d["NonCurrentLiabilities"]

    def _compute_total_liabilities(
        self, d: Dict[str, float], computed: Dict[str, float]
    ) -> None:
        tl_sum = _nan_sum([d["CurrentLiabilities"], d["NonCurrentLiabilities"]])
        if not np.isnan(tl_sum):
            d["TotalLiabilities"] = tl_sum
            computed["TotalLiabilities"] = d["TotalLiabilities"]

    # ---------- debt ----------
    def _debt_parts(self, d: Dict[str, float]) -> float:
        return _nan_sum(
            [
                d["CurrentDebt"],
                d["OtherCurrentBorrowings"],
                d["LongTermDebt"],
                d["LTDebtAndLeases"],
            ]
        )

    def _compute_total_debt(
        self, d: Dict[str, float], computed: Dict[str, float]
    ) -> None:
        td_sum = self._debt_parts(d)
        if not np.isnan(td_sum):
            d["TotalDebt"] = td_sum
            computed["TotalDebt"] = d["TotalDebt"]

    def _compute_net_debt(
        self, d: Dict[str, float], computed: Dict[str, float]
    ) -> None:
        if pd.notna(d.get("TotalDebt")) and (
            pd.notna(d.get("Cash"))
            or pd.notna(d.get("ShortTermInvestments"))
            or pd.notna(d.get("CashAndSTI"))
        ):
            cash_like = (
                d["CashAndSTI"]
                if pd.notna(d["CashAndSTI"])
                else _nan_sum([d["Cash"], d["ShortTermInvestments"]])
            )
            if pd.notna(cash_like):
                d["NetDebt"] = d["TotalDebt"] - cash_like
                computed["NetDebt"] = d["NetDebt"]

    # ---------- equity / identities ----------
    def _compute_equity(self, d: Dict[str, float], computed: Dict[str, float]) -> None:
        if pd.isna(d["TotalEquityInclMI"]):
            eq_sum = _nan_sum(
                [
                    d["CommonStockEquity"],
                    d["RetainedEarnings"],
                    d["OtherEquityAdjustments"],
                ]
            )
            if not np.isnan(eq_sum):
                d["TotalEquityInclMI"] = eq_sum
                computed["TotalEquityInclMI"] = d["TotalEquityInclMI"]

    def _enforce_balance_identity(
        self,
        d: Dict[str, float],
        computed: Dict[str, float],
        errs: Dict[str, float],
    ) -> None:
        ta = d.get("TotalAssets")
        tl = d.get("TotalLiabilities")
        te = d.get("TotalEquityInclMI")

        if pd.notna(ta) and pd.notna(tl) and pd.notna(te):
            errs["A=L+E"] = abs(ta - (tl + te))
            if errs["A=L+E"] > self.eps:
                d["TotalEquityInclMI"] = ta - tl
                computed["TotalEquityInclMI(residual)"] = d["TotalEquityInclMI"]
                errs["A=L+E(after)"] = abs(ta - (tl + d["TotalEquityInclMI"]))
        elif pd.notna(ta) and pd.notna(tl) and pd.isna(te):
            d["TotalEquityInclMI"] = ta - tl
            computed["TotalEquityInclMI(residual)"] = d["TotalEquityInclMI"]

    # ---------- other heuristics ----------
    def _estimate_minority_interest(
        self, d: Dict[str, float], computed: Dict[str, float]
    ) -> None:
        if (
            pd.isna(d["MinorityInterestEstimated"])
            and pd.notna(d["TotalEquityInclMI"])
            and pd.notna(d["StockholdersEquity"])
        ):
            d["MinorityInterestEstimated"] = (
                d["TotalEquityInclMI"] - d["StockholdersEquity"]
            )
            computed["MinorityInterestEstimated"] = d["MinorityInterestEstimated"]

    def _compute_working_capital(
        self, d: Dict[str, float], computed: Dict[str, float]
    ) -> None:
        if (
            pd.isna(d["WorkingCapital"])
            and pd.notna(d["CurrentAssets"])
            and pd.notna(d["CurrentLiabilities"])
        ):
            d["WorkingCapital"] = d["CurrentAssets"] - d["CurrentLiabilities"]
            computed["WorkingCapital"] = d["WorkingCapital"]

    def _compute_tangible_book(
        self, d: Dict[str, float], computed: Dict[str, float]
    ) -> None:
        if pd.isna(d["TangibleBookValue"]) and pd.notna(d["NetTangibleAssets"]):
            d["TangibleBookValue"] = d["NetTangibleAssets"]
            computed["TangibleBookValue"] = d["TangibleBookValue"]

    # ---------- error recording ----------
    def _record_errors(self, d: Dict[str, float], errs: Dict[str, float]) -> None:
        def err(name: str, lhs: float, rhs: float) -> None:
            if pd.notna(lhs) and pd.notna(rhs):
                errs[name] = abs(lhs - rhs)

        err(
            "NetPPE=Gross-Accum",
            d["NetPPE"],
            _nan_sum([d["GrossPPE"], -d["AccumDepreciation"]]),
        )
        err(
            "CA=sum(components)",
            d["CurrentAssets"],
            _nan_sum(d[k] for k in self.CA_COMPONENTS),
        )
        err(
            "NCA=sum(components)",
            d["NonCurrentAssets"],
            _nan_sum(d[k] for k in self.NCA_COMPONENTS),
        )
        err(
            "TA=CA+NCA",
            d["TotalAssets"],
            _nan_sum([d["CurrentAssets"], d["NonCurrentAssets"]]),
        )
        err(
            "CL=sum(components)",
            d["CurrentLiabilities"],
            _nan_sum(d[k] for k in self.CL_COMPONENTS),
        )
        err(
            "NCL=sum(components)",
            d["NonCurrentLiabilities"],
            _nan_sum(d[k] for k in self.NCL_COMPONENTS),
        )
        err(
            "TL=CL+NCL",
            d["TotalLiabilities"],
            _nan_sum([d["CurrentLiabilities"], d["NonCurrentLiabilities"]]),
        )
        td_parts = self._debt_parts(d)
        err("TotalDebt=parts", d["TotalDebt"], td_parts)

        if pd.notna(d.get("TotalDebt")):
            cash_like = (
                d["CashAndSTI"]
                if pd.notna(d["CashAndSTI"])
                else _nan_sum([d["Cash"], d["ShortTermInvestments"]])
            )
            if pd.notna(cash_like):
                err(
                    "NetDebt=TotalDebt-CashLike",
                    d["NetDebt"],
                    d["TotalDebt"] - cash_like,
                )


if __name__ == "__main__":
    # yf_bs is your yfinance balance sheet (dates on index, columns from your list)
    from src.jpm.question_1.data.yf import FinanceIngestor

    yf_ing = FinanceIngestor(
        "AAPL", cache_dir="/Users/tavisshore/Desktop/HK/data", ttl_days=0
    )
    yf_bs = yf_ing.balance_sheet()

    engine = BalanceSheetBaseline()
    canon, checks = engine.fit_transform(yf_bs)

    # Inspect violations (should be near zero after residual repair)
    print(checks.tail(3).round(2))
    # Your canonical, constraint-consistent balance sheet:
    print(canon.tail(3)[["TotalAssets", "TotalLiabilities", "TotalEquityInclMI"]])
