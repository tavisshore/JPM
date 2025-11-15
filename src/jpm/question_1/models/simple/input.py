from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, cast

import numpy as np
import pandas as pd
import tensorflow as tf

from src.jpm.question_1.data.yf import FinanceIngestor

from .balance import BalanceSheet
from .cash import CashFlow
from .income import IncomeStatement

# NOTE - improve upon this method
_YF_IS_MAP = {
    "Total Revenue": "total_revenue",
    "Cost Of Revenue": "cost_of_revenue",
    "Gross Profit": "gross_profit",
    "Operating Expense": "operating_expense",
    "Operating Income": "operating_income",
    "Interest Expense": "interest_expense",
    "Other Income Expense": "other_income_expense",
    "Income Before Tax": "income_before_tax",
    "Income Tax Expense": "income_tax_expense",
    "Net Income": "net_income",
}

_YF_BS_MAP = {
    "Cash And Cash Equivalents": "cash_and_equivalents",
    "Short Term Investments": "short_term_investments",
    "Accounts Receivable": "accounts_receivable",
    "Inventory": "inventory",
    "Other Current Assets": "other_current_assets",
    "Total Current Assets": "total_current_assets",
    "Property Plant And Equipment Net": "pp_and_e",
    "Goodwill And Other Intangible Assets": "goodwill_intangibles",
    "Other Non Current Assets": "other_non_current_assets",
    "Total Assets": "total_assets",
    "Accounts Payable": "accounts_payable",
    "Current Debt": "short_term_debt",
    "Other Current Liabilities": "other_current_liabilities",
    "Total Current Liabilities": "total_current_liabilities",
    "Long Term Debt": "long_term_debt",
    "Other Non Current Liabilities": "other_non_current_liabilities",
    "Total Liabilities Net Minority Interest": "total_liabilities",
    "Common Stock": "common_stock",
    "Retained Earnings": "retained_earnings",
    "Accumulated Other Comprehensive Income": "aoci",
    "Treasury Stock": "treasury_stock",
    "Stockholders Equity": "total_shareholder_equity",
}

_YF_CF_MAP = {
    "Operating Cash Flow": "cash_from_operations",
    "Investing Cash Flow": "cash_from_investing",
    "Financing Cash Flow": "cash_from_financing",
    "End Cash Position": None,  # reconcile via BS + net change
    "Begin Cash Position": None,
    "Net Change In Cash": "net_change_in_cash",
}


@dataclass
class FinancialRecord:
    ticker: str
    period_end: pd.Timestamp
    income: IncomeStatement
    balance: BalanceSheet
    cashflow: CashFlow

    validity_threshold: tf.Tensor = field(
        default_factory=lambda: cast(tf.Tensor, tf.constant(1e-3, dtype=tf.float32))
    )

    # Prior retained earnings + cash roll-forward checks
    prior_balance: Optional[BalanceSheet] = None

    def cross_validate(self) -> Dict[str, tf.Tensor]:
        errs: Dict[str, tf.Tensor] = {}
        errs.update({f"IS::{k}": v for k, v in self.income.validate().items()})
        errs.update({f"BS::{k}": v for k, v in self.balance.validate().items()})
        errs.update({f"CF::{k}": v for k, v in self.cashflow.validate().items()})

        # Cash roll-forward link: CF must reconcile to BS cash
        if self.prior_balance is not None:
            start_cash = self.prior_balance.cash_and_equivalents
            end_cash = self.balance.cash_and_equivalents
            errs["cash_rollforward"] = tf.abs(
                tf.subtract(
                    end_cash, tf.add(start_cash, self.cashflow.net_change_in_cash)
                )
            )  # Retained earnings link: RE_t = RE_{t-1} + NI - Dividends
            # (approx: use financing cash dividends if available)
            # yfinance CF often has 'Dividends Paid' as part of financing;
            # we don't have it here -> treat as latent.
            # We can at least check that change in RE approximately equals
            # NI minus a non-negative dividend proxy.
            # If RE decreased despite positive NI, flag it (could be large
            # dividends or restatement).
        return errs

    @property
    def is_valid(self) -> tf.Tensor:
        vals = [
            tf.convert_to_tensor(v, dtype=tf.float32)
            for v in self.cross_validate().values()
        ]
        return tf.reduce_all(tf.math.less(tf.stack(vals), self.validity_threshold))

    def to_vector(self) -> np.ndarray:
        # Surely there's a nicer way to do this... risky with ordering
        vec = [
            # Income Statement
            self.income.total_revenue,
            self.income.cost_of_revenue,
            self.income.gross_profit,
            self.income.operating_expense,
            self.income.operating_income,
            self.income.interest_expense,
            self.income.other_income_expense,
            self.income.income_before_tax,
            self.income.income_tax_expense,
            self.income.net_income,
            # Balance Sheet
            self.balance.cash_and_equivalents,
            self.balance.short_term_investments,
            self.balance.accounts_receivable,
            self.balance.inventory,
            self.balance.other_current_assets,
            self.balance.total_current_assets,
            self.balance.pp_and_e,
            self.balance.goodwill_intangibles,
            self.balance.other_non_current_assets,
            self.balance.total_assets,
            self.balance.accounts_payable,
            self.balance.short_term_debt,
            self.balance.other_current_liabilities,
            self.balance.total_current_liabilities,
            self.balance.long_term_debt,
            self.balance.other_non_current_liabilities,
            self.balance.total_liabilities,
            self.balance.common_stock,
            self.balance.retained_earnings,
            self.balance.aoci,
            self.balance.treasury_stock,
            self.balance.total_shareholder_equity,
            # Cash Flow
            self.cashflow.cash_from_operations,
            self.cashflow.cash_from_investing,
            self.cashflow.cash_from_financing,
            self.cashflow.net_change_in_cash,
        ]
        return np.array(vec, dtype=float)


@dataclass
class BalanceSheetSeries:
    data_ingestion: FinanceIngestor

    def __post_init__(self):
        self.bs_series = self.fetch_balance_series()

    def fetch_balance_series(self):  # -> pd.DataFrame:
        # Make more configurable - longer series
        balance_sheets = self.data_ingestion.balance_sheet()

        # For each row - create a BalanceSheet dataclass

        print(balance_sheets)
        # bs_data = {
        #     "period_end": [],
        #     **{field: [] for field in BalanceSheet.__dataclass_fields__.keys()}
        # }
        # for rec in records:
        #     bs_data["period_end"].append(rec.period_end)
        #     for field in BalanceSheet.__dataclass_fields__.keys():
        #         bs_data[field].append(getattr(rec.balance, field).numpy())
        # bs_df = pd.DataFrame(bs_data)
        # bs_df.set_index("period_end", inplace=True)
        # return bs_df


if __name__ == "__main__":
    ingestor = FinanceIngestor(
        "AAPL", cache_dir="/Users/tavisshore/Desktop/HK/data", ttl_days=7
    )
    cbs = BalanceSheetSeries(data_ingestion=ingestor)

    # print(bs_df)
