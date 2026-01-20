"""Question 1: Financial Forecasting and Analysis.

This module provides commonly used utilities for financial forecasting tasks.
"""

from jpm.config import get_args
from jpm.question_1.clients import LLMClient
from jpm.question_1.data import CreditDataset, EdgarData, StatementsDataset
from jpm.question_1.misc import as_series, set_seed
from jpm.question_1.models.lstm import LSTMForecaster
from jpm.question_1.models.validation.balance_sheet import BalanceSheet
from jpm.question_1.models.validation.income_statement import IncomeStatement
from jpm.question_1.models.xgb import CreditRatingModel

__all__ = [
    # Data loaders
    "EdgarData",
    # Datasets
    "CreditDataset",
    "StatementsDataset",
    # Models
    "BalanceSheet",
    "IncomeStatement",
    "LSTMForecaster",
    "CreditRatingModel",
    # Clients
    "LLMClient",
    # Utilities
    "as_series",
    "get_args",
    "set_seed",
]
