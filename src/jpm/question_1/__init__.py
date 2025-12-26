"""Question 1: Financial Forecasting and Analysis.

This module provides commonly used utilities for financial forecasting tasks.
"""

from jpm.question_1.clients import LLMClient
from jpm.question_1.config import (
    Config,
    DataConfig,
    LLMConfig,
    LossConfig,
    ModelConfig,
    TrainingConfig,
)
from jpm.question_1.data.ed import EdgarDataLoader
from jpm.question_1.misc import as_series, get_args, set_seed
from jpm.question_1.models.balance_sheet import BalanceSheet
from jpm.question_1.models.income_statement import IncomeStatement
from jpm.question_1.models.lstm import LSTMForecaster

__all__ = [
    # Config classes
    "Config",
    "DataConfig",
    "LLMConfig",
    "LossConfig",
    "ModelConfig",
    "TrainingConfig",
    # Data loaders
    "EdgarDataLoader",
    # Models
    "BalanceSheet",
    "IncomeStatement",
    "LSTMForecaster",
    # Clients
    "LLMClient",
    # Utilities
    "as_series",
    "get_args",
    "set_seed",
]
