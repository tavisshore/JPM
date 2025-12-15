import json

import numpy as np

from jpm.question_1.config import (
    Config,
    DataConfig,
    LLMConfig,
    LossConfig,
    ModelConfig,
    TrainingConfig,
)
from jpm.question_1.data.ed import EdgarDataLoader
from jpm.question_1.misc import get_args, set_seed
from jpm.question_1.models.balance_sheet import BalanceSheet
from jpm.question_1.models.income_statement import IncomeStatement
from jpm.question_1.models.lstm import LSTMForecaster

set_seed(42)
args = get_args()

data_cfg = DataConfig.from_args(args)
model_cfg = ModelConfig.from_args(args)
train_cfg = TrainingConfig.from_args(args)
loss_cfg = LossConfig.from_args(args)
llm_cfg = LLMConfig.from_args(args)
config = Config(
    data=data_cfg, model=model_cfg, training=train_cfg, loss=loss_cfg, llm=llm_cfg
)


results = {}

tickers = [
    "NVDA",
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "AVGO",
    "META",
    "TSLA",
    "BRK.B",
    "LLY",
    "WMT",
    "JPM",
    "V",
    "ORCL",
    "MA",
    "JNJ",
    "XOM",
    "PLTR",
    "NFLX",
    "AMD",
    "CSCO",
    "IBM",
    "MS",
    "CAT",
    "GS",
    "AXP",
    "RTX",
    "MCD",
    "TMUS",
    "PEP",
]

for ticker in tickers:
    config.data.ticker = ticker

    data = EdgarDataLoader(config=config)
    model = LSTMForecaster(config=config, data=data)

    model.fit()

    model.evaluate(stage="train")
    model.view_results(stage="train")

    validation_results = model.evaluate(stage="val")
    model.view_results(stage="val")

    # Pass outputs to BS Model
    bs = BalanceSheet(config=config, data=data, results=validation_results)
    bs_pct_error = bs.check_identity()

    # Income Statement to predict Net Income (Loss)
    i_s = IncomeStatement(config=config, data=data, results=validation_results)
    is_results = i_s.get_results()

    results[ticker] = {
        "net_income": is_results.features["Net Income (Loss)"].mae,
        "lstm": validation_results.model_mae,
        **validation_results.baseline_mae,
        "bs_pct_error": bs_pct_error,
    }

# Summarize results
print("\nFinal Results Summary:")
summary = {}
for key in results[tickers[0]].keys():
    vals = [results[ticker][key] for ticker in tickers]
    mean_val = np.mean(vals)
    std_val = np.std(vals)
    summary[key] = {"mean": mean_val, "std": std_val}

print(f"{'Metric':<15} {'Mean MAE':<15} {'Std Dev':<15}")
for key, stats in summary.items():
    print(f"{key:<15} {stats['mean']:<15.2f} {stats['std']:<15.2f}")

# Save detailed results to a JSON file
with open("lstm_evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)
