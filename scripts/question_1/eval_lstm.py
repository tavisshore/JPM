import gc
import json
import os
import sys

import numpy as np
import tensorflow as tf

from jpm.question_1 import (
    BalanceSheet,
    Config,
    DataConfig,
    EdgarData,
    EdgarDataset,
    IncomeStatement,
    LLMConfig,
    LossConfig,
    LSTMForecaster,
    ModelConfig,
    TrainingConfig,
    get_args,
    set_seed,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logging
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use single GPU to avoid contention

tf.get_logger().setLevel("ERROR")

# Limit TensorFlow memory growth to prevent OOM
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU memory growth setting failed: {e}", file=sys.stderr)


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

# Try to load partial results if they exist (for restart)
try:
    with open("lstm_evaluation_results_partial.json", "r") as f:
        results = json.load(f)
        print(f"Loaded {len(results)} existing results. Resuming...", flush=True)
except FileNotFoundError:
    pass

tickers = [
    "AAPL",
    "AMZN",
    "AVGO",
    "META",
    "TSLA",
    "WMT",
    "GS",
]

for idx, ticker in enumerate(tickers, 1):
    print(f"\n[{idx}/{len(tickers)}] Processing {ticker}...", flush=True)

    config.data.ticker = ticker

    data = EdgarData(config=config)
    dataset = EdgarDataset(edgar_data=data, target="lstm")

    model = LSTMForecaster(config=config, data=data, dataset=dataset)

    model.fit(verbose=0)

    model.evaluate(stage="train")

    validation_results = model.evaluate(stage="val", llm_config=llm_cfg)

    bs = BalanceSheet(config=config, data=data, results=validation_results)
    bs_pct_error = bs.check_identity()

    i_s = IncomeStatement(config=config, data=data, results=validation_results)
    i_s.view()
    is_results = i_s.get_results()

    results[ticker] = {
        "net_income": {
            "lstm": validation_results.net_income_model_mae,
            **validation_results.baseline_mae,
        },
        "balance_sheet": {
            "lstm": validation_results.model_mae,
            **validation_results.baseline_mae,
        },
        "bs_pct_error": bs_pct_error,
    }

    print(
        f"  ✓ {ticker}: LSTM MAE=${validation_results.model_mae / 1e9:.2f}bn",
        flush=True,
    )

    # Clear memory after each ticker
    del model, data, bs, i_s, validation_results, is_results

    gc.collect()
    tf.keras.backend.clear_session()

    # Save intermediate results
    with open("lstm_evaluation_results_partial.json", "w") as f:
        json.dump(results, f, indent=2)


# Summarize results
print("\nFinal Results Summary:")


summary = {}
for key in ["net_income", "balance_sheet"]:
    for model in results[tickers[0]][key].keys():
        summary_key = f"{key}_{model}"
        vals = [results[ticker][key][model] for ticker in tickers]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        summary[summary_key] = {"mean": mean_val, "std": std_val}


vals = [results[ticker]["bs_pct_error"] for ticker in tickers]
mean_val = np.mean(vals)
std_val = np.std(vals)
summary["bs_pct_error"] = {"mean": mean_val, "std": std_val}


print(f"{'Metric':<30} {'Mean MAE':<15} {'Std Dev':<15}")
for key, stats in summary.items():
    print(f"{key:<30} {stats['mean'] / 1e9:<15.2f} {stats['std'] / 1e9:<15.2f}")

# Save detailed results to a JSON file
with open("lstm_evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)
