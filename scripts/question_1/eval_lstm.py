import gc
import json
import os
import sys

import numpy as np

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

import tensorflow as tf

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

# Config variations to evaluate
CONFIG_VARIATIONS = [
    {"learn_identity": False, "enforce_balance": False},
    {"learn_identity": True, "enforce_balance": False},
    {"learn_identity": True, "enforce_balance": True},
]

tickers = [
    "MSFT",
    "AMZN",
    "AVGO",  # Performs so well on this
    "META",
]

all_config_results = {}

for var_idx, variation in enumerate(CONFIG_VARIATIONS, 1):
    config_name = f"learn_identity={variation['learn_identity']}, \
        enforce_balance={variation['enforce_balance']}"
    print(f"\n{'=' * 65}")
    print(f"Config {var_idx}/{len(CONFIG_VARIATIONS)}: {config_name}")
    print(f"{'=' * 65}")

    # Create loss config with this variation
    loss_cfg_var = LossConfig(
        enforce_balance=variation["enforce_balance"],
        learn_identity=variation["learn_identity"],
        identity_weight=loss_cfg.identity_weight,
        learn_subtotals=loss_cfg.learn_subtotals,
        subcategory_weight=loss_cfg.subcategory_weight,
    )
    config = Config(
        data=data_cfg,
        model=model_cfg,
        training=train_cfg,
        loss=loss_cfg_var,
        llm=llm_cfg,
    )

    results = {}

    for idx, ticker in enumerate(tickers, 1):
        print(f"\n[{idx}/{len(tickers)}] Processing {ticker}...", flush=True)

        try:
            config.data.ticker = ticker

            data = EdgarData(config=config)
            dataset = EdgarDataset(edgar_data=data, target="lstm")

            model = LSTMForecaster(config=config, data=data, dataset=dataset)

            model.fit(verbose=0)

            model.evaluate(stage="train")

            if config.llm.use_llm:
                validation_results = model.evaluate(stage="val", llm_config=llm_cfg)
            else:
                validation_results = model.evaluate(stage="val")

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
                f"  {ticker}: LSTM MAE=${validation_results.model_mae / 1e9:.2f}bn",
                flush=True,
            )

            # Clear memory after each ticker
            del model, data, bs, i_s, validation_results, is_results

            gc.collect()
            tf.keras.backend.clear_session()

        except Exception as E:
            print(E)

    # Store results for this config
    all_config_results[config_name] = results

    # Print summary for this config
    print(f"\n--- Results Summary for: {config_name} ---")
    summary = {}
    for key in ["net_income", "balance_sheet"]:
        for model_name in results[tickers[0]][key].keys():
            summary_key = f"{key}_{model_name}"
            vals = [results[ticker][key][model_name] for ticker in tickers]
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            summary[summary_key] = {"mean": mean_val, "std": std_val}

    vals = [results[ticker]["bs_pct_error"] for ticker in tickers]
    mean_val = np.mean(vals)
    std_val = np.std(vals)
    summary["bs_pct_error"] = {"mean": mean_val, "std": std_val}

    print(f"{'Metric':<30} {'Mean MAE (bn)':<15} {'Std Dev (bn)':<15}")
    print("-" * 60)
    for key, stats in summary.items():
        print(f"{key:<30} {stats['mean'] / 1e9:<15.2f} {stats['std'] / 1e9:<15.2f}")


# Final comparison across all configs
print("\n" + "=" * 80)
print("FINAL COMPARISON ACROSS ALL CONFIGURATIONS")
print("=" * 80)

for config_name, results in all_config_results.items():
    print(f"\n{config_name}:")
    lstm_mae_vals = [results[ticker]["balance_sheet"]["lstm"] for ticker in tickers]
    bs_pct_vals = [results[ticker]["bs_pct_error"] for ticker in tickers]
    print(
        f"  BS LSTM MAE: {np.mean(lstm_mae_vals) / 1e9:.2f}bn \
            (+/- {np.std(lstm_mae_vals) / 1e9:.2f}bn)"
    )
    print(
        f"  BS Identity Error:      \
            {np.mean(bs_pct_vals):.2%} (+/- {np.std(bs_pct_vals):.2%})"
    )

# Save all results to JSON
with open("lstm_evaluation_results_all_configs.json", "w") as f:
    json.dump(all_config_results, f, indent=2)
