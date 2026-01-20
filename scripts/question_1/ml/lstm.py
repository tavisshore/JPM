"""
LSTM Training Script for Balance Sheet Forecasting.

Supports both deterministic and probabilistic LSTM models with configurable
variations:

Standard Deterministic Models:
- Can use enforce_balance constraint
- Can learn balance sheet identity

Probabilistic Models:
- LSTM with probabilistic outputs (Multivariate Normal distribution)
- Provides uncertainty quantification
- Includes calibration diagnostics
- Cannot use enforce_balance (incompatible with probabilistic outputs)
"""

import gc
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from jpm.config import Config, DataConfig, LLMConfig, LSTMConfig
from jpm.question_1 import (
    BalanceSheet,
    EdgarData,
    IncomeStatement,
    LSTMForecaster,
    StatementsDataset,
    get_args,
    set_seed,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
lstm_cfg = LSTMConfig.from_args(args)
llm_cfg = LLMConfig.from_args(args)

view_plot = True

CONFIG_VARIATIONS = [
    # Deterministic models
    # {"learn_identity": False, "enforce_balance": False, "probabilistic": False, "variational": False},
    # {"learn_identity": True, "enforce_balance": False, "probabilistic": False, "variational": False},
    {
        "learn_identity": True,
        "enforce_balance": True,
        "probabilistic": False,
        "variational": False,
    },
    # Probabilistic models (cannot use enforce_balance with probabilistic=True)
    {
        "learn_identity": True,
        "enforce_balance": False,
        "probabilistic": True,
        "variational": False,
    },
    # Variational models (Bayesian uncertainty via weight distributions)
    {
        "learn_identity": True,
        "enforce_balance": False,
        "probabilistic": False,
        "variational": True,
    },
]


tickers = """
AAPL
""".split()

all_config_results = {}
failed = []
for var_idx, variation in enumerate(CONFIG_VARIATIONS, 1):
    config_name = f"learn_identity={variation['learn_identity']}, \
        enforce_balance={variation['enforce_balance']}, \
        probabilistic={variation['probabilistic']}, \
        variational={variation['variational']}"
    print(f"\n{'=' * 90}")
    print(f"Config {var_idx}/{len(CONFIG_VARIATIONS)}: {config_name}")
    print(f"{'=' * 90}")

    lstm_cfg_var = replace(
        lstm_cfg,
        enforce_balance=variation["enforce_balance"],
        learn_identity=variation["learn_identity"],
        probabilistic=variation["probabilistic"],
        variational=variation["variational"],
    )
    config = Config(
        data=data_cfg,
        lstm=lstm_cfg_var,
        llm=llm_cfg,
    )

    results = {}

    for idx, ticker in enumerate(tickers, 1):
        print(f"\n[{idx}/{len(tickers)}] Processing {ticker}...", flush=True)

        try:
            config.data.ticker = ticker

            data = EdgarData(config=config)
            dataset = StatementsDataset(edgar_data=data)
            model = LSTMForecaster(config=config, data=data, dataset=dataset)
            model.fit(verbose=0)
            model.evaluate(stage="train")

            if config.llm.use_llm:
                validation_results = model.evaluate(stage="val", llm_config=llm_cfg)
            else:
                validation_results = model.evaluate(stage="val")

            # Display results based on model type
            if config.lstm.probabilistic:
                print(
                    f"\n  {ticker}: Probabilistic Results (Validation Set):", flush=True
                )
                model.view_probabilistic_results(stage="val")
                print(f"\n  {ticker}: Calibration Diagnostics:", flush=True)
                model.view_calibration_results(stage="val")
            elif config.lstm.variational:
                print(
                    f"\n  {ticker}: Variational Results (Validation Set):", flush=True
                )
                model.view_results(stage="val")
            else:
                model.view_results(stage="val")

            bs = BalanceSheet(
                config=config, data=data, dataset=dataset, results=validation_results
            )
            # bs.view()
            bs_pct_error = bs.check_identity(verbose=False)

            i_s = IncomeStatement(
                config=config, dataset=dataset, results=validation_results
            )
            # i_s.view()
            # is_results = i_s.get_results()

            # Compute calibration metrics for probabilistic models
            calibration_metrics = None
            if config.lstm.probabilistic:
                try:
                    calibration_metrics = model.compute_calibration_metrics(stage="val")
                except Exception as e:
                    print(f"Warning: Could not compute calibration metrics: {e}")

            results[ticker] = {
                "net_income": {
                    "lstm": validation_results.net_income_model_mae,
                    **validation_results.net_income_baseline_mae,
                },
                "balance_sheet": {
                    "lstm": validation_results.model_mae,
                    **validation_results.baseline_mae,
                },
                "bs_pct_error": bs_pct_error,
                "probabilistic": config.lstm.probabilistic,
                "variational": config.lstm.variational,
            }

            # Add calibration metrics if available
            if calibration_metrics:
                results[ticker]["calibration"] = {
                    "calibration_error": calibration_metrics["calibration_error"],
                    "sharpness": calibration_metrics["sharpness"],
                    "crps": calibration_metrics["crps"],
                    "coverage": calibration_metrics["coverage"],
                }

            # Now predict the future on the final lookback window dataset.predict_dataset
            predictions = model.predict(dataset.predict_dataset)
            print(f"\n  {ticker}: Future Balance Sheet Prediction:", flush=True)

            bs = BalanceSheet(
                config=config, data=data, dataset=dataset, results=predictions
            )
            bs.view()
            bs_pct_error = bs.check_identity()
            i_s = IncomeStatement(config=config, dataset=dataset, results=predictions)
            i_s.view_predict()

            time.sleep(5)
            # Clear memory after each ticker
            del (
                model,
                data,
                dataset,
                bs,
                i_s,
                validation_results,
                predictions,
            )

            gc.collect()
            tf.keras.backend.clear_session()

        except Exception as E:
            print(f"\n{E}", flush=True)
            failed.append(ticker)

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

    # Print calibration metrics if probabilistic
    if variation["probabilistic"] and "calibration" in results[tickers[0]]:
        print(f"\n{'Calibration Metric':<30} {'Mean':<15} {'Std Dev':<15}")
        print("-" * 60)

        # Calibration error
        cal_errors = [
            results[ticker]["calibration"]["calibration_error"]
            for ticker in tickers
            if "calibration" in results[ticker]
        ]
        if cal_errors:
            print(
                f"{'Calibration Error':<30} {np.mean(cal_errors):<15.4f} "
                f"{np.std(cal_errors):<15.4f}"
            )

        # Sharpness
        sharpness_vals = [
            results[ticker]["calibration"]["sharpness"]
            for ticker in tickers
            if "calibration" in results[ticker]
        ]
        if sharpness_vals:
            print(
                f"{'Sharpness (95% PI width)':<30} {np.mean(sharpness_vals) / 1e9:<15.2f} "
                f"{np.std(sharpness_vals) / 1e9:<15.2f}"
            )

        # CRPS
        crps_vals = [
            results[ticker]["calibration"]["crps"]
            for ticker in tickers
            if "calibration" in results[ticker]
            and results[ticker]["calibration"]["crps"] is not None
        ]
        if crps_vals:
            print(
                f"{'CRPS':<30} {np.mean(crps_vals) / 1e9:<15.2f} "
                f"{np.std(crps_vals) / 1e9:<15.2f}"
            )


# Final comparison across all configs
print("\n" + "=" * 80)
print("FINAL COMPARISON ACROSS ALL CONFIGURATIONS")
print("=" * 80)

for config_name, results in all_config_results.items():
    print(f"\n{config_name}:")
    lstm_mae_vals = [results[ticker]["balance_sheet"]["lstm"] for ticker in tickers]
    bs_pct_vals = [results[ticker]["bs_pct_error"] for ticker in tickers]
    is_probabilistic = results[tickers[0]].get("probabilistic", False)

    print(
        f"  BS LSTM MAE: {np.mean(lstm_mae_vals) / 1e9:.2f}bn \
            (+/- {np.std(lstm_mae_vals) / 1e9:.2f}bn)"
    )
    print(
        f"  BS Identity Error:      \
            {np.mean(bs_pct_vals):.2%} (+/- {np.std(bs_pct_vals):.2%})"
    )

    # Add calibration info for probabilistic models
    if is_probabilistic and "calibration" in results[tickers[0]]:
        cal_errors = [
            results[ticker]["calibration"]["calibration_error"]
            for ticker in tickers
            if "calibration" in results[ticker]
        ]
        if cal_errors:
            print(
                f"  Calibration Error:      "
                f"{np.mean(cal_errors):.4f} (+/- {np.std(cal_errors):.4f})"
            )

        crps_vals = [
            results[ticker]["calibration"]["crps"]
            for ticker in tickers
            if "calibration" in results[ticker]
            and results[ticker]["calibration"]["crps"] is not None
        ]
        if crps_vals:
            print(
                f"  CRPS:                   "
                f"{np.mean(crps_vals) / 1e9:.2f}bn (+/- {np.std(crps_vals) / 1e9:.2f}bn)"
            )

# Save all results to JSON
results_dir = Path("results/question_1/")
results_dir.parent.mkdir(parents=True, exist_ok=True)

with open(results_dir / "lstm_evaluation.json", "w") as f:
    json.dump(all_config_results, f, indent=2)


# After collecting all results
sorted_tickers = sorted(
    results.items(),
    key=lambda x: x[1]["net_income"]["lstm"] + x[1]["balance_sheet"]["lstm"],
)
for ticker, metrics in sorted_tickers[:20]:
    combined_mae = metrics["net_income"]["lstm"] + metrics["balance_sheet"]["lstm"]
    print(
        f"{ticker}: {combined_mae:.4f} (NI: {metrics['net_income']['lstm']:.4f}, BS: {metrics['balance_sheet']['lstm']:.4f})"
    )
