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

from jpm.config import Config, DataConfig, LLMConfig, LSTMConfig, get_args
from jpm.question_1 import (
    BalanceSheet,
    EdgarData,
    IncomeStatement,
    LSTMForecaster,
    StatementsDataset,
)
from jpm.utils import compute_results_summary, get_tickers, set_seed

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

results_dir = Path("results/question_1/")
results_dir.parent.mkdir(parents=True, exist_ok=True)

data_cfg = DataConfig.from_args(args)
lstm_cfg = LSTMConfig.from_args(args)
llm_cfg = LLMConfig.from_args(args)

CONFIG_VARIATIONS = [
    # Deterministic models
    # {"learn_identity": False, "enforce_balance": False, "probabilistic": False, "variational": False},
    # {"learn_identity": True, "enforce_balance": False, "probabilistic": False, "variational": False},
    # {
    #     "learn_identity": True,
    #     "enforce_balance": True,
    #     "probabilistic": False,
    #     "variational": False,
    #     "learnable_seasonal_weight": False,
    # },
    # + learned seasonal weight
    {
        "learn_identity": False,
        "enforce_balance": False,
        "probabilistic": True,
        "variational": False,
        "learnable_seasonal_weight": False,
    },
    # Probabilistic models (cannot use enforce_balance with probabilistic=True)
    # {
    #     "learn_identity": True,
    #     "enforce_balance": False,
    #     "probabilistic": True,
    #     "variational": False,
    # },
    # Variational models (Bayesian uncertainty via weight distributions)
    # {
    #     "learn_identity": True,
    #     "enforce_balance": False,
    #     "probabilistic": False,
    #     "variational": True,
    # },
]

# If argument ticker used, only run that ticker
if args.ticker:
    tickers = [args.ticker]
else:
    tickers = get_tickers(args.industry, length=args.total_tickers)

all_config_results = {}
failed = []
for var_idx, variation in enumerate(CONFIG_VARIATIONS, 1):
    config_name = f"learn_identity={variation['learn_identity']}, \
        enforce_balance={variation['enforce_balance']}, \
        probabilistic={variation['probabilistic']}, \
        variational={variation['variational']}, \
        learnable_seasonal_weight={variation.get('learnable_seasonal_weight', False)}"

    print(f"\n{'=' * 90}")
    print(f"Config {var_idx}/{len(CONFIG_VARIATIONS)}: {config_name}")
    print(f"{'=' * 90}")

    lstm_cfg_new = replace(
        lstm_cfg,
        enforce_balance=variation["enforce_balance"],
        learn_identity=variation["learn_identity"],
        probabilistic=variation["probabilistic"],
        variational=variation["variational"],
    )
    data_cfg_new = replace(
        data_cfg,
        learnable_seasonal_weight=variation.get("learnable_seasonal_weight", False),
    )
    config = Config(
        data=data_cfg_new,
        lstm=lstm_cfg_new,
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
            bs.view()
            bs_pct_error = bs.check_identity(verbose=False)

            i_s = IncomeStatement(
                config=config, dataset=dataset, results=validation_results
            )
            i_s.view()
            is_results = i_s.get_results()

            # Compute calibration metrics for probabilistic models
            calibration_metrics = None
            if config.lstm.probabilistic:
                try:
                    calibration_metrics = model.compute_calibration_metrics(stage="val")
                except Exception as e:
                    print(f"Warning: Could not compute calibration metrics: {e}")

                # Plot uncertainty heatmap and save
                heatmap_path = Path(
                    f"results/question_1/ml/{ticker}_uncertainty_heatmap.png"
                )
                heatmap_path.parent.mkdir(parents=True, exist_ok=True)
                # model.plot_uncertainty_heatmap(stage="val", save_path=heatmap_path)
                # model.plot_series_with_uncertainty(
                #     feature_name="Net Income",
                #     save_path=Path(
                #         f"results/question_1/ml/{ticker}_net_income_uncertainty.png"
                #     ),
                #     n_periods=20,
                # )

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
            # Store final seasonal weights if learned
            if config.data.learnable_seasonal_weight:
                results[ticker]["final_seasonal_weight"] = model.get_seasonal_weight()

            # Now predict the future on the final lookback window dataset.predict_dataset
            predictions = model.predict(dataset.predict_dataset)

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

    # Summarise results
    summary, valid_tickers = compute_results_summary(results, tickers, config_name)
    vals = [results[ticker]["bs_pct_error"] for ticker in valid_tickers]
    mean_val = np.mean(vals)
    std_val = np.std(vals)
    summary["bs_pct_error"] = {"mean": mean_val, "std": std_val}

    print(f"{'Metric':<30} {'Mean MAE (bn)':<15} {'Std Dev (bn)':<15}")
    print("-" * 60)
    for key, stats in summary.items():
        print(f"{key:<30} {stats['mean'] / 1e9:<15.2f} {stats['std'] / 1e9:<15.2f}")

    # Print calibration metrics if probabilistic
    if variation["probabilistic"] and "calibration" in results[valid_tickers[0]]:
        print(f"\n{'Calibration Metric':<30} {'Mean':<15} {'Std Dev':<15}")
        print("-" * 60)

        # Calibration error
        cal_errors = [
            results[ticker]["calibration"]["calibration_error"]
            for ticker in valid_tickers
            if "calibration" in results[ticker]
            and "calibration_error" in results[ticker]["calibration"]
        ]
        if cal_errors:
            print(
                f"{'Calibration Error':<30} {np.mean(cal_errors):<15.4f} "
                f"{np.std(cal_errors):<15.4f}"
            )

        # Sharpness
        sharpness_vals = [
            results[ticker]["calibration"]["sharpness"]
            for ticker in valid_tickers
            if "calibration" in results[ticker]
            and "sharpness" in results[ticker]["calibration"]
        ]
        if sharpness_vals:
            print(
                f"{'Sharpness (95% PI width)':<30} {np.mean(sharpness_vals) / 1e9:<15.2f} "
                f"{np.std(sharpness_vals) / 1e9:<15.2f}"
            )

        # CRPS
        crps_vals = [
            results[ticker]["calibration"]["crps"]
            for ticker in valid_tickers
            if "calibration" in results[ticker]
            and "crps" in results[ticker]["calibration"]
            and results[ticker]["calibration"]["crps"] is not None
        ]
        if crps_vals:
            print(
                f"{'CRPS':<30} {np.mean(crps_vals) / 1e9:<15.2f} "
                f"{np.std(crps_vals) / 1e9:<15.2f}"
            )

print("\n" + "=" * 80)
print("FINAL COMPARISON ACROSS ALL CONFIGURATIONS")
print("=" * 80)

# Only successful tickers
successful_tickers = [t for t in tickers if t not in failed]

for config_name, results in all_config_results.items():
    print(f"\n{config_name}:")
    lstm_mae_vals = [
        results[ticker]["balance_sheet"]["lstm"] for ticker in successful_tickers
    ]
    bs_pct_vals = [results[ticker]["bs_pct_error"] for ticker in successful_tickers]

    # Check flags across all tickers
    is_probabilistic = any(
        results[ticker].get("probabilistic", False) for ticker in successful_tickers
    )
    is_seasonal_learned = any(
        results[ticker].get("final_seasonal_weight") is not None
        for ticker in successful_tickers
    )

    print(
        f"  BS LSTM MAE: {np.mean(lstm_mae_vals) / 1e9:.2f}bn "
        f"(+/- {np.std(lstm_mae_vals) / 1e9:.2f}bn)"
    )
    print(f"  BS Identity: {np.mean(bs_pct_vals):.2%} (+/- {np.std(bs_pct_vals):.2%})")

    if is_probabilistic:
        cal_errors = [
            results[ticker]["calibration"]["calibration_error"]
            for ticker in successful_tickers
            if "calibration" in results[ticker]
        ]
        if cal_errors:
            print(
                f"  Calibration Error:      "
                f"{np.mean(cal_errors):.4f} (+/- {np.std(cal_errors):.4f})"
            )

        crps_vals = [
            results[ticker]["calibration"]["crps"]
            for ticker in successful_tickers
            if "calibration" in results[ticker]
            and results[ticker]["calibration"]["crps"] is not None
        ]
        if crps_vals:
            print(
                f"  CRPS:                   "
                f"{np.mean(crps_vals) / 1e9:.2f}bn (+/- {np.std(crps_vals) / 1e9:.2f}bn)"
            )

    if is_seasonal_learned:
        seasonal_weights = [
            results[ticker]["final_seasonal_weight"]
            for ticker in successful_tickers
            if results[ticker].get("final_seasonal_weight") is not None
        ]

        if seasonal_weights:
            seasonal_weights = np.asarray(seasonal_weights)
            mean_w = seasonal_weights.mean(axis=0)
            std_w = seasonal_weights.std(axis=0)
            print(f"  Final seasonal weights {mean_w:.4f} (+/- {std_w:.4f})")

with open(results_dir / "lstm_evaluation.json", "w") as f:
    json.dump(all_config_results, f, indent=2)
