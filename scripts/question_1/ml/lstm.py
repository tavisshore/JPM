import gc
import json
import os
import sys
from dataclasses import replace

import numpy as np

from jpm.question_1 import (
    BalanceSheet,
    Config,
    DataConfig,
    EdgarData,
    IncomeStatement,
    LLMConfig,
    LSTMConfig,
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


CONFIG_VARIATIONS = [
    {"learn_identity": False, "enforce_balance": False},
    {"learn_identity": True, "enforce_balance": False},
    {"learn_identity": True, "enforce_balance": True},
]


tickers = """
AAPL MSFT GOOGL AMZN NVDA META TSLA BRK.B UNH XOM JNJ JPM V PG MA HD CVX ABBV MRK AVGO
COST PEP ADBE LLY WMT TMO CSCO MCD ACN CRM ABT NFLX DHR TXN NKE DIS ORCL VZ INTC
CMCSA PFE PM NEE WFC UPS COP RTX IBM BA QCOM AMD INTU NOW CAT GS MS SPGI LOW AXP
ISRG HON BKNG TJX BLK AMAT GILD SYK DE LRCX VRTX PLD MMC MDLV CI C ADI REGN SCHW
ZTS ETN CB SO MU AON SLB BSX FISV BMY DUK ITW BDX PNC CME APD EQIX USB ICE MCO
MMM GE COF NSC TGT HCA PYPL PH EOG MAR MO EMR NOC CSX TFC WM PSA WELL KLAC KMI
APH CCI ROP SHW HUM ORLY GM ADSK NXPI F PCAR AIG MET ALL AEP AJG ROST KMB SRE
EW CTAS CARR MSCI PAYX IDXX AFL DD FCX FTNT A O PRU CTVA ODFL RSG YUM KDP FAST
CDNS SNPS GWW MNST CHTR VRSK EL MCHP RCL CPRT GD CMG ANSS DAL AME IT BIIB LHX
""".split()

all_config_results = {}
failed = []
for var_idx, variation in enumerate(CONFIG_VARIATIONS, 1):
    config_name = f"learn_identity={variation['learn_identity']}, \
        enforce_balance={variation['enforce_balance']}"
    print(f"\n{'=' * 65}")
    print(f"Config {var_idx}/{len(CONFIG_VARIATIONS)}: {config_name}")
    print(f"{'=' * 65}")

    lstm_cfg_var = replace(
        lstm_cfg,
        enforce_balance=variation["enforce_balance"],
        learn_identity=variation["learn_identity"],
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

            bs = BalanceSheet(
                config=config, data=data, dataset=dataset, results=validation_results
            )
            bs_pct_error = bs.check_identity()

            i_s = IncomeStatement(
                config=config, dataset=dataset, results=validation_results
            )
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
            del model, data, dataset, bs, i_s, validation_results, is_results

            gc.collect()
            tf.keras.backend.clear_session()

        except Exception as E:
            print(E)
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

with open("temp/lstm_evaluation_results_all_configs.json", "w") as f:
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

print()
print(failed)
