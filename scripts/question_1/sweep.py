from __future__ import annotations

import gc
import itertools
import json
from copy import deepcopy

import tensorflow as tf

from jpm.question_1 import Config, EdgarDataLoader, LSTMForecaster, set_seed

GRID = {
    "data.batch_size": [32],
    "data.lookback": [3, 4],
    "data.horizon": [1],
    "data.periods": [20],
    "data.seasonal_weight": [1.1],
    "model.lstm_units": [256],
    "model.lstm_layers": [2],
    "model.dense_units": [256],
    "model.dropout": [0.1],
    "model.variational": [False],
    "training.lr": [1e-4],
    "training.epochs": [250],
    "loss.enforce_balance": [False, True],
    "loss.learn_identity": [False, True],
    "loss.identity_weight": [1e-4],
}


def apply_config_value(cfg: Config, key: str, value) -> Config:
    """Apply a dot-delimited key to a Config copy and return it."""
    target, field = key.split(".")
    section = deepcopy(getattr(cfg, target))
    setattr(section, field, value)
    setattr(cfg, target, section)
    return cfg


def main():
    set_seed(42)

    tickers = [
        "NVDA",
        "AAPL",
        "AMZN",
        "AVGO",
        "META",
        "TSLA",
        "BRK.B",
        "WMT",
        "PLTR",
        "GS",
    ]

    base = Config()
    keys, values = zip(*GRID.items(), strict=True)

    best = None
    for combo in itertools.product(*values):
        cfg = deepcopy(base)
        for k, v in zip(keys, combo, strict=True):
            cfg = apply_config_value(cfg, k, v)

        val_maes = []
        for _, ticker in enumerate(tickers, 1):
            cfg.data.ticker = ticker
            data = EdgarDataLoader(config=cfg)
            model = LSTMForecaster(config=cfg, data=data)
            history = model.fit(verbose=0)
            val_maes.append(float(history.history["val_mae"][-1]))

            # Clear memory after each ticker
            del model, data, history
            gc.collect()
            tf.keras.backend.clear_session()

        val_mae = sum(val_maes) / len(val_maes)

        result = {
            "config": dict(zip(keys, combo, strict=True)),
            "val_mae": val_mae,
        }

        if best is None or val_mae < best["val_mae"]:
            best = result

    print("best:", json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
