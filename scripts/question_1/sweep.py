from __future__ import annotations

import itertools
import json
from copy import deepcopy

from jpm.question_1.config import Config
from jpm.question_1.data.ed import EdgarDataLoader
from jpm.question_1.models.lstm import LSTMForecaster

GRID = {
    "data.batch_size": [32],
    "data.lookback": [5, 9],
    "data.horizon": [1],
    "data.seasonal_weight": [x / 100.0 for x in range(100, 120, 1)],
    "model.lstm_units": [256],
    "model.lstm_layers": [2],
    "model.dense_units": [256],
    "model.dropout": [0.1],
    "model.variational": [False],
    "training.lr": [1e-4],
    "training.epochs": [500],
    "loss.enforce_balance": [False],
    "loss.learn_identity": [True],
    "loss.identity_weight": [1e-4],
    "loss.learn_subtotals": [False],
    "loss.subcategory_weight": [1e-5],
}


def apply_config_value(cfg: Config, key: str, value) -> Config:
    """Apply a dot-delimited key to a Config copy and return it."""
    target, field = key.split(".")
    section = deepcopy(getattr(cfg, target))
    setattr(section, field, value)
    setattr(cfg, target, section)
    return cfg


def main():
    base = Config()
    keys, values = zip(*GRID.items(), strict=True)

    best = None
    for combo in itertools.product(*values):
        cfg = deepcopy(base)
        for k, v in zip(keys, combo, strict=True):
            cfg = apply_config_value(cfg, k, v)

        data = EdgarDataLoader(cfg)
        model = LSTMForecaster(cfg, data)
        history = model.fit()

        hist = history.history
        metric_name = "val_mae" if "val_mae" in hist else "mae"
        val_mae = float(hist[metric_name][-1])

        result = {
            "config": dict(zip(keys, combo, strict=True)),
            "val_mae": val_mae,
        }
        print(json.dumps(result))

        if best is None or val_mae < best["val_mae"]:
            best = result

    print("best:", json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
