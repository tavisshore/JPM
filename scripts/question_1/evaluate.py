"""Run the LSTM forecaster multiple times and aggregate validation metrics."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np

from jpm.question_1.config import Config
from jpm.question_1.data.ed import EdgarDataLoader
from jpm.question_1.misc import set_seed
from jpm.question_1.models.lstm import LSTMForecaster


def _predict_unscaled(
    model: LSTMForecaster, data: EdgarDataLoader
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model on the validation set and return unscaled preds,
    targets, history."""

    ds = data.val_dataset if data.val_dataset is not None else data.train_dataset

    x_batches = []
    y_batches = []
    for x_batch, y_batch in ds:
        x_batches.append(x_batch.numpy())
        y_batches.append(y_batch.numpy())

    history = np.concatenate(x_batches, axis=0)
    y_gt = np.concatenate(y_batches, axis=0)

    if model.config.model.probabilistic:
        if model.config.model.mc_samples > 1:
            dists = [
                model.model(history, training=False)
                for _ in range(model.config.model.mc_samples)
            ]
            means = np.stack([d.mean().numpy() for d in dists], axis=0)
            y_pred = np.mean(means, axis=0)
        else:
            dist = model.model(history, training=False)
            y_pred = dist.mean().numpy()
    elif model.config.model.variational and model.config.model.mc_samples > 1:
        samples = [
            model.model.predict(history, verbose=0)
            for _ in range(model.config.model.mc_samples)
        ]
        y_pred = np.mean(samples, axis=0)
    else:
        y_pred = model.model.predict(history, verbose=0)

    y_pred_unscaled = y_pred * data.target_std + data.target_mean
    y_gt_unscaled = y_gt * data.target_std + data.target_mean
    history_unscaled = history * data.target_std + data.target_mean
    return y_pred_unscaled, y_gt_unscaled, history_unscaled


def _identity_pct_error(pred_unscaled: np.ndarray, data: EdgarDataLoader) -> float:
    """Compute mean percentage violation of Assets = Liabilities + Equity."""

    assets = np.sum(pred_unscaled[:, data.feature_mappings["assets"]], axis=-1)
    liabilities = np.sum(
        pred_unscaled[:, data.feature_mappings["liabilities"]], axis=-1
    )
    equity = np.sum(pred_unscaled[:, data.feature_mappings["equity"]], axis=-1)

    eps = 1e-12
    violation = assets - (liabilities + equity)
    denom = np.abs(assets) + np.abs(liabilities) + np.abs(equity) + eps
    rel_violation = np.abs(violation) / denom
    return float(rel_violation.mean() * 100.0)


def _net_income_pct_error(
    pred_unscaled: np.ndarray, gt_unscaled: np.ndarray, data: EdgarDataLoader
) -> float:
    """Compute percentage error for net income if that feature exists; else NaN."""

    key = "net_income_loss"
    if key not in data.feat_to_idx:
        return float("nan")

    idx = data.feat_to_idx[key]
    pred = float(pred_unscaled[:, idx].mean())
    gt = float(gt_unscaled[:, idx].mean())
    denom = abs(gt) if abs(gt) > 1e-12 else 1e-12
    return abs(pred - gt) / denom * 100.0


def run_once(base_config: Config) -> Dict[str, float]:
    config = deepcopy(base_config)
    data = EdgarDataLoader(config=config)
    model = LSTMForecaster(config=config, data=data)
    model.fit(verbose=0)
    results = model.evaluate(stage="val")

    pred_unscaled, gt_unscaled, history_unscaled = _predict_unscaled(model, data)

    metrics = {
        "model_mae": results.model_mae,
        "net_income_pct_error": _net_income_pct_error(pred_unscaled, gt_unscaled, data),
        "identity_pct_error": _identity_pct_error(pred_unscaled, data),
        "net_income_pred": float(results.net_income_pred),
    }

    # For report
    metrics.update({f"baseline_mae_{k}": v for k, v in results.baseline_mae.items()})
    metrics.update(
        {
            f"net_income_baseline_pred_{k}": float(v)
            for k, v in results.net_income_baseline_pred.items()
        }
    )
    return metrics


def aggregate(runs: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Compute mean and std for each metric across runs."""

    keys = runs[0].keys()
    summary = {}
    for k in keys:
        values = np.array([r[k] for r in runs], dtype=np.float64)
        summary[k] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
    )
    args = parser.parse_args()

    set_seed(42)

    # Sort this later, just edit the config there - ablation etc.
    base_config = Config()
    all_runs = []

    for i in range(args.runs):
        print(f"Run {i + 1}/{args.runs}...", flush=True)
        metrics = run_once(base_config)
        all_runs.append(metrics)

    summary = aggregate(all_runs)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    # There seems to be an issue - getting worse results
    main()
