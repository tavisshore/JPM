from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping

import numpy as np


@dataclass
class Metric:
    """Container for a single metric value, ground truth, and absolute error."""

    value: float
    mae: float
    gt: float = 0.0


@dataclass
class TickerResults:
    """Aggregated per-ticker results, including per-feature metrics and baselines."""

    assets: Metric
    liabilities: Metric
    equity: Metric
    features: Dict[str, Metric]
    model_mae: float = 0.0
    baseline_mae: Dict[str, float] = field(default_factory=dict)
    skill: Dict[str, float] = field(default_factory=dict)
    net_income_model_mae: float = 0.0
    net_income_baseline_mae: Dict[str, float] = field(default_factory=dict)
    net_income_skill: Dict[str, float] = field(default_factory=dict)
    net_income_pred: float = 0.0
    net_income_gt: float = 0.0
    net_income_baseline_pred: Dict[str, float] = field(default_factory=dict)

    def feature_values(self) -> Dict[str, float]:
        return {name: m.value for name, m in self.features.items()}


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_baseline_predictions(
    history: np.ndarray, seasonal_lag: int = 4
) -> Dict[str, np.ndarray]:
    """Build simple one-step-ahead baselines from a history window."""
    if history.ndim != 3:
        raise ValueError("history must have shape (batch, lookback, features)")
    if seasonal_lag < 1:
        raise ValueError("seasonal_lag must be >= 1")

    last_value = history[:, -1, :]
    global_mean = np.mean(history, axis=(0, 1))
    global_mean_pred = np.broadcast_to(global_mean, last_value.shape)

    if history.shape[1] >= seasonal_lag:
        seasonal_naive = history[:, -seasonal_lag, :]
    else:
        seasonal_naive = last_value

    return {
        "global_mean": global_mean_pred,
        "last_value": last_value,
        "seasonal_naive": seasonal_naive,
    }


def baseline_skill_scores(
    y_true: np.ndarray,
    model_pred: np.ndarray,
    history: np.ndarray,
    seasonal_lag: int = 4,
) -> Dict[str, Mapping[str, float]]:
    """Compare model predictions to baselines using MAE-derived skill scores."""
    if y_true.shape != model_pred.shape:
        raise ValueError("y_true and model_pred must have the same shape")
    if history.shape[0] != y_true.shape[0]:
        raise ValueError("history batch dimension must match y_true/model_pred")

    baselines = compute_baseline_predictions(history, seasonal_lag=seasonal_lag)
    model_mae = _mae(y_true, model_pred)

    baseline_mae: Dict[str, float] = {}
    skills: Dict[str, float] = {}
    eps = 1e-12

    for name, baseline_pred in baselines.items():
        mae = _mae(y_true, baseline_pred)
        baseline_mae[name] = mae
        denom = mae if mae > eps else eps
        skills[name] = 1.0 - model_mae / denom

    return {
        "model_mae": model_mae,
        "baseline_mae": baseline_mae,
        "skill": skills,
    }
