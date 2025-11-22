from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence

import numpy as np


@dataclass
class Metric:
    value: float
    mae: float
    pct: float
    diagnostics: Dict[str, float] = field(default_factory=dict)

    def add_diagnostics(
        self,
        actual: Sequence[float],
        predicted: Sequence[float],
        baseline: Sequence[float] | None = None,
        naive_lag: int = 1,
        dm_baseline: Sequence[float] | None = None,
    ) -> Dict[str, float]:
        """
        Compute extended metrics and store them in `diagnostics`.
        """
        diag: Dict[str, float] = {}
        if baseline is None:
            baseline = actual
        try:
            diag["naive_u"] = float(naive_baseline_error(actual, predicted, naive_lag))
        except ValueError:
            diag["naive_u"] = np.nan

        diag["directional_accuracy"] = float(directional_accuracy(actual, predicted))

        try:
            diag["moce"] = float(magnitude_of_change_error(actual, predicted))
        except ValueError:
            diag["moce"] = np.nan

        try:
            diag["resid_autocorr"] = float(residual_autocorrelation(actual, predicted))
        except ValueError:
            diag["resid_autocorr"] = np.nan

        try:
            diag["scale_free_change_error"] = float(
                scale_free_change_error(actual, predicted)
            )
        except ValueError:
            diag["scale_free_change_error"] = np.nan

        if dm_baseline is not None:
            try:
                diag["dm_stat"] = float(
                    diebold_mariano(actual, predicted, dm_baseline, loss="mse")
                )
            except ValueError:
                diag["dm_stat"] = np.nan
        self.diagnostics.update(diag)
        return diag

    def get_diagnostic(self, name: str) -> float | None:
        """Return a stored diagnostic metric by name if available."""
        return self.diagnostics.get(name)


@dataclass
class TickerResults:
    """
    Per-ticker results:
    - aggregated sections: assets / liabilities / equity
    - per-feature metrics.
    """

    assets: Metric
    liabilities: Metric
    equity: Metric
    features: Dict[str, Metric]

    def feature_values(self) -> Dict[str, float]:
        return {name: m.value for name, m in self.features.items()}

    def feature_diagnostics(self, diagnostic: str) -> Dict[str, float | None]:
        """Collect a diagnostic metric across all tracked features."""
        return {
            name: metric.get_diagnostic(diagnostic)
            for name, metric in self.features.items()
        }


def naive_baseline_error(
    actual: Sequence[float], predicted: Sequence[float], naive_lag: int = 1
) -> float:
    """
    Compare a forecast vs. naive baseline that repeats the last value.
    Returns ratio of model MAE to naive MAE (Theil's U-like).
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("actual/predicted must match shape")
    if len(actual) <= naive_lag:
        raise ValueError("Need more observations than naive_lag")
    actual_trim = actual[naive_lag:]
    predicted_trim = predicted[naive_lag:]
    baseline = actual[:-naive_lag]
    mae_model = np.mean(np.abs(actual_trim - predicted_trim))
    mae_naive = np.mean(np.abs(actual_trim - baseline))
    return mae_model / mae_naive if mae_naive else np.inf


def directional_accuracy(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Share of times the predicted change matches the sign of actual change."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("actual/predicted must match shape")
    actual_diff = np.sign(np.diff(actual))
    predicted_diff = np.sign(np.diff(predicted))
    matches = actual_diff == predicted_diff
    return np.mean(matches) if len(matches) else np.nan


def magnitude_of_change_error(
    actual: Sequence[float], predicted: Sequence[float]
) -> float:
    """Mean absolute error on first differences."""
    actual_diff = np.diff(np.asarray(actual, dtype=float))
    predicted_diff = np.diff(np.asarray(predicted, dtype=float))
    if actual_diff.size == 0 or actual_diff.size != predicted_diff.size:
        raise ValueError("Need >=2 points and matching shapes for MoCE")
    return np.mean(np.abs(actual_diff - predicted_diff))


def residual_autocorrelation(
    actual: Sequence[float], predicted: Sequence[float], lag: int = 1
) -> float:
    """Compute autocorrelation of residuals at a specified lag."""
    residuals = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    if len(residuals) <= lag:
        raise ValueError("lag must be smaller than number of residuals")
    resid_mean = residuals.mean()
    numerator = np.sum((residuals[:-lag] - resid_mean) * (residuals[lag:] - resid_mean))
    denominator = np.sum((residuals - resid_mean) ** 2)
    return numerator / denominator if denominator else np.nan


def scale_free_change_error(
    actual: Sequence[float], predicted: Sequence[float], epsilon: float = 1e-8
) -> float:
    """
    Scale-free accuracy on changes: mean absolute percentage error on differences.
    """
    actual_diff = np.diff(np.asarray(actual, dtype=float))
    predicted_diff = np.diff(np.asarray(predicted, dtype=float))
    if actual_diff.size == 0 or actual_diff.size != predicted_diff.size:
        raise ValueError("Need >=2 points and matching shapes for change error")
    return np.mean(
        np.abs(actual_diff - predicted_diff) / (np.abs(actual_diff) + epsilon)
    )


def diebold_mariano(
    actual: Sequence[float],
    forecast_a: Sequence[float],
    forecast_b: Sequence[float],
    loss: str = "mse",
) -> float:
    """
    Diebold-Mariano statistic comparing forecast_a vs forecast_b.
    Returns the DM test statistic (higher magnitude => more significant difference).
    """
    actual = np.asarray(actual, dtype=float)
    forecast_a = np.asarray(forecast_a, dtype=float)
    forecast_b = np.asarray(forecast_b, dtype=float)
    if not (actual.size == forecast_a.size == forecast_b.size):
        raise ValueError("actual and forecasts must have equal length")

    if loss == "mse":
        loss_a = (actual - forecast_a) ** 2
        loss_b = (actual - forecast_b) ** 2
    elif loss == "mae":
        loss_a = np.abs(actual - forecast_a)
        loss_b = np.abs(actual - forecast_b)
    else:
        raise ValueError("loss must be 'mse' or 'mae'")

    d_t = loss_a - loss_b
    mean_diff = np.mean(d_t)
    n = len(d_t)
    autocov = np.array(
        [
            np.sum((d_t[: n - lag] - mean_diff) * (d_t[lag:] - mean_diff)) / n
            for lag in range(n)
        ]
    )
    variance = autocov[0] + 2 * np.sum(autocov[1:])
    if variance <= 0:
        return np.nan
    dm_stat = mean_diff / np.sqrt(variance / n)
    return dm_stat
