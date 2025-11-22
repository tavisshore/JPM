import numpy as np
import pytest

from jpm.question_1.models.metrics import (
    Metric,
    TickerResults,
    baseline_skill_scores,
    compute_baseline_predictions,
)

unit = pytest.mark.unit


@unit
def test_ticker_results_feature_values_returns_metric_values():
    """feature_values should return value components keyed by feature name."""
    features = {
        "cash": Metric(value=100.5, mae=1.2),
        "debt": Metric(value=250.0, mae=2.5),
    }
    ticker_results = TickerResults(
        assets=Metric(0, 0),
        liabilities=Metric(0, 0),
        equity=Metric(0, 0),
        features=features,
    )

    assert ticker_results.feature_values() == {
        "cash": pytest.approx(100.5),
        "debt": pytest.approx(250.0),
    }


@unit
def test_compute_baseline_predictions_shapes_and_values():
    history = np.array(
        [
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
            [[2.0, 10.0], [4.0, 20.0], [6.0, 30.0]],
        ]
    )  # shape (2, 3, 2)

    baselines = compute_baseline_predictions(history, seasonal_lag=2)

    assert baselines["last_value"].shape == (2, 2)
    # Global mean across batch and time
    np.testing.assert_allclose(baselines["global_mean"][0], [3.0, 20.0])
    # Seasonal lag of 2 steps back
    np.testing.assert_allclose(baselines["seasonal_naive"], [[2.0, 20.0], [4.0, 20.0]])


@unit
def test_baseline_skill_scores_reports_positive_skill_when_better_than_baselines():
    history = np.array(
        [
            [[1.0], [2.0], [3.0], [4.0]],
            [[2.0], [4.0], [6.0], [8.0]],
        ]
    )
    y_true = np.array([[5.0], [10.0]])
    model_pred = np.array([[5.5], [9.5]])  # MAE = 0.5

    results = baseline_skill_scores(
        y_true=y_true, model_pred=model_pred, history=history, seasonal_lag=4
    )

    assert pytest.approx(results["model_mae"]) == 0.5
    assert set(results["baseline_mae"]) == {
        "global_mean",
        "last_value",
        "seasonal_naive",
    }
    # Model is better than all baselines so skill should be > 0
    assert all(score > 0 for score in results["skill"].values())
