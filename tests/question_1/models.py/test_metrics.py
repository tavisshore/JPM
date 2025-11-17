import pytest

from jpm.question_1.models.metrics import Metric, TickerResults

unit = pytest.mark.unit


@unit
def test_ticker_results_feature_values_returns_metric_values():
    """feature_values should return value components keyed by feature name."""
    features = {
        "cash": Metric(value=100.5, mae=1.2, pct=0.01),
        "debt": Metric(value=250.0, mae=2.5, pct=0.02),
    }
    ticker_results = TickerResults(
        assets=Metric(0, 0, 0),
        liabilities=Metric(0, 0, 0),
        equity=Metric(0, 0, 0),
        features=features,
    )

    assert ticker_results.feature_values() == {
        "cash": pytest.approx(100.5),
        "debt": pytest.approx(250.0),
    }
