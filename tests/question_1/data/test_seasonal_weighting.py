import numpy as np
import pandas as pd


def test_seasonal_weight_applied(monkeypatch, tmp_path):
    """
    Ensure seasonal weighting scales the intended lagged timestep in the train windows.
    """
    monkeypatch.setenv("EDGAR_EMAIL", "test@example.com")

    from jpm.question_1.config import Config, DataConfig
    from jpm.question_1.data import utils as data_utils
    from jpm.question_1.data.ed import EdgarDataLoader

    class DummyScaler:
        def fit_transform(self, X):
            self.mean_ = np.zeros(X.shape[1], dtype="float64")
            self.scale_ = np.ones(X.shape[1], dtype="float64")
            return np.asarray(X, dtype="float64")

    monkeypatch.setattr("jpm.question_1.data.ed.StandardScaler", DummyScaler)

    bs_cols = data_utils.get_leaf_values(data_utils.get_bs_structure())
    T = 6
    df = pd.DataFrame({col: np.arange(T, dtype=float) for col in bs_cols})

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = cache_dir / "AAPL.parquet"
    df.to_parquet(parquet_path)

    data_cfg = DataConfig(
        cache_dir=str(cache_dir),
        lookback=5,
        horizon=1,
        withhold_periods=0,
        seasonal_weight=2.0,
        seasonal_lag=4,
        target_type="full",
    )
    config = Config(data=data_cfg)

    loader = EdgarDataLoader(config=config)

    X_batch, _ = next(iter(loader.train_dataset))
    X = X_batch.numpy()

    assert X.shape[1] == data_cfg.lookback

    # Only the seasonal lag timestep should be scaled by seasonal_weight
    seasonal_idx = data_cfg.lookback - data_cfg.seasonal_lag
    expected = np.stack(
        [np.full(len(bs_cols), t, dtype=float) for t in range(data_cfg.lookback)],
        axis=0,
    )
    expected[seasonal_idx] *= data_cfg.seasonal_weight

    np.testing.assert_allclose(X[0], expected)
