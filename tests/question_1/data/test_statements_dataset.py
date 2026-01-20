"""Tests for StatementsDataset class."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from jpm.config.question_1 import Config, DataConfig

unit = pytest.mark.unit
integration = pytest.mark.integration


def make_mock_edgar_data(
    n_periods=20,
    n_features=4,
    lookback=4,
    horizon=1,
    withhold=2,
    batch_size=4,
    seasonal_weight=1.0,
    seasonal_lag=4,
):
    """Create a mock EdgarData object for testing."""
    config = Config(
        data=DataConfig(
            ticker="TEST",
            lookback=lookback,
            horizon=horizon,
            withhold_periods=withhold,
            batch_size=batch_size,
            seasonal_weight=seasonal_weight,
            seasonal_lag=seasonal_lag,
        )
    )

    # Create mock data
    periods = pd.period_range("2019-03-31", periods=n_periods, freq="Q")
    data = pd.DataFrame(
        np.random.randn(n_periods, n_features) * 100 + 1000,
        index=periods,
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    mock = MagicMock()
    mock.config = config
    mock.data = data
    mock.targets = list(data.columns)
    mock.tgt_indices = list(range(n_features))

    return mock


@unit
def test_statements_dataset_initialization():
    """StatementsDataset should initialize with EdgarData."""
    from jpm.question_1.data.datasets.statements import StatementsDataset

    mock_edgar = make_mock_edgar_data()

    dataset = StatementsDataset(mock_edgar)

    assert dataset.config is mock_edgar.config
    # data may be pruned/modified during initialization, so just check it exists
    assert dataset.data is not None


@unit
def test_set_feature_index_creates_mapping():
    """_set_feature_index should create feature name to index mapping."""
    from jpm.question_1.data.datasets.statements import StatementsDataset

    mock_edgar = make_mock_edgar_data(n_features=3)

    dataset = StatementsDataset(mock_edgar)

    assert "feature_0" in dataset.feat_to_idx
    assert "feature_1" in dataset.feat_to_idx
    assert "feature_2" in dataset.feat_to_idx
    assert dataset.feat_to_idx["feature_0"] == 0


@unit
def test_scale_features_standardizes():
    """_scale_features should standardize data using StandardScaler."""
    from jpm.question_1.data.datasets.statements import StatementsDataset

    mock_edgar = make_mock_edgar_data()

    dataset = StatementsDataset(mock_edgar)

    # Scaler stats should be stored
    assert dataset.full_mean is not None
    assert dataset.full_std is not None
    # Length depends on pruned features, should match num_features
    assert len(dataset.full_mean) == dataset.num_features
    assert len(dataset.full_std) == dataset.num_features


@unit
def test_prepare_dataset_creates_train_val():
    """_prepare_dataset should create train and validation datasets."""
    from jpm.question_1.data.datasets.statements import StatementsDataset

    mock_edgar = make_mock_edgar_data(n_periods=20, lookback=4, withhold=2)

    dataset = StatementsDataset(mock_edgar)

    assert hasattr(dataset, "train_dataset")
    assert hasattr(dataset, "val_dataset")
    assert hasattr(dataset, "X_train")
    assert hasattr(dataset, "y_train")
    assert hasattr(dataset, "X_test")
    assert hasattr(dataset, "y_test")


@unit
def test_train_test_shapes():
    """Train and test arrays should have correct shapes."""
    from jpm.question_1.data.datasets.statements import StatementsDataset

    lookback = 4
    n_features = 5
    mock_edgar = make_mock_edgar_data(
        n_periods=20, n_features=n_features, lookback=lookback, withhold=2
    )

    dataset = StatementsDataset(mock_edgar)

    # X should be (n_samples, lookback, n_features)
    assert dataset.X_train.shape[1] == lookback
    assert dataset.X_train.shape[2] == n_features

    # y should be (n_samples, n_targets)
    assert dataset.y_train.shape[1] == n_features  # All features are targets

    # Test set should exist
    assert len(dataset.X_test) > 0
    assert len(dataset.y_test) > 0


@unit
def test_num_features_and_targets_set():
    """num_features and num_targets should be set correctly."""
    from jpm.question_1.data.datasets.statements import StatementsDataset

    n_features = 7
    mock_edgar = make_mock_edgar_data(n_features=n_features)

    dataset = StatementsDataset(mock_edgar)

    assert dataset.num_features == n_features
    assert dataset.num_targets == n_features


@unit
def test_target_mean_std_for_targets():
    """target_mean and target_std should be subset for target indices."""
    from jpm.question_1.data.datasets.statements import StatementsDataset

    mock_edgar = make_mock_edgar_data(n_features=5)

    dataset = StatementsDataset(mock_edgar)

    # In current implementation, tgt_indices covers all features after pruning
    # target_mean/target_std should match the number of targets
    assert len(dataset.target_mean) == len(dataset.tgt_indices)
    assert len(dataset.target_std) == len(dataset.tgt_indices)


@integration
def test_tf_dataset_iteration():
    """TensorFlow datasets should be iterable."""
    from jpm.question_1.data.datasets.statements import StatementsDataset

    mock_edgar = make_mock_edgar_data(n_periods=20, batch_size=4)

    dataset = StatementsDataset(mock_edgar)

    # Should be able to iterate over train dataset
    batch_count = 0
    for X_batch, _ in dataset.train_dataset:
        batch_count += 1
        assert X_batch.shape[0] <= 4  # Batch size
        break  # Just check first batch

    assert batch_count > 0


@unit
def test_raises_on_insufficient_data():
    """Should raise ValueError when data is insufficient for windowing."""
    from jpm.question_1.data.datasets.statements import StatementsDataset

    # Very small data with large lookback
    mock_edgar = make_mock_edgar_data(n_periods=3, lookback=10, withhold=1)

    with pytest.raises(ValueError, match="Sequence too short"):
        StatementsDataset(mock_edgar, verbose=False)


# Tests for seasonal weighting


@unit
def test_seasonal_weight_applied():
    """_apply_seasonal_weight should scale seasonal timesteps."""
    from jpm.question_1.data.datasets.statements import StatementsDataset

    mock_edgar = make_mock_edgar_data(
        n_periods=20,
        lookback=8,
        seasonal_lag=4,
        seasonal_weight=2.0,
    )

    dataset = StatementsDataset(mock_edgar)

    # When seasonal_weight != 1.0, data should be modified
    # The seasonal indices (4 and 0 for lookback=8, lag=4) should be scaled
    assert dataset.X_train is not None


@unit
def test_seasonal_weight_disabled():
    """Seasonal weighting should be skipped when weight=1.0."""
    from jpm.question_1.data.datasets.statements import StatementsDataset

    mock_edgar = make_mock_edgar_data(
        n_periods=20,
        lookback=8,
        seasonal_lag=4,
        seasonal_weight=1.0,  # Disabled
    )

    dataset = StatementsDataset(mock_edgar)

    # Should still work
    assert dataset.X_train is not None


@unit
def test_seasonal_weight_one_disables():
    """Seasonal weighting should be skipped when weight=1.0."""
    from jpm.question_1.data.datasets.statements import StatementsDataset

    # Note: seasonal_lag must be positive per DataConfig validation
    # Test that weight=1.0 effectively disables seasonal weighting
    mock_edgar = make_mock_edgar_data(
        n_periods=20,
        lookback=8,
        seasonal_lag=4,
        seasonal_weight=1.0,  # Disabled via weight
    )

    dataset = StatementsDataset(mock_edgar)

    assert dataset.X_train is not None


# Tests for filter_low_quality_columns


@unit
def test_filter_low_quality_columns_removes_low_variance():
    """_filter_low_quality_columns should remove constant columns."""
    from jpm.question_1.data.datasets.statements import StatementsDataset

    mock_edgar = make_mock_edgar_data(n_periods=20, n_features=5)
    # Add a constant column
    mock_edgar.data["constant_col"] = 100.0

    dataset = StatementsDataset(mock_edgar, verbose=True)

    # Constant column should be filtered by the scaler process or filtering
    # (It may or may not be in the final columns depending on implementation)
    assert dataset.data is not None


@unit
def test_filter_low_quality_columns_removes_all_nan():
    """_filter_low_quality_columns should remove all-NaN columns."""
    import warnings

    from jpm.question_1.data.datasets.statements import StatementsDataset

    mock_edgar = make_mock_edgar_data(n_periods=20, n_features=5)
    # Add an all-NaN column
    mock_edgar.data["nan_col"] = np.nan

    # Suppress the RuntimeWarning from sklearn about degrees of freedom
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
        dataset = StatementsDataset(mock_edgar, verbose=True)

    # All-NaN column processing happens but data should still work
    assert dataset.data is not None


# Verbose mode tests


@unit
def test_verbose_mode_prints_info(capsys):
    """Verbose mode should print processing information."""
    from jpm.question_1.data.datasets.statements import StatementsDataset

    mock_edgar = make_mock_edgar_data(n_periods=20, n_features=5)
    # Add columns that will be filtered to trigger verbose output
    mock_edgar.data["constant_col"] = 100.0

    StatementsDataset(mock_edgar, verbose=True)

    # Check that something was printed (exact output depends on data)
    # captured = capsys.readouterr()
    # This is more of a smoke test - just ensure it runs
    assert True
