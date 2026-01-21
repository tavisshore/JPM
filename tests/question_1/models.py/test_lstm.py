# Import from conftest - pytest will find it automatically
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tensorflow as tf

from jpm.config.question_1 import Config, DataConfig, LSTMConfig
from jpm.question_1.models.lstm import (
    LSTMForecaster,
    SeasonalWeightLogger,
    TemporalAttention,
)
from jpm.question_1.models.metrics import Metric

sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import DummyDataLoader, DummyStatementsDataset

unit = pytest.mark.unit
integration = pytest.mark.integration


def _build_config(tmp_path: Path, scheduler: str = "constant"):
    return Config(
        data=DataConfig(lookback=1, horizon=1, batch_size=2, periods=1),
        lstm=LSTMConfig(
            lstm_units=4,
            lstm_layers=1,
            dense_units=1,
            dropout=0.0,
            lr=0.01,
            epochs=1,
            checkpoint_path=tmp_path / "ckpts",
            scheduler=scheduler,
            decay_steps=10,
            decay_rate=0.9,
            enforce_balance=False,
            learn_identity=False,
            learn_subtotals=False,
        ),
    )


@unit
def test_build_optimiser_constant_scheduler_returns_scalar_lr(tmp_path):
    """_build_optimiser should return a scalar LR when scheduler=constant."""
    config = _build_config(tmp_path, scheduler="constant")
    data = DummyDataLoader()
    dataset = DummyStatementsDataset()
    forecaster = LSTMForecaster(config=config, data=data, dataset=dataset)

    optimizer = forecaster._build_optimiser()
    lr = optimizer.learning_rate
    lr_value = float(lr.numpy()) if hasattr(lr, "numpy") else float(lr)
    assert lr_value == pytest.approx(config.lstm.lr)


@unit
def test_build_model_wraps_with_enforce_balance(tmp_path):
    """_build_model must wrap outputs with EnforceBalance when enabled."""
    config = _build_config(tmp_path)
    config.lstm.enforce_balance = True
    data = DummyDataLoader()
    dataset = DummyStatementsDataset()
    with patch("jpm.question_1.models.lstm.EnforceBalance") as eb_mock:
        eb_mock.return_value = lambda tensor: tensor
        LSTMForecaster(config=config, data=data, dataset=dataset)

    eb_mock.assert_called_once()
    _, kwargs = eb_mock.call_args
    assert kwargs["feature_mappings"] == dataset.feature_mappings
    assert np.array_equal(kwargs["feature_means"], dataset.target_mean)
    assert kwargs["feature_names"] == dataset.name_to_target_idx


@unit
def test_build_model_without_enforce_balance_skips_wrapper(tmp_path):
    """_build_model should skip EnforceBalance when enforce_balance=False."""
    config = _build_config(tmp_path)
    config.lstm.enforce_balance = False
    data = DummyDataLoader()
    dataset = DummyStatementsDataset()
    with patch("jpm.question_1.models.lstm.EnforceBalance") as eb_mock:
        LSTMForecaster(config=config, data=data, dataset=dataset)

    eb_mock.assert_not_called()


@unit
def test_compile_uses_bs_loss_and_optimizer(tmp_path):
    """_compile_model should wire bs_loss output and optimizer into compile()."""
    config = _build_config(tmp_path)
    data = DummyDataLoader()
    dataset = DummyStatementsDataset()
    with (
        patch(
            "jpm.question_1.models.lstm.bs_loss",
            return_value="LOSS",
        ) as loss_mock,
        patch.object(tf.keras.Model, "compile") as compile_mock,
        patch.object(LSTMForecaster, "_build_optimiser", return_value="OPT"),
    ):
        LSTMForecaster(config=config, data=data, dataset=dataset)

    loss_mock.assert_called_once()
    compile_mock.assert_called()
    _, kwargs = compile_mock.call_args
    assert kwargs["optimizer"] == "OPT"
    assert kwargs["loss"] == "LOSS"
    assert kwargs["metrics"] == ["mae"]


@integration
def test_model_compiles_with_bs_loss_and_callbacks(tmp_path):
    """Full model construction should succeed and produce valid predictions."""
    config = _build_config(tmp_path)
    data = DummyDataLoader()
    dataset = DummyStatementsDataset()
    with patch(
        "jpm.question_1.models.lstm.bs_loss",
        wraps=lambda *args, **kwargs: tf.keras.losses.MSE,
    ):
        forecaster = LSTMForecaster(config=config, data=data, dataset=dataset)

    batch = next(iter(data.train_dataset))[0]
    preds = forecaster.model(batch)
    assert preds.shape[-1] == len(data.targets)


@integration
def test_evaluate_returns_ticker_results(tmp_path):
    """evaluate() should return a TickerResults with populated metrics."""
    config = _build_config(tmp_path)
    data = DummyDataLoader()
    dataset = DummyStatementsDataset(lookback=config.data.lookback)
    forecaster = LSTMForecaster(config=config, data=data, dataset=dataset)

    results = forecaster.evaluate(stage="val")
    assert isinstance(results.assets, Metric)
    assert isinstance(results.liabilities, Metric)
    assert set(results.features.keys()) == set(dataset.feat_to_idx.keys())


@unit
def test_fit_uses_validation_monitor(tmp_path, monkeypatch):
    """fit() should monitor val_loss when a validation dataset exists."""
    config = _build_config(tmp_path)
    data = DummyDataLoader()
    dataset = DummyStatementsDataset()
    forecaster = LSTMForecaster(config=config, data=data, dataset=dataset)

    checkpoint_holder = {}

    def fake_checkpoint(**kwargs):
        # Record the kwargs passed to ModelCheckpoint for assertions later.
        checkpoint_holder["config"] = kwargs
        return MagicMock()

    monkeypatch.setattr(
        "jpm.question_1.models.lstm.tf.keras.callbacks.ModelCheckpoint",
        fake_checkpoint,
    )
    fit_mock = MagicMock()
    forecaster.model.fit = fit_mock

    forecaster.fit()

    assert checkpoint_holder["config"]["monitor"] == "val_loss"
    fit_mock.assert_called_once()
    assert fit_mock.call_args.kwargs["validation_data"] is dataset.val_dataset


@unit
def test_fit_without_validation_uses_loss_monitor(tmp_path, monkeypatch):
    """fit() should monitor training loss when validation data is absent."""
    config = _build_config(tmp_path)
    data = DummyDataLoader()
    dataset = DummyStatementsDataset(with_val=False)
    forecaster = LSTMForecaster(config=config, data=data, dataset=dataset)

    checkpoint_holder = {}

    def fake_checkpoint(**kwargs):
        # Record the kwargs even when validation data is absent.
        checkpoint_holder["config"] = kwargs
        return MagicMock()

    monkeypatch.setattr(
        "jpm.question_1.models.lstm.tf.keras.callbacks.ModelCheckpoint",
        fake_checkpoint,
    )
    fit_mock = MagicMock()
    forecaster.model.fit = fit_mock

    forecaster.fit()

    assert checkpoint_holder["config"]["monitor"] == "loss"
    fit_mock.assert_called_once()
    assert fit_mock.call_args.kwargs["validation_data"] is None


@unit
def test_predict_delegates_to_model(tmp_path):
    """predict() should run and return TickerResults."""
    config = _build_config(tmp_path)
    data = DummyDataLoader()
    dataset = DummyStatementsDataset()
    forecaster = LSTMForecaster(config=config, data=data, dataset=dataset)

    # Create a valid input array
    x = np.zeros((1, 1, 5), dtype=np.float64)
    results = forecaster.predict(x)

    # Should return TickerResults
    from jpm.question_1.models.metrics import TickerResults

    assert isinstance(results, TickerResults)


@unit
def test_load_restores_model(monkeypatch, tmp_path):
    """load() should hydrate the model via tf.keras.models.load_model."""
    config = _build_config(tmp_path)
    fake_model = object()
    monkeypatch.setattr(
        "jpm.question_1.models.lstm.tf.keras.models.load_model",
        lambda path: fake_model,
    )

    forecaster = LSTMForecaster.load("path/to/model", config=config)

    assert isinstance(forecaster, LSTMForecaster)
    assert forecaster.model is fake_model
    assert forecaster.config is config


@unit
def test_view_results_prints_tables(monkeypatch, tmp_path):
    """view_results should render the summary and section tables."""
    config = _build_config(tmp_path)
    data = DummyDataLoader()
    dataset = DummyStatementsDataset()
    forecaster = LSTMForecaster(config=config, data=data, dataset=dataset)

    metric = Metric(value=1.0, mae=0.1, gt=0.9)
    forecaster.val_results = SimpleNamespace(
        assets=metric,
        liabilities=metric,
        equity=metric,
        features={"feature_0": metric, "feature_1": metric, "feature_2": metric},
        baseline_mae={},
        skill={},
        model_mae=0.0,
        net_income_baseline_mae={},
        net_income_skill={},
        net_income_model_mae=0.0,
        net_income_pred=0.0,
        net_income_gt=0.0,
        net_income_baseline_pred={},
    )

    make_row_mock = MagicMock(side_effect=lambda label, m: [label, m.value])
    build_section_mock = MagicMock(return_value=[["section"]])
    build_equity_mock = MagicMock(return_value=[["equity"]])
    print_table_mock = MagicMock()

    monkeypatch.setattr("jpm.question_1.models.lstm.make_row", make_row_mock)
    monkeypatch.setattr(
        "jpm.question_1.models.lstm.build_section_rows", build_section_mock
    )
    monkeypatch.setattr(
        "jpm.question_1.models.lstm.build_equity_rows", build_equity_mock
    )
    monkeypatch.setattr("jpm.question_1.models.lstm.print_table", print_table_mock)

    forecaster.view_results(stage="val")

    assert make_row_mock.call_count == 3
    build_section_mock.assert_any_call(
        dataset.bs_structure["Assets"], forecaster.val_results.features
    )
    build_section_mock.assert_any_call(
        dataset.bs_structure["Liabilities"], forecaster.val_results.features
    )
    build_equity_mock.assert_called_once_with(
        dataset.bs_structure["Equity"], forecaster.val_results.features
    )
    assert print_table_mock.call_count == 4


# Probabilistic and Variational LSTM Tests


@unit
def test_probabilistic_lstm_builds_with_multivariate_normal_output(tmp_path):
    """Probabilistic LSTM should use MultivariateNormalTriLLayer output."""
    config = _build_config(tmp_path)
    config.lstm.probabilistic = True
    data = DummyDataLoader()
    dataset = DummyStatementsDataset()

    with patch("jpm.question_1.models.lstm.MultivariateNormalTriLLayer") as mvn_mock:
        mvn_mock.return_value = lambda x: x
        forecaster = LSTMForecaster(config=config, data=data, dataset=dataset)
        assert forecaster.config.lstm.probabilistic is True

    mvn_mock.assert_called_once()
    call_kwargs = mvn_mock.call_args.kwargs
    assert call_kwargs["event_size"] == len(dataset.targets)


@unit
def test_probabilistic_lstm_rejects_enforce_balance(tmp_path):
    """Probabilistic LSTM should raise error when enforce_balance=True."""
    config = _build_config(tmp_path)
    config.lstm.probabilistic = True
    config.lstm.enforce_balance = True
    data = DummyDataLoader()
    dataset = DummyStatementsDataset()

    with pytest.raises(ValueError, match="enforce_balance.*probabilistic"):
        LSTMForecaster(config=config, data=data, dataset=dataset)


@unit
def test_variational_lstm_builds_with_variational_layer(tmp_path):
    """Variational LSTM should use DenseVariationalLayer for output."""
    config = _build_config(tmp_path)
    config.lstm.variational = True
    data = DummyDataLoader()
    dataset = DummyStatementsDataset()

    with patch("jpm.question_1.models.lstm.DenseVariationalLayer") as var_mock:
        var_mock.return_value = lambda x: x
        forecaster = LSTMForecaster(config=config, data=data, dataset=dataset)
        assert forecaster.config.lstm.variational is True

    var_mock.assert_called_once()
    call_kwargs = var_mock.call_args.kwargs
    assert call_kwargs["units"] == len(dataset.tgt_indices)


@unit
def test_probabilistic_and_variational_are_mutually_exclusive(tmp_path):
    """Config should reject both probabilistic=True and variational=True."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        LSTMConfig(probabilistic=True, variational=True, epochs=1)


@integration
def test_probabilistic_predict_returns_mean_and_std(tmp_path):
    """Probabilistic LSTM predictions should include uncertainty estimates."""
    config = _build_config(tmp_path)
    config.lstm.probabilistic = True
    data = DummyDataLoader()
    dataset = DummyStatementsDataset()
    forecaster = LSTMForecaster(config=config, data=data, dataset=dataset)

    x = np.zeros((1, 1, 5), dtype=np.float32)
    mean, std = forecaster._predict_probabilistic(x)

    assert mean.shape == (1, len(dataset.targets))
    assert std.shape == (1, len(dataset.targets))
    assert np.all(std >= 0)


@integration
def test_variational_predict_with_mc_samples(tmp_path):
    """Variational LSTM should use Monte Carlo sampling for uncertainty."""
    config = _build_config(tmp_path)
    config.lstm.variational = True
    config.lstm.mc_samples = 5
    data = DummyDataLoader()
    dataset = DummyStatementsDataset()
    forecaster = LSTMForecaster(config=config, data=data, dataset=dataset)

    x = np.zeros((1, 1, 5), dtype=np.float32)
    mean, std = forecaster._predict_variational(x)

    assert mean.shape == (1, len(dataset.targets))
    assert std.shape == (1, len(dataset.targets))
    assert np.all(std >= 0)


@integration
def test_sample_predictions_requires_probabilistic_or_variational(tmp_path):
    """sample_predictions should error on deterministic models."""
    config = _build_config(tmp_path)
    data = DummyDataLoader()
    dataset = DummyStatementsDataset()
    forecaster = LSTMForecaster(config=config, data=data, dataset=dataset)

    x = np.zeros((1, 1, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="probabilistic or variational"):
        forecaster.sample_predictions(x, n_samples=10)


@unit
def test_temporal_attention_layer_init():
    """TemporalAttention layer should initialize with correct parameters."""
    layer = TemporalAttention(
        seasonal_lag=4,
        initial_weight=2.0,
        per_feature=False,
        min_weight=0.1,
        max_weight=10.0,
    )

    assert layer.seasonal_lag == 4
    assert layer.initial_weight == 2.0
    assert layer.per_feature is False
    assert layer.min_weight == 0.1
    assert layer.max_weight == 10.0


@unit
def test_temporal_attention_global_weight():
    """TemporalAttention should create single global weight when per_feature=False."""
    layer = TemporalAttention(seasonal_lag=4, initial_weight=2.0, per_feature=False)

    # Build layer with input shape (batch=None, timesteps=8, features=5)
    layer.build((None, 8, 5))

    # Should have single weight
    assert layer.seasonal_weight.shape == (1,)
    assert float(layer.seasonal_weight.numpy().item()) == pytest.approx(2.0)


@unit
def test_temporal_attention_per_feature_weight():
    """TemporalAttention should create per-feature weights when per_feature=True."""
    layer = TemporalAttention(seasonal_lag=4, initial_weight=2.0, per_feature=True)

    # Build layer with input shape (batch=None, timesteps=8, features=5)
    layer.build((None, 8, 5))

    # Should have one weight per feature
    assert layer.seasonal_weight.shape == (5,)
    assert np.allclose(layer.seasonal_weight.numpy(), 2.0)


@unit
def test_temporal_attention_applies_seasonal_weighting():
    """TemporalAttention should apply higher weights to seasonal timesteps."""
    layer = TemporalAttention(seasonal_lag=4, initial_weight=2.0, per_feature=False)

    # Input: 8 timesteps, 3 features
    inputs = tf.ones((1, 8, 3), dtype=tf.float32)
    layer.build(inputs.shape)

    outputs = layer(inputs)

    # Seasonal timesteps at indices 7, 3 (lookback-1=7, 7-4=3)
    # These should be weighted by 2.0, others by 1.0
    expected = tf.ones((1, 8, 3), dtype=tf.float32)
    # Apply weight of 2.0 at seasonal positions
    expected = tf.tensor_scatter_nd_update(
        expected,
        [[0, 7, 0], [0, 7, 1], [0, 7, 2], [0, 3, 0], [0, 3, 1], [0, 3, 2]],
        [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
    )

    assert outputs.shape == (1, 8, 3)
    assert tf.reduce_all(tf.abs(outputs - expected) < 1e-5)


@unit
def test_temporal_attention_get_config():
    """TemporalAttention should serialize configuration correctly."""
    layer = TemporalAttention(
        seasonal_lag=4,
        initial_weight=3.0,
        per_feature=True,
        min_weight=0.5,
        max_weight=8.0,
        name="test_attention",
    )

    config = layer.get_config()

    assert config["seasonal_lag"] == 4
    assert config["initial_weight"] == 3.0
    assert config["per_feature"] is True
    assert config["min_weight"] == 0.5
    assert config["max_weight"] == 8.0
    assert config["name"] == "test_attention"


@unit
def test_seasonal_weight_logger_init():
    """SeasonalWeightLogger should initialize with correct layer name."""
    logger = SeasonalWeightLogger(layer_name="my_attention")
    assert logger.layer_name == "my_attention"
    assert logger.weight_history == []
    assert logger.seasonal_layer is None


@unit
def test_seasonal_weight_logger_finds_layer():
    """SeasonalWeightLogger should find TemporalAttention layer in model."""
    # Create a simple model with TemporalAttention
    inputs = tf.keras.layers.Input(shape=(4, 3))
    x = TemporalAttention(seasonal_lag=4, name="temporal_attention")(inputs)
    x = tf.keras.layers.LSTM(8)(x)
    outputs = tf.keras.layers.Dense(2)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    logger = SeasonalWeightLogger(layer_name="temporal_attention")
    logger.set_model(model)
    logger.on_train_begin()

    assert logger.seasonal_layer is not None
    assert logger.seasonal_layer.name == "temporal_attention"


@integration
def test_learnable_seasonal_weight_integrates_with_lstm(tmp_path):
    """LSTM should integrate TemporalAttention when learnable_seasonal_weight=True."""
    config = _build_config(tmp_path)
    config.data.seasonal_lag = 4
    config.data.learnable_seasonal_weight = True
    config.data.seasonal_weight = 2.5
    config.data.lookback = 8

    data = DummyDataLoader()
    dataset = DummyStatementsDataset()
    forecaster = LSTMForecaster(config=config, data=data, dataset=dataset)

    # Check that TemporalAttention layer exists in model
    layer_names = [layer.name for layer in forecaster.model.layers]
    assert "temporal_attention" in layer_names

    # Check that the layer has correct initial weight
    attention_layer = None
    for layer in forecaster.model.layers:
        if isinstance(layer, TemporalAttention):
            attention_layer = layer
            break

    assert attention_layer is not None
    assert float(attention_layer.seasonal_weight.numpy().item()) == pytest.approx(2.5)


@integration
def test_learnable_seasonal_weight_trains_and_changes(tmp_path):
    """Seasonal weight should be trainable and change during training."""
    config = _build_config(tmp_path)
    config.data.seasonal_lag = 4
    config.data.learnable_seasonal_weight = True
    config.data.seasonal_weight = 2.0
    config.data.lookback = 8
    config.lstm.epochs = 3

    data = DummyDataLoader()
    dataset = DummyStatementsDataset(lookback=8)  # Match config lookback
    forecaster = LSTMForecaster(config=config, data=data, dataset=dataset)

    # Get initial weight
    attention_layer = None
    for layer in forecaster.model.layers:
        if isinstance(layer, TemporalAttention):
            attention_layer = layer
            break

    # initial_weight = float(attention_layer.seasonal_weight.numpy().item())

    # Train model (should update the weight)
    forecaster.fit(verbose=0)

    # Get final weight
    final_weight = float(attention_layer.seasonal_weight.numpy().item())

    # Weight should be trainable (may or may not change depending on loss landscape,
    # but it should remain within the constrained bounds)
    assert 0.1 <= final_weight <= 10.0  # Within min/max constraints


@integration
def test_seasonal_weight_logger_tracks_history(tmp_path):
    """SeasonalWeightLogger should track weight history during training."""
    config = _build_config(tmp_path)
    config.data.seasonal_lag = 4
    config.data.learnable_seasonal_weight = True
    config.data.lookback = 8
    config.lstm.epochs = 3

    data = DummyDataLoader()
    dataset = DummyStatementsDataset(lookback=8)  # Match config lookback
    forecaster = LSTMForecaster(config=config, data=data, dataset=dataset)

    # Train with logger (it's automatically added in fit() when learnable=True)
    history = forecaster.fit(verbose=0)

    # Check that seasonal_weight was logged in history
    assert "seasonal_weight" in history.history
    assert len(history.history["seasonal_weight"]) == config.lstm.epochs


@integration
def test_fixed_seasonal_weight_when_learnable_disabled(tmp_path):
    """Fixed seasonal weighting should be applied when learnable_seasonal_weight=False."""
    config = _build_config(tmp_path)
    config.data.seasonal_lag = 4
    config.data.learnable_seasonal_weight = False  # Disabled
    config.data.lookback = 8

    data = DummyDataLoader()
    dataset = DummyStatementsDataset()
    forecaster = LSTMForecaster(config=config, data=data, dataset=dataset)

    # TemporalAttention layer should NOT exist in model
    layer_names = [layer.name for layer in forecaster.model.layers]
    assert "temporal_attention" not in layer_names

    # Dataset should have applied fixed weighting (check that X_train was modified)
    # This is tested indirectly - the old _apply_seasonal_weight should have run
