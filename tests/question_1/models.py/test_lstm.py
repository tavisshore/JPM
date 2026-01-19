from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tensorflow as tf

from jpm.question_1.config import Config, DataConfig, LSTMConfig
from jpm.question_1.models.lstm import LSTMForecaster
from jpm.question_1.models.metrics import Metric
from tests.question_1.conftest import DummyDataLoader, DummyStatementsDataset

unit = pytest.mark.unit
integration = pytest.mark.integration


def _build_config(tmp_path: Path, scheduler: str = "constant"):
    return Config(
        data=DataConfig(lookback=1, horizon=1, batch_size=2, periods=1),
        model=LSTMConfig(
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
def test_build_optimizer_constant_scheduler_returns_scalar_lr(tmp_path):
    """_build_optimizer should return a scalar LR when scheduler=constant."""
    config = _build_config(tmp_path, scheduler="constant")
    data = DummyDataLoader()
    dataset = DummyStatementsDataset()
    forecaster = LSTMForecaster(config=config, data=data, dataset=dataset)

    optimizer = forecaster._build_optimizer()
    lr = optimizer.learning_rate
    lr_value = float(lr.numpy()) if hasattr(lr, "numpy") else float(lr)
    assert lr_value == pytest.approx(config.training.lr)


@unit
def test_build_model_wraps_with_enforce_balance(tmp_path):
    """_build_model must wrap outputs with EnforceBalance when enabled."""
    config = _build_config(tmp_path)
    config.loss.enforce_balance = True
    data = DummyDataLoader()
    dataset = DummyStatementsDataset()
    with patch("jpm.question_1.models.lstm.EnforceBalance") as eb_mock:
        eb_mock.return_value = lambda tensor: tensor
        LSTMForecaster(config=config, data=data, dataset=dataset)

    eb_mock.assert_called_once()
    _, kwargs = eb_mock.call_args
    assert kwargs["feature_mappings"] == data.feature_mappings
    assert np.array_equal(kwargs["feature_means"], data.target_mean)
    assert kwargs["feature_names"] == data.bs_keys


@unit
def test_build_model_without_enforce_balance_skips_wrapper(tmp_path):
    """_build_model should skip EnforceBalance when enforce_balance=False."""
    config = _build_config(tmp_path)
    config.loss.enforce_balance = False
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
        patch.object(LSTMForecaster, "_build_optimizer", return_value="OPT"),
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
    dataset = DummyStatementsDataset()
    forecaster = LSTMForecaster(config=config, data=data, dataset=dataset)

    results = forecaster.evaluate(stage="val")
    assert isinstance(results.assets, Metric)
    assert isinstance(results.liabilities, Metric)
    assert set(results.features.keys()) == set(data.feat_to_idx.keys())


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
    assert fit_mock.call_args.kwargs["validation_data"] is data.val_dataset


@unit
def test_fit_without_validation_uses_loss_monitor(tmp_path, monkeypatch):
    """fit() should monitor training loss when validation data is absent."""
    config = _build_config(tmp_path)
    data = DummyDataLoader(with_val=False)
    dataset = DummyStatementsDataset()
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
    """predict() should delegate directly to the underlying Keras model."""
    config = _build_config(tmp_path)
    data = DummyDataLoader()
    dataset = DummyStatementsDataset()
    forecaster = LSTMForecaster(config=config, data=data, dataset=dataset)
    predict_mock = MagicMock(return_value="preds")
    forecaster.model.predict = predict_mock

    assert forecaster.predict("input") == "preds"
    predict_mock.assert_called_once_with("input")


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
        features={"feature0": metric, "feature1": metric},
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
        data.bs_structure["assets"], forecaster.val_results.features
    )
    build_section_mock.assert_any_call(
        data.bs_structure["liabilities"], forecaster.val_results.features
    )
    build_equity_mock.assert_called_once_with(
        data.bs_structure["equity"], forecaster.val_results.features
    )
    assert print_table_mock.call_count == 4
