from pathlib import Path
from types import SimpleNamespace

import pytest

from jpm.question_1.config import (
    Config,
    DataConfig,
    LossConfig,
    ModelConfig,
    TrainingConfig,
)

unit = pytest.mark.unit
integration = pytest.mark.integration


@unit
def test_data_config_from_args_overrides_and_defaults():
    """Provided args override defaults; None falls back to dataclass defaults."""
    args = SimpleNamespace(
        ticker="AAPL",
        cache_dir=None,
        periods=16,
        lookback=None,
        horizon=2,
        batch_size=None,
        target_type="full",
    )

    cfg = DataConfig.from_args(args)

    assert cfg.ticker == "AAPL"
    assert cfg.periods == 16
    assert cfg.horizon == 2
    # None should revert to the default values
    assert cfg.cache_dir == DataConfig().cache_dir
    assert cfg.lookback == DataConfig().lookback
    assert cfg.batch_size == DataConfig().batch_size


@unit
def test_model_config_from_args_handles_partial_args():
    """ModelConfig.from_args should mix overrides with defaults."""
    args = SimpleNamespace(
        lstm_units=256,
        lstm_layers=None,
        dense_units=64,
        dropout=None,
    )

    cfg = ModelConfig.from_args(args)

    assert cfg.lstm_units == 256
    assert cfg.dense_units == 64
    assert cfg.lstm_layers == ModelConfig().lstm_layers
    assert cfg.dropout == ModelConfig().dropout


@unit
def test_training_config_from_args_preserves_paths_and_scheduler():
    """TrainingConfig.from_args should keep custom paths/scheduler settings."""
    args = SimpleNamespace(
        lr=5e-4,
        epochs=20,
        checkpoint_path=Path("custom_ckpts"),
        scheduler="cosine",
        decay_steps=None,
        decay_rate=0.8,
    )

    cfg = TrainingConfig.from_args(args)

    assert cfg.lr == pytest.approx(5e-4)
    assert cfg.epochs == 20
    assert cfg.checkpoint_path == Path("custom_ckpts")
    assert cfg.scheduler == "cosine"
    assert cfg.decay_steps == TrainingConfig().decay_steps
    assert cfg.decay_rate == pytest.approx(0.8)


@unit
def test_training_config_from_args_handles_missing_attributes():
    """Args without every field should still fall back to defaults."""
    args = SimpleNamespace(lr=1e-3)  # epochs/decay_* missing

    cfg = TrainingConfig.from_args(args)

    assert cfg.lr == pytest.approx(1e-3)
    assert cfg.epochs == TrainingConfig().epochs
    assert cfg.decay_steps == TrainingConfig().decay_steps
    assert cfg.decay_rate == TrainingConfig().decay_rate


@unit
def test_loss_config_from_args_overrides_selected_fields():
    """LossConfig.from_args should override provided weights/flags."""
    args = SimpleNamespace(
        enforce_balance=True,
        learn_identity=True,
        identity_weight=0.2,
        learn_subtotals=None,
        subcategory_weight=5e-5,
    )

    cfg = LossConfig.from_args(args)

    assert cfg.enforce_balance is True
    assert cfg.learn_identity is True
    assert cfg.identity_weight == pytest.approx(0.2)
    assert cfg.learn_subtotals == LossConfig().learn_subtotals
    assert cfg.subcategory_weight == pytest.approx(5e-5)


@unit
def test_data_config_from_args_ignores_unrecognized_attributes():
    """DataConfig.from_args should ignore unknown attributes on the args object."""
    args = SimpleNamespace(ticker="META", extra_flag=True)

    cfg = DataConfig.from_args(args)

    assert cfg.ticker == "META"
    # Ensure extra attribute did not create new fields or errors
    assert hasattr(cfg, "periods")
    assert cfg.periods == DataConfig().periods


@integration
def test_config_composes_custom_subconfigs():
    """Config should compose sub-configs created via from_args helpers."""
    data_args = SimpleNamespace(ticker="AAPL", periods=8)
    model_args = SimpleNamespace(lstm_units=64, dense_units=32)
    training_args = SimpleNamespace(lr=1e-4, epochs=50)
    loss_args = SimpleNamespace(enforce_balance=True, identity_weight=0.5)

    cfg = Config(
        data=DataConfig.from_args(data_args),
        model=ModelConfig.from_args(model_args),
        training=TrainingConfig.from_args(training_args),
        loss=LossConfig.from_args(loss_args),
    )

    assert cfg.data.ticker == "AAPL"
    assert cfg.model.lstm_units == 64
    assert cfg.training.lr == pytest.approx(1e-4)
    assert cfg.training.epochs == 50
    assert cfg.loss.enforce_balance is True
    assert cfg.loss.identity_weight == pytest.approx(0.5)
