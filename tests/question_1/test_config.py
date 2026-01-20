from pathlib import Path
from types import SimpleNamespace

import pytest

from jpm.config.question_1 import Config, DataConfig, LSTMConfig

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

    cfg = LSTMConfig.from_args(args)

    assert cfg.lstm_units == 256
    assert cfg.dense_units == 64
    assert cfg.lstm_layers == LSTMConfig().lstm_layers
    assert cfg.dropout == LSTMConfig().dropout


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

    cfg = LSTMConfig.from_args(args)

    assert cfg.lr == pytest.approx(5e-4)
    assert cfg.epochs == 20
    assert cfg.checkpoint_path == Path("custom_ckpts")
    assert cfg.scheduler == "cosine"
    assert cfg.decay_steps == LSTMConfig().decay_steps
    assert cfg.decay_rate == pytest.approx(0.8)


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

    cfg = LSTMConfig.from_args(args)

    assert cfg.enforce_balance is True
    assert cfg.learn_identity is True
    assert cfg.identity_weight == pytest.approx(0.2)
    assert cfg.learn_subtotals == LSTMConfig().learn_subtotals
    assert cfg.subcategory_weight == pytest.approx(5e-5)


@integration
def test_config_composes_custom_subconfigs():
    """Config should compose sub-configs created via from_args helpers."""
    data_args = SimpleNamespace(ticker="AAPL", periods=8)
    lstm_args = SimpleNamespace(lstm_units=64, dense_units=32)

    cfg = Config(
        data=DataConfig.from_args(data_args),
        lstm=LSTMConfig.from_args(lstm_args),
    )

    assert cfg.data.ticker == "AAPL"
    assert cfg.lstm.lstm_units == 64
    assert cfg.lstm.lr == pytest.approx(1e-4)
    assert cfg.lstm.epochs == 500
    assert cfg.lstm.enforce_balance is False
    assert cfg.lstm.identity_weight == pytest.approx(1e-4)
