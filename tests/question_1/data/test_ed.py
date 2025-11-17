from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
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


def make_config(tmp_path):
    return Config(
        data=DataConfig(
            ticker="AAPL",
            cache_dir=str(tmp_path),
            periods=2,
            lookback=1,
            horizon=1,
            batch_size=2,
            target_type="full",
        ),
        model=ModelConfig(),
        training=TrainingConfig(),
        loss=LossConfig(),
    )


@unit
def test_map_features_creates_expected_mappings(tmp_path):
    """map_features should translate structure names into target indices."""
    from jpm.question_1.data.ed import EdgarDataLoader

    config = make_config(tmp_path)
    loader = object.__new__(EdgarDataLoader)
    loader.config = config
    loader.bs_structure = {
        "assets": {
            "current_assets": ["cash", "inventory"],
            "non_current_assets": ["ppe"],
        },
        "liabilities": {
            "current_liabilities": ["payables"],
            "non_current_liabilities": ["lt_debt"],
        },
        "equity": ["equity"],
    }
    loader.targets = ["cash", "inventory", "ppe", "payables", "lt_debt", "equity"]
    loader.map_features()

    assert loader.feature_mappings["assets"] == [0, 1, 2]
    assert loader.feature_mappings["liabilities"] == [3, 4]
    assert loader.feature_mappings["equity"] == [5]


@unit
def test_process_statement_normalizes_and_filters(monkeypatch, tmp_path):
    """_process_statement should lower-case concepts and filter to needed columns."""
    from jpm.question_1.data.ed import EdgarDataLoader

    config = make_config(tmp_path)
    loader = object.__new__(EdgarDataLoader)
    loader.config = config

    df = pd.DataFrame(
        {
            "concept": ["us-gaap_Cash", "us-gaap_Cash"],
            "2022-12-31": [100, 200],
            "2023-12-31": [150, 250],
        }
    )
    stmt = MagicMock()
    stmt.to_dataframe.return_value = df

    processed = loader._process_statement(
        stmt=stmt, kind="balance sheet", needed_cols=["cash"]
    )
    assert list(processed.columns) == ["cash"]
    # Duplicate concept collapsed; latest value retained
    assert processed.iloc[0, 0] == 100


@unit
def test_process_statement_raises_when_missing_statement(tmp_path):
    """_process_statement should raise ValueError when stmt is None."""
    from jpm.question_1.data.ed import EdgarDataLoader

    config = make_config(tmp_path)
    loader = object.__new__(EdgarDataLoader)
    loader.config = config

    with pytest.raises(ValueError):
        loader._process_statement(stmt=None, kind="balance sheet", needed_cols=["cash"])


@unit
def test_map_features_ignores_missing_targets(tmp_path):
    """map_features should skip structure entries that are absent from targets."""
    from jpm.question_1.data.ed import EdgarDataLoader

    config = make_config(tmp_path)
    loader = object.__new__(EdgarDataLoader)
    loader.config = config
    loader.bs_structure = {
        "assets": {
            "current_assets": ["cash", "goodwill_not_tracked"],
            "non_current_assets": [],
        },
        "liabilities": {"current_liabilities": [], "non_current_liabilities": []},
        "equity": [],
    }
    loader.targets = ["cash"]
    loader.map_features()

    assert loader.feature_mappings["current_assets"] == [0]


@integration
def test_create_dataset_uses_cache_when_available(monkeypatch, tmp_path):
    """EdgarDataLoader should load cached parquet instead of hitting EDGAR
    when available."""
    from jpm.question_1.data.ed import EdgarDataLoader

    config = make_config(tmp_path)
    fake_df = pd.DataFrame(
        {
            "cash": [100.0, 110.0, 120.0],
            "inventory": [50.0, 55.0, 60.0],
        }
    )
    cache_path = Path(config.data.cache_dir) / f"{config.data.ticker}.parquet"
    fake_df.to_parquet(cache_path)

    with (
        patch("jpm.question_1.data.ed.bs_identity") as bs_identity_mock,
        patch("jpm.question_1.data.ed.get_targets", return_value=list(fake_df.columns)),
    ):
        loader = EdgarDataLoader(config=config)

    assert bs_identity_mock.called
    assert loader.train_dataset is not None
    assert loader.val_dataset is not None
