from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class DataConfig:
    """Data-related configuration."""

    ticker: str = "AAPL"
    cache_dir: str = "/scratch/datasets/jpm"
    periods: int = 40  # quarters -> 10 years
    lookback: int = 4
    horizon: int = 1
    batch_size: int = 32
    target_type: str = "full"

    @classmethod
    def from_args(cls, args):
        """Create a DataConfig from argparse.Namespace."""
        kwargs = {}
        for f in fields(cls):
            arg_val = getattr(args, f.name, None)
            kwargs[f.name] = f.default if arg_val is None else arg_val
        return cls(**kwargs)


@dataclass
class ModelConfig:
    """Model hyperparameters."""

    lstm_units: int = 128
    lstm_layers: int = 1
    dense_units: int = 128
    dropout: float = 0.1

    @classmethod
    def from_args(cls, args):
        """Create a ModelConfig from argparse.Namespace."""
        kwargs = {}
        for f in fields(cls):
            arg_val = getattr(args, f.name, None)
            kwargs[f.name] = f.default if arg_val is None else arg_val
        return cls(**kwargs)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    lr: float = 1e-3
    decay_steps: int = 100
    decay_rate: float = 0.9
    scheduler: str = "exponential"  # Options: "exponential", "cosine", or "constant"
    epochs: int = 1000
    checkpoint_path: Path = Path("ckpts")

    @classmethod
    def from_args(cls, args):
        """Create a TrainingConfig from argparse.Namespace."""
        kwargs = {}
        for f in fields(cls):
            arg_val = getattr(args, f.name, None)
            kwargs[f.name] = f.default if arg_val is None else arg_val
        return cls(**kwargs)


@dataclass
class LossConfig:
    """Loss term configuration."""

    enforce_balance: bool = True
    learn_identity: bool = True
    identity_weight: float = 1e-4
    learn_subtotals: bool = False
    subcategory_weight: float = 1e-5

    @classmethod
    def from_args(cls, args):
        """Create a LossConfig from argparse.Namespace."""
        kwargs = {}
        for f in fields(cls):
            arg_val = getattr(args, f.name, None)
            kwargs[f.name] = f.default if arg_val is None else arg_val
        return cls(**kwargs)


@dataclass
class Config:
    """Root configuration grouping data, model, training, and loss settings."""

    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    loss: LossConfig = LossConfig()
