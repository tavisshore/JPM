from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class DataConfig:
    """Data-related configuration."""

    ticker: str = "AAPL"
    cache_dir: str = "/scratch/datasets/jpm"
    periods: int = 60  # quarters -> 15 years, post-2008
    lookback: int = 5
    horizon: int = 1
    batch_size: int = 32
    target_type: str = "full"
    withhold_periods: int = 2  # test set size in quarters
    # >1.0 weighs for the seasonal lag timestep
    seasonal_weight: float = 1.15  # 11
    seasonal_lag: int = 4  # don't change

    def __post_init__(self) -> None:
        self._validate_strings()
        self._validate_positive_ints()
        self._validate_positive_floats()
        self._validate_choices()

    def _validate_strings(self) -> None:
        if not isinstance(self.ticker, str) or not self.ticker.strip():
            raise ValueError("ticker must be a non-empty string")
        if self.cache_dir is None or not str(self.cache_dir).strip():
            raise ValueError("cache_dir must be provided")

    def _validate_positive_ints(self) -> None:
        for name, val in (
            ("periods", self.periods),
            ("lookback", self.lookback),
            ("horizon", self.horizon),
            ("batch_size", self.batch_size),
            ("seasonal_lag", self.seasonal_lag),
        ):
            if val <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.withhold_periods < 0:
            raise ValueError("withhold_periods must be non-negative")

    def _validate_positive_floats(self) -> None:
        if self.seasonal_weight <= 0:
            raise ValueError("seasonal_weight must be positive")

    def _validate_choices(self) -> None:
        if self.target_type not in {"full", "bs", "net_income"}:
            raise ValueError("target_type must be one of {'full', 'bs', 'net_income'}")

    @classmethod
    def from_args(cls, args):
        """Create a DataConfig from argparse.Namespace."""
        kwargs = {}
        for f in fields(cls):
            arg_val = getattr(args, f.name, None)
            if f.name == "seasonal_lag":
                kwargs[f.name] = f.default  # keep seasonal lag fixed
                continue
            kwargs[f.name] = f.default if arg_val is None else arg_val
        # Populate fields from CLI, otherwise use defaults
        return cls(**kwargs)


@dataclass
class ModelConfig:
    """Model hyperparameters."""

    lstm_units: int = 256  # 368
    lstm_layers: int = 2
    dense_units: int = 256  # 256
    dropout: float = 0.2
    variational: bool = False
    probabilistic: bool = False
    mc_samples: int = 1

    def __post_init__(self) -> None:
        if self.lstm_units <= 0:
            raise ValueError("lstm_units must be positive")
        if self.lstm_layers <= 0:
            raise ValueError("lstm_layers must be positive")
        if self.dense_units < 0:
            raise ValueError("dense_units must be >= 0")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.mc_samples <= 0:
            raise ValueError("mc_samples must be positive")

    @classmethod
    def from_args(cls, args):
        """Create a ModelConfig from argparse.Namespace."""
        kwargs = {}
        for f in fields(cls):
            arg_val = getattr(args, f.name, None)
            kwargs[f.name] = f.default if arg_val is None else arg_val
        # Avoid leaving unspecified fields unset when mixing CLI and defaults
        return cls(**kwargs)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    lr: float = 1e-4
    decay_steps: int = 100
    decay_rate: float = 0.9
    scheduler: str = "exponential"
    epochs: int = 500  # Raise again after dev
    checkpoint_path: Path = Path("ckpts")

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.decay_steps <= 0:
            raise ValueError("decay_steps must be positive")
        if self.decay_rate <= 0:
            raise ValueError("decay_rate must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.scheduler not in {"exponential", "cosine", "constant"}:
            raise ValueError("scheduler must be 'exponential', 'cosine', or 'constant'")

    @classmethod
    def from_args(cls, args):
        """Create a TrainingConfig from argparse.Namespace."""
        kwargs = {}
        for f in fields(cls):
            arg_val = getattr(args, f.name, None)
            kwargs[f.name] = f.default if arg_val is None else arg_val
        # Keeps learning-rate scheduling optional for quick experiments
        return cls(**kwargs)


@dataclass
class LossConfig:
    """Loss term configuration."""

    enforce_balance: bool = True
    learn_identity: bool = True
    identity_weight: float = 1e-2
    learn_subtotals: bool = False
    subcategory_weight: float = 1e-5

    def __post_init__(self) -> None:
        if self.identity_weight < 0:
            raise ValueError("identity_weight must be non-negative")
        if self.subcategory_weight < 0:
            raise ValueError("subcategory_weight must be non-negative")

    @classmethod
    def from_args(cls, args):
        """Create a LossConfig from argparse.Namespace."""
        kwargs = {}
        for f in fields(cls):
            arg_val = getattr(args, f.name, None)
            kwargs[f.name] = f.default if arg_val is None else arg_val
        # Enables toggling loss penalties via CLI switches
        return cls(**kwargs)


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-5-mini"  # dev nano, eval mini
    temperature: float = 0.05
    max_tokens: int = 8192
    adjust: bool = False

    @classmethod
    def from_args(cls, args):
        """Create a LLMConfig from argparse.Namespace."""
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
    llm: LLMConfig = LLMConfig()
