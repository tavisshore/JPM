from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

REPORTS = {
    "alibaba": {
        "path": "assets/question_1/alibaba_report.pdf",
        "pages": [(254, 264), (269, 270), (307, 317), (324, 328)],
    },
    "exxon": {
        "path": "assets/question_1/exxon_report.pdf",
        "pages": [(88, 92), (101, 101), (103, 105), (107, 110)],
    },
    "gm": {
        "path": "assets/question_1/gm_report.pdf",
        "pages": [(61, 64), (72, 88), (95, 101)],
    },
    "goog": {
        "path": "assets/question_1/goog_report.pdf",
        "pages": [(49, 49), (57, 61), (70, 74), (76, 79), (81, 84)],
    },
    "jpm": {
        "path": "assets/question_1/jpm_report.pdf",
        "pages": [(174, 178), (188, 189), (195, 197)],
    },
    "lvmh": {
        "path": "assets/question_1/lvmh_report.pdf",
        "pages": [(24, 28), (39, 57)],
    },
    "msft": {
        "path": "assets/question_1/msft_report.pdf",
        "pages": [(44, 48), (57, 60)],
    },
    "tencent": {
        "path": "assets/question_1/tencent_report.pdf",
        "pages": [(124, 134), (186, 186)],
    },
    "vw": {"path": "assets/question_1/vw_report.pdf", "pages": [(472, 478)]},
}


@dataclass
class DataConfig:
    ticker: str = "AAPL"
    cache_dir: Path = Path("/scratch/datasets/jpm")
    save_dir: Path = Path("/scratch/projects/JPM/temp")
    plots_dir: Path = Path("/scratch/projects/JPM/temp/plots")
    periods: int = 60  # quarters -> 15 years, post-2008
    lookback: int = 4
    horizon: int = 1
    batch_size: int = 32
    target_type: str = "full"
    withhold_periods: int = 1  # test set size in quarters
    # >1.0 weighs for the seasonal lag timestep
    seasonal_weight: float = 1.1
    seasonal_lag: int = 4

    def __post_init__(self) -> None:
        self._validate_strings()
        self._validate_positive_ints()
        self._validate_positive_floats()
        self._validate_choices()
        self._create_directories()

    def _validate_strings(self) -> None:
        if not isinstance(self.ticker, str) or not self.ticker.strip():
            raise ValueError("ticker must be a non-empty string")

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

    def _create_directories(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def get_report_path(self) -> str | None:
        """Get the report path for the current ticker."""
        key = self.ticker.lower()
        if key in REPORTS:
            return REPORTS[key]["path"]
        return None

    def get_report_pages(self) -> list[tuple[int, int]] | None:
        """Get the report pages for the current ticker."""
        key = self.ticker.lower()
        if key in REPORTS:
            return REPORTS[key]["pages"]
        return None

    @classmethod
    def from_args(cls, args):
        kwargs = {}
        path_fields = {"cache_dir", "save_dir", "plots_dir"}
        for f in fields(cls):
            arg_val = getattr(args, f.name, None)
            if f.name == "seasonal_lag":
                kwargs[f.name] = f.default  # keep seasonal lag fixed
                continue
            val = f.default if arg_val is None else arg_val
            if f.name in path_fields and isinstance(val, str):
                val = Path(val)
            kwargs[f.name] = val
        return cls(**kwargs)


@dataclass
class LSTMConfig:
    # Model architecture
    lstm_units: int = 256
    lstm_layers: int = 2
    dense_units: int = 256
    dropout: float = 0.1
    variational: bool = False
    probabilistic: bool = False
    mc_samples: int = 1
    # Training hyperparameters
    lr: float = 1e-4
    decay_steps: int = 100
    decay_rate: float = 0.9
    scheduler: str = "exponential"
    epochs: int = 500
    checkpoint_path: Path = Path("ckpts")
    # Loss configuration
    enforce_balance: bool = False
    learn_identity: bool = False
    identity_weight: float = 1e-4
    learn_subtotals: bool = False
    subcategory_weight: float = 1e-5

    def __post_init__(self) -> None:  # noqa: C901
        # Model validation
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
        # Training validation
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
        # Loss validation
        if self.identity_weight < 0:
            raise ValueError("identity_weight must be non-negative")
        if self.subcategory_weight < 0:
            raise ValueError("subcategory_weight must be non-negative")

    @classmethod
    def from_args(cls, args):
        kwargs = {}
        for f in fields(cls):
            arg_val = getattr(args, f.name, None)
            kwargs[f.name] = f.default if arg_val is None else arg_val
        return cls(**kwargs)


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-5-mini"  # dev nano, eval mini, gpt-4o-2024-08-06
    temperature: float = 0.0  # 0.05
    max_tokens: int = 8192
    use_llm: bool = False
    adjust: bool = True

    @classmethod
    def from_args(cls, args):
        """Create a LLMConfig from argparse.Namespace."""
        kwargs = {}
        for f in fields(cls):
            arg_val = getattr(args, f.name, None)
            kwargs[f.name] = f.default if arg_val is None else arg_val
        return cls(**kwargs)


@dataclass
class XGBConfig:
    max_depth: int = 4
    learning_rate: float = 0.05
    n_estimators: int = 300
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    use_gpu: bool = True
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if not 0 < self.subsample <= 1:
            raise ValueError("subsample must be in (0, 1]")
        if not 0 < self.colsample_bytree <= 1:
            raise ValueError("colsample_bytree must be in (0, 1]")
        if self.reg_alpha < 0:
            raise ValueError("reg_alpha must be non-negative")
        if self.reg_lambda < 0:
            raise ValueError("reg_lambda must be non-negative")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_args(cls, args):
        """Create an XGBConfig from argparse.Namespace."""
        kwargs = {}
        for f in fields(cls):
            arg_val = getattr(args, f.name, None)
            kwargs[f.name] = f.default if arg_val is None else arg_val
        return cls(**kwargs)


@dataclass
class Config:
    """Root configuration grouping data, LSTM, LLM, and XGBoost settings."""

    data: DataConfig = field(default_factory=DataConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    xgb: XGBConfig = field(default_factory=XGBConfig)
