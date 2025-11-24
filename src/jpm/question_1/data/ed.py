import os
from pathlib import Path

import edgar
import numpy as np
import pandas as pd
import tensorflow as tf
from edgar import Company
from edgar.xbrl import XBRLS
from sklearn.preprocessing import StandardScaler

from jpm.question_1.config import Config
from jpm.question_1.data.utils import (
    bs_identity,
    build_windows,
    get_bs_structure,
    get_cf_structure,
    get_is_structure,
    get_leaf_values,
    get_targets,
    xbrl_to_snake,
)

# SEC requires user identification via email
email = os.getenv("EDGAR_EMAIL")
if not email:
    raise ValueError(
        "EDGAR_EMAIL environment variable not set"
        "set with 'export EDGAR_EMAIL='your_email@jpm.com'"
    )

edgar.set_identity(email)

format_limit = {
    "AAPL": "2018-12",
}


class EdgarDataLoader:
    """Load, scale, and window Edgar filings into train/val datasets."""

    def __init__(
        self,
        config: Config,
    ) -> None:
        self.config = config
        self.cache_statement = Path(
            f"{self.config.data.cache_dir}/{self.config.data.ticker}.parquet"
        )
        self.cache_statement.parent.mkdir(parents=True, exist_ok=True)
        self.bs_structure = get_bs_structure(ticker=self.config.data.ticker)
        # Prefer cached parquet to avoid repeated SEC fetches
        self.create_dataset()

    def create_dataset(self) -> None:
        self._validate_target_type()
        self.bs_keys = get_leaf_values(get_bs_structure(ticker=self.config.data.ticker))

        self.data = self._load_or_fetch_data()
        self._validate_data()

        self._set_feature_index()
        tar = get_targets(
            mode=self.config.data.target_type, ticker=self.config.data.ticker
        )
        self._prepare_targets(tar)

        X_scaled, scaler = self._scale_features()
        self._set_scaler_stats(scaler)

        self.map_features()
        X_train, y_train, X_test, y_test = build_windows(
            X=X_scaled,
            lookback=self.config.data.lookback,
            horizon=self.config.data.horizon,
            tgt_indices=self.tgt_indices,
            withhold=self.config.data.withhold_periods,
        )

        X_train, X_test = self._apply_seasonal_weight(X_train, X_test)

        self.num_features = X_train.shape[-1]  # Input dim
        self.num_targets = len(self.tgt_indices)  # Output dim
        # Minimal tf.data pipeline with shuffle/prefetch to smooth training
        self.train_dataset = (
            tf.data.Dataset.from_tensor_slices(
                (X_train.astype("float64"), y_train.astype("float64"))
            )
            .shuffle(len(X_train))
            .batch(self.config.data.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        self.val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(
            self.config.data.batch_size
        )

    def _validate_target_type(self) -> None:
        if self.config.data.target_type not in {"full", "bs", "net_income"}:
            raise ValueError(
                f"Unsupported target_type '{self.config.data.target_type}'. "
                "Use 'full', 'bs', or 'net_income'."
            )

    def _load_or_fetch_data(self) -> pd.DataFrame:
        if self.cache_statement.exists():
            try:
                return pd.read_parquet(self.cache_statement)
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    f"Failed to load cached statement at {self.cache_statement}"
                ) from exc
        self.company = Company(self.config.data.ticker)
        self.create_statements()
        return self.data

    def _validate_data(self) -> None:
        if self.data.empty:
            raise ValueError("Loaded financial data is empty; cannot build dataset")

        non_numeric_cols = [
            col
            for col, dtype in self.data.dtypes.items()
            if not np.issubdtype(dtype, np.number)
        ]
        if non_numeric_cols:
            raise TypeError(
                f"All features must be numeric before scaling; "
                f"found non-numeric columns: {non_numeric_cols}"
            )
        bs_identity(self.data, ticker=self.config.data.ticker)

    def _set_feature_index(self) -> None:
        self.feat_to_idx = {n: i for i, n in enumerate(self.data.columns.tolist())}

    def _prepare_targets(self, tar: list[str]) -> None:
        if not tar:
            raise ValueError("No targets resolved for the provided configuration")

        if self.config.data.target_type != "full":
            self.targets = [t for t in tar if t in self.feat_to_idx]
            self.tgt_indices = [self.feat_to_idx[t] for t in self.targets]
        else:
            self.targets = list(self.data.columns)
            self.tgt_indices = list(range(len(self.targets)))

        if not self.targets:
            raise ValueError(
                "Resolved target set is empty; check ticker and target_type settings"
            )

    def _scale_features(self) -> tuple[np.ndarray, StandardScaler]:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.data.values.astype("float64"))
        return X_scaled, scaler

    def _set_scaler_stats(self, scaler: StandardScaler) -> None:
        self.full_mean = np.asarray(scaler.mean_, dtype="float64")
        self.full_std = np.asarray(scaler.scale_, dtype="float64")
        self.target_mean = self.full_mean[self.tgt_indices]
        self.target_std = self.full_std[self.tgt_indices]

    def _apply_seasonal_weight(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        seasonal_step = self.config.data.seasonal_lag
        if self.config.data.seasonal_weight == 1.0 or seasonal_step <= 0:
            return X_train, X_test

        seasonal_indices = []
        idx = self.config.data.lookback - seasonal_step
        while idx >= 0:
            seasonal_indices.append(idx)
            idx -= seasonal_step

        if not seasonal_indices:
            return X_train, X_test

        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train[:, seasonal_indices, :] *= self.config.data.seasonal_weight
        X_test[:, seasonal_indices, :] *= self.config.data.seasonal_weight
        return X_train, X_test

    def map_features(self) -> None:
        """
        With the bs, is, and cf structures known, map feature names to indices
        For loss, calculations etc.
        """

        # Map feature names to target indices for fast gathers
        name_to_target_idx = {name: i for i, name in enumerate(self.targets)}

        self.feature_mappings = {
            "assets": [
                name_to_target_idx[n]
                for group in ("current_assets", "non_current_assets")
                for n in self.bs_structure["assets"][group]
                if n in name_to_target_idx
            ],
            "liabilities": [
                name_to_target_idx[n]
                for group in ("current_liabilities", "non_current_liabilities")
                for n in self.bs_structure["liabilities"][group]
                if n in name_to_target_idx
            ],
            "equity": [
                name_to_target_idx[n]
                for n in self.bs_structure["equity"]
                if n in name_to_target_idx
            ],
            "current_assets": [
                name_to_target_idx[n]
                for n in self.bs_structure["assets"]["current_assets"]
                if n in name_to_target_idx
            ],
            "non_current_assets": [
                name_to_target_idx[n]
                for n in self.bs_structure["assets"]["non_current_assets"]
                if n in name_to_target_idx
            ],
            "current_liabilities": [
                name_to_target_idx[n]
                for n in self.bs_structure["liabilities"]["current_liabilities"]
                if n in name_to_target_idx
            ],
            "non_current_liabilities": [
                name_to_target_idx[n]
                for n in self.bs_structure["liabilities"]["non_current_liabilities"]
                if n in name_to_target_idx
            ],
        }

    def create_statements(self) -> None:
        self.filings = self.company.get_filings(form="10-Q")
        self.xbrls = XBRLS.from_filings(self.filings)

        # Process each statement
        self.bs_df = self._process_statement(
            stmt=self.xbrls.statements.balance_sheet(
                max_periods=self.config.data.periods
            ),
            kind="balance sheet",
            needed_cols=self.bs_keys,
        )
        self.is_df = self._process_statement(
            stmt=self.xbrls.statements.income_statement(
                max_periods=self.config.data.periods
            ),
            kind="income statement",
            needed_cols=get_leaf_values(
                get_is_structure(ticker=self.config.data.ticker)
            ),
        )
        self.cf_df = self._process_statement(
            stmt=self.xbrls.statements.cashflow_statement(
                max_periods=self.config.data.periods
            ),
            kind="cash flow statement",
            needed_cols=get_leaf_values(
                get_cf_structure(ticker=self.config.data.ticker)
            ),
        )

        # Align all statements on common dates (monthly periods)
        for attr in ("bs_df", "is_df", "cf_df"):
            df = getattr(self, attr)
            df.index = df.index.to_period("M")

        self.data = pd.concat(
            [self.bs_df, self.is_df, self.cf_df], axis=1, join="inner"
        )
        self.data.to_parquet(self.cache_statement)

    def _process_statement(
        self,
        stmt,
        kind: str,
        needed_cols: list[str],
    ) -> pd.DataFrame:
        """
        Common XBRL → tidy DataFrame pipeline.
        """
        if stmt is None:
            raise ValueError(f"No {kind} found for {self.config.data.ticker}")

        df = stmt.to_dataframe()

        # Split meta vs date cols
        meta_cols = [c for c in ("label", "concept") if c in df.columns]
        date_cols = [c for c in df.columns if c not in meta_cols]

        # concept x dates -> dates x concept
        wide = df.set_index("concept")[date_cols].T

        # Index normalisation
        wide.index = pd.to_datetime(wide.index)
        wide = wide.sort_index()
        wide.index.name = "period_end"

        # Drop records before format change
        if self.config.data.ticker in format_limit:
            wide = wide[wide.index >= format_limit[self.config.data.ticker]]

        # Normalise column names, collapse duplicates, clean NaNs
        wide.columns = [xbrl_to_snake(col) for col in wide.columns]
        collapsed = wide.T.groupby(level=0).first().T  # collapse duplicate concepts
        collapsed = collapsed.replace(r"^\s*$", np.nan, regex=True)
        collapsed = collapsed.dropna(axis=1, how="all")
        collapsed = collapsed.fillna(0)

        # Keep only required concepts; missing ones raise for visibility
        missing_cols = [col for col in needed_cols if col not in collapsed.columns]
        if missing_cols:
            raise ValueError(f"{kind.title()} missing required columns: {missing_cols}")
        return collapsed[needed_cols]


if __name__ == "__main__":
    config = Config()
    loader = EdgarDataLoader(config=config)
