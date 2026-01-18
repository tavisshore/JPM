import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from jpm.question_1.data.ed import EdgarData
from jpm.question_1.data.utils import build_windows


class StatementsDataset:
    """Prepares train/val datasets from EdgarData for model training."""

    def __init__(
        self, edgar_data: "EdgarData", target: str = "lstm", verbose: bool = False
    ) -> None:
        """
        Initialize dataset from EdgarData.

        Parameters:
        -----------
        edgar_data : EdgarData
            Instance of EdgarData with loaded and processed financial data
        target : str
            Target model type ("lstm" or "xgboost")
        verbose : bool
            Whether to print detailed information during processing
        """
        self.edgar_data = edgar_data
        self.config = edgar_data.config
        self.target = target
        self.verbose = verbose

        # Copy necessary attributes from EdgarData
        self.data = edgar_data.data
        self.tgt_indices = edgar_data.tgt_indices
        self.targets = edgar_data.targets

        # TODO Starting from now, go back quarterly - remove data after a break
        # print(f"Searching for break in data for {self.config.data.ticker}...")
        # self.data = self.data.sort_index(ascending=False)
        # quarterly_index = self.data.index.asfreq("Q")
        # expected = pd.period_range(
        #     end=quarterly_index[0], periods=len(self.data), freq="Q"
        # )[::-1]
        # mask = (self.data.index == expected).cumprod().astype(bool)
        # self.data = self.data[mask]

        # Some values aren't present until the new years Q1
        # Drop new years back to where Income Before Taxes is not 0

        # Prepare the dataset
        self._prepare_dataset()

    def _prepare_dataset(self) -> None:
        """Prepare scaled features and create train/val datasets."""

        # Remove any leaves from derived structure that are in mapped_df
        # mapped_df = mapped_df.drop(
        #     columns=get_fs_struct(kind)["drop_summations"], errors="ignore"
        # )
        # TODO this at dataset time - might use for ratios

        self._set_feature_index()

        X_scaled, scaler = self._scale_features()
        self._set_scaler_stats(scaler)

        if self.target == "lstm":
            # print(f"\nin: {self.data.index}\n")
            X_train, y_train, X_test, y_test = build_windows(
                config=self.config,
                X=X_scaled,
                tgt_indices=self.tgt_indices,
                index=self.data.index,
            )

            # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

            X_train, X_test = self._apply_seasonal_weight(X_train, X_test)
            self.X_train, self.y_train = X_train, y_train
            self.X_test, self.y_test = X_test, y_test

            if len(X_train) == 0:
                raise ValueError(
                    f"\nNo valid consecutive windows found for training. "
                    f"Data has {len(self.data)} periods with gaps. "
                    f"Try reducing lookback ({self.config.data.lookback}) or "
                    f"withhold ({self.config.data.withhold_periods}).\n"
                )

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

            self.val_dataset = tf.data.Dataset.from_tensor_slices(
                (X_test, y_test)
            ).batch(self.config.data.withhold_periods)
        elif self.target == "xgboost":
            # TODO: Implement xgboost dataset preparation
            pass

    def _filter_low_quality_columns(self) -> None:
        """Remove columns with low variance,
        high frequency of single value, or all NaN."""
        # Drop columns where most common value exceeds threshold
        threshold = 0.5
        cols_to_drop = []
        for col in self.data.columns:
            vc = self.data[col].value_counts(dropna=False)
            if len(vc) > 0 and (vc.iloc[0] / len(self.data)) > threshold:
                cols_to_drop.append(col)
        self.data = self.data.drop(columns=cols_to_drop)

        # Drop all-NaN columns
        all_nan_cols = self.data.columns[self.data.isna().all()].tolist()
        if all_nan_cols:
            if self.verbose:
                print(f"Dropping {len(all_nan_cols)} all-NaN columns: {all_nan_cols}")
            self.data = self.data.drop(columns=all_nan_cols)

        # Remove columns with very low variance (before fillna to avoid
        # treating NaN-heavy columns as constant after they become zeros)
        stds = self.data.apply(lambda x: np.nanstd(x.astype(float)))
        low_variance_cols = stds[stds < 1e-6].index.tolist()
        if low_variance_cols:
            if self.verbose:
                print(
                    f"Dropping {len(low_variance_cols)} low-variance columns: "
                    f"{low_variance_cols}"
                )
            self.data = self.data.drop(columns=low_variance_cols)

    def _set_feature_index(self) -> None:
        """Create feature name to index mapping."""
        self.feat_to_idx = {n: i for i, n in enumerate(self.data.columns.tolist())}

    def _scale_features(self) -> tuple[np.ndarray, StandardScaler]:
        """Scale features using StandardScaler."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.data.values.astype("float64"))
        return X_scaled, scaler

    def _set_scaler_stats(self, scaler: StandardScaler) -> None:
        """Store scaler statistics for later use."""
        self.full_mean = np.asarray(scaler.mean_, dtype="float64")
        self.full_std = np.asarray(scaler.scale_, dtype="float64")
        self.target_mean = self.full_mean[self.tgt_indices]
        self.target_std = self.full_std[self.tgt_indices]

    def _apply_seasonal_weight(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply seasonal weighting to training and test data."""
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
