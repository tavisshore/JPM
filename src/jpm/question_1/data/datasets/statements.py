import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from jpm.question_1.data.ed import EdgarData
from jpm.question_1.data.structures import get_fs_struct
from jpm.question_1.data.utils import build_windows, remap_financial_dataframe
from jpm.question_1.misc import get_leaf_keys


def prune_features_for_lstm(
    df: pd.DataFrame,
    variance_threshold: float = 1e-5,
    corr_threshold: float = 0.95,
    cosine_threshold: float = 0.995,
    keep_columns: list | None = None,
):
    """
    Prune features for LSTM training.

    Steps:
    1. Remove non-numeric columns
    2. Remove low-variance columns
    3. Remove highly correlated columns
    4. Remove near-duplicate columns via cosine similarity

    Parameters
    ----------
    df : pd.DataFrame
        Time-ordered dataframe (rows = time)
    variance_threshold : float
        Minimum variance to keep a feature
    corr_threshold : float
        Pearson correlation threshold for redundancy
    cosine_threshold : float
        Cosine similarity threshold for near-duplicates
    keep_columns : list, optional
        Columns that must never be removed

    Returns
    -------
    pruned_df : pd.DataFrame
    removed : dict
        Removed columns by reason
    """

    keep_columns = set(keep_columns or [])

    # Numeric only
    numeric_df = df.select_dtypes(include=[np.number])

    # Low variance filter
    vt = VarianceThreshold(threshold=variance_threshold)
    vt.fit(numeric_df)

    kept_mask = vt.get_support()
    low_var_cols = numeric_df.columns[~kept_mask]

    numeric_df = numeric_df.drop(columns=low_var_cols)

    # Correlation pruning
    corr = numeric_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        if col in keep_columns:
            continue
        if any(upper[col] > corr_threshold):
            to_drop.add(col)

    numeric_df = numeric_df.drop(columns=to_drop)

    # Cosine similarity pruning
    scaler = StandardScaler()
    X = scaler.fit_transform(numeric_df.values.T)

    sim = cosine_similarity(X)

    to_drop = set()
    cols = numeric_df.columns

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if sim[i, j] > cosine_threshold:
                if cols[j] not in keep_columns:
                    to_drop.add(cols[j])

    numeric_df = numeric_df.drop(columns=to_drop)

    return numeric_df


class StatementsDataset:
    """Prepares train/val datasets from EdgarData for model training."""

    def __init__(self, edgar_data: "EdgarData", verbose: bool = False) -> None:
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
        self.verbose = verbose

        # Copy necessary attributes from EdgarData
        self.data = edgar_data.data

        # NOTE Starting from now, go back quarterly - remove data after a break
        # print(f"Searching for break in data for {self.config.data.ticker}...")
        # self.data = self.data.sort_index(ascending=False)
        # quarterly_index = self.data.index.asfreq("Q")
        # expected = pd.period_range(
        #     end=quarterly_index[0], periods=len(self.data), freq="Q"
        # )[::-1]
        # mask = (self.data.index == expected).cumprod().astype(bool)
        # self.data = self.data[mask]

        self._prepare_dataset()

    def _prepare_dataset(self) -> None:
        """Prepare scaled features and create train/val datasets."""

        # Remove columns that are derivations / sums of other columns
        derived_cols = []
        for k in ["balance_sheet", "income_statement", "cash_flow", "equity"]:
            derived_cols.extend(get_fs_struct(k)["drop_summations"])
        self.data = self.data.drop(columns=derived_cols, errors="ignore")

        # Removing non-BS columns that correlate, little info etc.
        self._get_structure()
        bs_cols = [item for lst in self.bs_structure.values() for item in lst] + [
            "Net Income"
        ]
        self.data = prune_features_for_lstm(self.data, keep_columns=bs_cols)
        self.tgt_indices = list(range(len(self.data.columns)))

        self._prepare_targets()

        self._set_feature_index()

        X_scaled, scaler = self._scale_features()
        self._set_scaler_stats(scaler)

        X_train, y_train, X_test, y_test = build_windows(
            config=self.config,
            X=X_scaled,
            tgt_indices=self.tgt_indices,
            index=self.data.index,
        )

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

        self.val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(
            self.config.data.withhold_periods
        )

    def _get_structure(self) -> None:
        fs_structure = get_fs_struct("all")
        self.balance_sheet_structure = fs_structure["balance_sheet"]
        self.income_statement_structure = fs_structure["income_statement"]

        bs_keys = get_leaf_keys(self.balance_sheet_structure["prediction_structure"])
        bs_final_keys = [
            k
            for k in bs_keys
            if k not in self.balance_sheet_structure["drop_summations"]
        ]
        self.bs_keys = bs_final_keys

        # Store FS structures with leaf keys and no __unmapped__
        self.bs_structure = {
            "Assets": [
                k
                for k in remap_financial_dataframe(
                    self.data,
                    self.balance_sheet_structure["prediction_structure"]["Assets"],
                ).columns.tolist()
                if k not in self.balance_sheet_structure["drop_summations"]
            ],
            "Liabilities": [
                k
                for k in remap_financial_dataframe(
                    self.data,
                    self.balance_sheet_structure["prediction_structure"]["Liabilities"],
                ).columns.tolist()
                if k not in self.balance_sheet_structure["drop_summations"]
            ],
            "Equity": [
                k
                for k in remap_financial_dataframe(
                    self.data,
                    self.balance_sheet_structure["prediction_structure"]["Equity"],
                ).columns.tolist()
                if k not in self.balance_sheet_structure["drop_summations"]
            ],
        }

        self.is_structure = {
            "Revenues": [
                k
                for k in remap_financial_dataframe(
                    self.data,
                    self.income_statement_structure["prediction_structure"]["Revenues"],
                ).columns.tolist()
                if k not in self.income_statement_structure["drop_summations"]
            ],
            "Expenses": [
                k
                for k in remap_financial_dataframe(
                    self.data,
                    self.income_statement_structure["prediction_structure"][
                        "Cost of Revenue"
                    ],
                ).columns.tolist()
                if k not in self.income_statement_structure["drop_summations"]
            ],
        }

    def _prepare_targets(self) -> None:
        self.targets = list(self.data.columns)
        self.name_to_target_idx = {name: i for i, name in enumerate(self.targets)}

        asset_mappings = [
            self.name_to_target_idx[n]
            for n in get_leaf_keys(
                self.balance_sheet_structure["prediction_structure"]["Assets"]
            )
            if n in self.name_to_target_idx
        ]
        liability_mappings = [
            self.name_to_target_idx[n]
            for n in get_leaf_keys(
                self.balance_sheet_structure["prediction_structure"]["Liabilities"]
            )
            if n in self.name_to_target_idx
        ]
        equity_mappings = [
            self.name_to_target_idx[n]
            for n in get_leaf_keys(
                self.balance_sheet_structure["prediction_structure"]["Equity"]
            )
            if n in self.name_to_target_idx
        ]

        # For evaluation - identity etc.
        self.feature_mappings = {
            "assets": asset_mappings,
            "liabilities": liability_mappings,
            "equity": equity_mappings,
        }

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
