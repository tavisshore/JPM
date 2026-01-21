#!/usr/bin/env python3
"""
CreditDataset class for credit rating prediction with TensorFlow/XGBoost.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


class CreditDataset:
    def __init__(
        self,
        data_dir: Path = Path("/scratch/datasets/jpm"),
        pattern: str = "[!_]*_ratings.parquet",
        val_size: float = 0.1,
        test_size: float = 0.1,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.data_dir = data_dir / "ratings"
        self.pattern = pattern
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose

        self.feature_cols: List[str] = []
        self.is_fitted = False

        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._X_val: Optional[np.ndarray] = None
        self._y_val: Optional[np.ndarray] = None
        self._X_test: Optional[np.ndarray] = None
        self._y_test: Optional[np.ndarray] = None
        self._X_predict: Optional[np.ndarray] = None

        self._meta_train: Optional[pd.DataFrame] = None
        self._meta_val: Optional[pd.DataFrame] = None
        self._meta_test: Optional[pd.DataFrame] = None
        self._meta_predict: Optional[pd.DataFrame] = None

    def _load_and_prepare(self, file_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load parquet and prepare for time series prediction."""
        df = pd.read_parquet(file_path)
        ticker = file_path.stem.split("_")[0]
        df["ticker"] = ticker

        # Reset index to make quarter a column for processing
        df = df.reset_index()

        # Ensure quarter column exists
        if "quarter" not in df.columns:
            raise ValueError(f"Expected 'quarter' column in {file_path.name}")

        # Remove defaulted/distressed ratings
        df = df[~df["rating"].str.contains("-PD", na=False)].copy()

        # Minimum data requirement
        if len(df) < 2:
            raise ValueError(f"Insufficient data: {len(df)} rows (need at least 2)")

        df = df.sort_values("quarter").reset_index(drop=True)
        df["target_rating"] = df["rating"].shift(-1)

        # Remove last row (no target) - safe now because len(df) >= 2
        df_trainable = df[:-1].copy()
        df_predict = df.iloc[[-1]].copy()
        return df_trainable, df_predict

    def _process_features(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """Process features and handle missing values."""
        feature_cols = [
            col
            for col in df.columns
            if col not in ["quarter", "rating", "target_rating", "ticker", "index"]
        ]
        print(feature_cols)
        # If no feature columns - data error, skip
        if len(feature_cols) == 0:
            return None

        # Replace inf/-inf with NaN
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

        # Fill NaN with per-ticker median
        df[feature_cols] = df.groupby("ticker")[feature_cols].transform(
            lambda x: x.fillna(x.median())
        )

        # Fill any remaining NaN with global median (for all-NaN ticker columns)
        global_medians = df[feature_cols].median()
        df[feature_cols] = df[feature_cols].fillna(global_medians)

        # Final fallback to 0 for any still-NaN columns (all values were NaN globally)
        df[feature_cols] = df[feature_cols].fillna(0)

        return df

    def load(self, feature_names: list | None = None) -> "CreditDataset":
        """Load and process all data files.

        Args:
            feature_names: Optional list of feature column names to use.
                          If provided, only these columns will be used as features.
                          If None, all available feature columns will be used.
        """
        # files = sorted(self.data_dir.glob(self.pattern))
        files = sorted(
            [
                f
                for f in Path(self.data_dir).glob("*_ratings.parquet")
                if not f.stem.endswith("partial_ratings")
            ]
        )

        trainable_dfs = []
        predict_dfs = []
        skipped = []
        for file in files:
            try:
                df_train, df_pred = self._load_and_prepare(file)

                # if df_train or df_pred contains no feature columns, skip
                # train_zeroed = (
                #     df_train[feature_names].replace(0, np.nan).isna().all(axis=0).any()
                # )
                # pred_zeroed = (
                #     df_pred[feature_names].replace(0, np.nan).isna().all(axis=0).any()
                # )
                # print(train_zeroed, pred_zeroed)
                # if train_zeroed or pred_zeroed:
                #     skipped.append(
                #         (file.name, "No valid feature columns after processing")
                #     )
                #     continue
                # print(df_train.head())
                # print(df_pred.head())

                # If nane, skip
                # if df_train.isna().sum().sum() > 0 or df_pred.isna().sum().sum() > 0:
                #     skipped.append((file.name, "NaN values found after processing"))
                #     continue

                trainable_dfs.append(df_train)
                predict_dfs.append(df_pred)
            except Exception as e:
                skipped.append((file.name, str(e)))

        if not trainable_dfs:
            raise ValueError("No valid data files loaded")

        df_all = pd.concat(trainable_dfs, ignore_index=True)
        df_predict_all = pd.concat(predict_dfs, ignore_index=True)

        # Select subset of features if specified
        if feature_names:
            meta_cols = ["quarter", "rating", "target_rating", "ticker", "index"]
            keep_cols = [c for c in meta_cols if c in df_all.columns] + feature_names
            df_all = df_all[[c for c in keep_cols if c in df_all.columns]]
            df_predict_all = df_predict_all[
                [c for c in keep_cols if c in df_predict_all.columns]
            ]

        # Process features
        df_all = self._process_features(df_all)
        df_predict_all = self._process_features(df_predict_all)

        # Store feature columns (use provided list or derive from data)
        if feature_names is not None:
            self.feature_cols = [c for c in feature_names if c in df_all.columns]
        else:
            self.feature_cols = [
                col
                for col in df_all.columns
                if col not in ["quarter", "rating", "target_rating", "ticker", "index"]
            ]

        # Define rating collapse mapping - Simplifies with such small data
        self.rating_map = {
            "Aaa": 0,  # Prime
            "Aa3": 1,
            "Aa2": 1,
            "Aa1": 1,  # High
            "A3": 2,
            "A2": 2,
            "A1": 2,  # Medium (41 samples)
            "Baa3": 3,
            "Baa2": 3,
            "Baa1": 3,  # Lower
            "Ba2": 3,
            "Ba1": 3,  # Non-Investment Grade Speculative (1 sample)
            # Add more once I download more data
        }

        # Decode to rating names
        self.class_names = {
            "0": "Prime",
            "1": "High",
            "2": "Medium",
            "3": "Low",
            # For this exercise - still good though
        }

        # Apply mapping to collapse classes
        df_all["target_encoded"] = df_all["target_rating"].map(self.rating_map)

        # Verify no missing mappings
        if df_all["target_encoded"].isna().any():
            unmapped = df_all[df_all["target_encoded"].isna()]["target_rating"].unique()
            raise ValueError(f"Unmapped ratings found: {unmapped}")

        # Split data
        X = df_all[self.feature_cols].values
        y = df_all["target_encoded"].values
        metadata = df_all[["ticker", "quarter", "rating", "target_rating"]].reset_index(
            drop=True
        )

        # Check class distribution for stratification
        class_counts = pd.Series(y).value_counts()
        min_class_count = class_counts.min()

        if min_class_count < 2 and self.verbose:
            print(
                "\nWarning: Some classes have only 1 sample. Using non-stratified split."
            )

        stratify_y = y if min_class_count >= 2 else None

        # Train+val vs test
        X_temp, self._X_test, y_temp, self._y_test, meta_temp, self._meta_test = (
            train_test_split(
                X,
                y,
                metadata,
                test_size=self.test_size,
                stratify=stratify_y,
                random_state=self.random_state,
            )
        )

        # Train vs val - recheck stratification
        val_ratio = self.val_size / (1 - self.test_size)
        temp_class_counts = pd.Series(y_temp).value_counts()
        stratify_temp = y_temp if temp_class_counts.min() >= 2 else None

        (
            self._X_train,
            self._X_val,
            self._y_train,
            self._y_val,
            self._meta_train,
            self._meta_val,
        ) = train_test_split(
            X_temp,
            y_temp,
            meta_temp,
            test_size=val_ratio,
            stratify=stratify_temp,
            random_state=self.random_state,
        )

        # Prediction set
        self._X_predict = df_predict_all[self.feature_cols].values
        self._meta_predict = df_predict_all[
            ["ticker", "quarter", "rating"]
        ].reset_index(drop=True)

        self.is_fitted = True

        return self

    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get training data as numpy arrays."""
        if not self.is_fitted:
            raise ValueError("Dataset not loaded. Call load() first.")
        return self._X_train, self._y_train

    def get_val_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get validation data as numpy arrays."""
        if not self.is_fitted:
            raise ValueError("Dataset not loaded. Call load() first.")
        return self._X_val, self._y_val

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get test data as numpy arrays."""
        if not self.is_fitted:
            raise ValueError("Dataset not loaded. Call load() first.")
        return self._X_test, self._y_test

    def get_predict_data(self) -> np.ndarray:
        """Get prediction data (future quarters with unknown ratings)."""
        if not self.is_fitted:
            raise ValueError("Dataset not loaded. Call load() first.")
        return self._X_predict

    def get_train_dataset(self, batch_size: int = 32) -> tf.data.Dataset:
        """Get TensorFlow Dataset for training."""
        X, y = self.get_train_data()
        dataset = tf.data.Dataset.from_tensor_slices(
            (X.astype(np.float32), y.astype(np.int32))
        )
        return dataset.shuffle(len(X)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def get_val_dataset(self, batch_size: int = 32) -> tf.data.Dataset:
        """Get TensorFlow Dataset for validation."""
        X, y = self.get_val_data()
        dataset = tf.data.Dataset.from_tensor_slices(
            (X.astype(np.float32), y.astype(np.int32))
        )
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def get_test_dataset(self, batch_size: int = 32) -> tf.data.Dataset:
        """Get TensorFlow Dataset for testing."""
        X, y = self.get_test_data()
        dataset = tf.data.Dataset.from_tensor_slices(
            (X.astype(np.float32), y.astype(np.int32))
        )
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def get_predict_dataset(self, batch_size: int = 32) -> tf.data.Dataset:
        """Get TensorFlow Dataset for prediction."""
        X = self.get_predict_data()
        dataset = tf.data.Dataset.from_tensor_slices(X.astype(np.float32))
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def get_metadata(self, split: str = "train") -> pd.DataFrame:
        """Get metadata (ticker, quarter, ratings) for a split."""
        if not self.is_fitted:
            raise ValueError("Dataset not loaded. Call load() first.")

        metadata_map = {
            "train": self._meta_train,
            "val": self._meta_val,
            "test": self._meta_test,
            "predict": self._meta_predict,
        }

        if split not in metadata_map:
            raise ValueError(f"Split must be one of {list(metadata_map.keys())}")

        return metadata_map[split].copy()

    def decode_labels(self, indices):
        labels = []
        for ind in indices:
            labels.append(self.class_names[str(ind)])
        return labels

    def get_info(self) -> Dict:
        """Get dataset information."""
        if not self.is_fitted:
            raise ValueError("Dataset not loaded. Call load() first.")

        # Get class distributions
        train_dist = pd.Series(self._y_train).value_counts().sort_index()
        val_dist = pd.Series(self._y_val).value_counts().sort_index()
        test_dist = pd.Series(self._y_test).value_counts().sort_index()

        info = {
            "n_train": len(self._X_train),
            "n_val": len(self._X_val),
            "n_test": len(self._X_test),
            "n_predict": len(self._X_predict),
            "n_features": len(self.feature_cols),
            "n_classes": len(self.class_names),
            "feature_names": self.feature_cols,
            "classes": list(self.class_names.values()),
        }

        # Pretty print
        print("\n" + "=" * 70)
        print("DATASET INFORMATION")
        print("=" * 70)
        print(f"Total Samples: {info['n_train'] + info['n_val'] + info['n_test']}")
        print(f"Features:      {info['n_features']}")
        print(f"Classes:       {info['n_classes']}")

        print(
            f"\n{'Rating':<10} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8} {'%':>7}"
        )
        print("-" * 70)

        for i, rating in enumerate(list(self.class_names.values())):
            train_count = train_dist.get(i, 0)
            val_count = val_dist.get(i, 0)
            test_count = test_dist.get(i, 0)
            total = train_count + val_count + test_count
            pct = 100 * total / (info["n_train"] + info["n_val"] + info["n_test"])

            print(
                f"{rating:<10} {train_count:>8} {val_count:>8} {test_count:>8} "
                f"{total:>8} {pct:>6.1f}%"
            )

        print("-" * 70)
        print(
            f"{'TOTAL':<10} {info['n_train']:>8} {info['n_val']:>8} "
            f"{info['n_test']:>8} {info['n_train'] + info['n_val'] + info['n_test']:>8} "
            f"{'100.0%':>7}"
        )
        print(f"\nPrediction set: {info['n_predict']} samples")
        print("=" * 70 + "\n")

        return info

    def save(self, output_dir: str = "/scratch/datasets/jpm/processed/credit"):
        """Save processed datasets and metadata."""
        if not self.is_fitted:
            raise ValueError("Dataset not loaded. Call load() first.")

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save arrays
        np.save(output_path / "X_train.npy", self._X_train)
        np.save(output_path / "y_train.npy", self._y_train)
        np.save(output_path / "X_val.npy", self._X_val)
        np.save(output_path / "y_val.npy", self._y_val)
        np.save(output_path / "X_test.npy", self._X_test)
        np.save(output_path / "y_test.npy", self._y_test)
        np.save(output_path / "X_predict.npy", self._X_predict)

        # Save metadata
        self._meta_train.to_parquet(output_path / "meta_train.parquet")
        self._meta_val.to_parquet(output_path / "meta_val.parquet")
        self._meta_test.to_parquet(output_path / "meta_test.parquet")
        self._meta_predict.to_parquet(output_path / "meta_predict.parquet")

        if self.verbose:
            print(f"Datasets saved to {output_path}")


# Example usage
if __name__ == "__main__":
    # Initialize and load
    dataset = CreditDataset(data_dir="/scratch/datasets/jpm/ratings", verbose=True)
    dataset.load()

    # Get data for XGBoost
    X_train, y_train = dataset.get_train_data()
    X_val, y_val = dataset.get_val_data()

    # Or get TensorFlow datasets
    train_ds = dataset.get_train_dataset(batch_size=64)
    val_ds = dataset.get_val_dataset(batch_size=64)

    # Get info
    dataset.get_info()

    # Save for later
    # dataset.save()
