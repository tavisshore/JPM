"""Shared test fixtures and dummy classes for question_1 tests."""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf


class DummyDataLoader:
    """Dummy EdgarData replacement for testing."""

    def __init__(self, with_val: bool = True):
        self.num_features = 5
        self.targets = [f"feature{i}" for i in range(5)]
        self.feature_mappings = {
            "assets": [0, 1],
            "liabilities": [2, 3],
            "equity": [4],
        }
        self.bs_keys = self.targets
        # bs_structure should be flat dict without nesting
        self.bs_structure = {
            "Assets": ["feature0", "feature1"],
            "Liabilities": ["feature2", "feature3"],
            "Equity": ["feature4"],
        }
        self.is_structure = {
            "Revenues": [],
            "Expenses": [],
        }
        self.target_mean = np.zeros(5, dtype=np.float64)
        self.target_std = np.ones(5, dtype=np.float64)
        self.feat_to_idx = {f"feature{i}": i for i in range(5)}
        # Create data with correct lookback and feature dimensions
        x = np.zeros((2, 1, 5), dtype=np.float64)
        y = np.zeros((2, 5), dtype=np.float64)
        self.train_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(2)
        self.val_dataset = (
            tf.data.Dataset.from_tensor_slices((x, y)).batch(2) if with_val else None
        )


class DummyStatementsDataset:
    """Dummy StatementsDataset replacement for testing LSTMForecaster."""

    def __init__(
        self,
        num_features: int = 5,
        num_samples: int = 10,
        lookback: int = 4,
        with_val: bool = True,
    ):
        self.num_features = num_features
        self.num_targets = num_features
        self.tgt_indices = list(range(num_features))

        # Feature names - ensure we have "Net Income" as the last feature
        if num_features >= 1:
            self.targets = [f"feature_{i}" for i in range(num_features - 1)] + [
                "Net Income"
            ]
        else:
            self.targets = [f"feature_{i}" for i in range(num_features)]

        self.bs_keys = self.targets[:3]  # First 3 features are balance sheet

        # Feature mappings for balance sheet identity
        self.feature_mappings = {
            "assets": [0],
            "liabilities": [1],
            "equity": [2],
        }

        # Balance sheet structure (flat dict format)
        self.bs_structure = {
            "Assets": ["feature_0"],
            "Liabilities": ["feature_1"],
            "Equity": ["feature_2"],
        }

        # Income statement structure
        self.is_structure = {
            "Revenues": ["feature_3"] if num_features > 3 else [],
            "Expenses": ["Net Income"] if num_features >= 1 else [],
        }

        # Scaler stats
        self.target_mean = np.zeros(num_features, dtype=np.float64)
        self.target_std = np.ones(num_features, dtype=np.float64)
        self.full_mean = self.target_mean
        self.full_std = self.target_std

        # Feature index mapping
        self.feat_to_idx = {name: i for i, name in enumerate(self.targets)}
        self.name_to_target_idx = self.feat_to_idx

        # Create dummy data
        X = np.random.randn(num_samples, lookback, num_features).astype(np.float64)
        y = np.random.randn(num_samples, num_features).astype(np.float64)

        # Split into train/val
        split_idx = int(num_samples * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_val
        self.y_test = y_val

        # TensorFlow datasets
        self.train_dataset = (
            tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(len(X_train))
            .batch(2)
            .prefetch(tf.data.AUTOTUNE)
        )

        self.val_dataset = (
            tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(2)
            if with_val
            else None
        )


class DummyCreditDataset:
    """Dummy CreditDataset replacement for testing XGBoost models."""

    def __init__(
        self,
        num_features: int = 5,
        num_classes: int = 4,
        num_samples: int = 100,
    ):
        self.num_features = num_features
        self.n_classes = num_classes
        self.is_fitted = True

        # Feature columns
        self.feature_cols = [f"feature_{i}" for i in range(num_features)]

        # Rating mappings
        self.rating_map = {f"Rating_{i}": i for i in range(num_classes)}
        self.class_names = {str(i): f"Class_{i}" for i in range(num_classes)}

        # Generate random data
        np.random.seed(42)
        X = np.random.randn(num_samples, num_features).astype(np.float32)
        y = np.random.randint(0, num_classes, size=num_samples)

        # Split into train/val/test
        train_end = int(num_samples * 0.7)
        val_end = int(num_samples * 0.85)

        self._X_train = X[:train_end]
        self._y_train = y[:train_end]
        self._X_val = X[train_end:val_end]
        self._y_val = y[train_end:val_end]
        self._X_test = X[val_end:]
        self._y_test = y[val_end:]
        self._X_predict = X[:5]  # Small prediction set

        # Metadata
        self._meta_train = pd.DataFrame(
            {
                "ticker": ["AAPL"] * len(self._X_train),
                "quarter": pd.date_range(
                    "2020-01-01", periods=len(self._X_train), freq="QE"
                ),
                "rating": ["A1"] * len(self._X_train),
                "target_rating": ["A1"] * len(self._X_train),
            }
        )
        self._meta_val = pd.DataFrame(
            {
                "ticker": ["AAPL"] * len(self._X_val),
                "quarter": pd.date_range(
                    "2022-01-01", periods=len(self._X_val), freq="QE"
                ),
                "rating": ["A1"] * len(self._X_val),
                "target_rating": ["A1"] * len(self._X_val),
            }
        )
        self._meta_test = pd.DataFrame(
            {
                "ticker": ["AAPL"] * len(self._X_test),
                "quarter": pd.date_range(
                    "2023-01-01", periods=len(self._X_test), freq="QE"
                ),
                "rating": ["A1"] * len(self._X_test),
                "target_rating": ["A1"] * len(self._X_test),
            }
        )
        self._meta_predict = pd.DataFrame(
            {
                "ticker": ["AAPL"] * len(self._X_predict),
                "quarter": pd.date_range(
                    "2024-01-01", periods=len(self._X_predict), freq="QE"
                ),
                "rating": ["A1"] * len(self._X_predict),
            }
        )

    def load(self, feature_names: List[str] | None = None) -> "DummyCreditDataset":
        """Mimic load() method - already loaded in __init__."""
        if feature_names is not None:
            # Filter to requested features
            n = min(len(feature_names), self.num_features)
            self.feature_cols = feature_names[:n]
            self._X_train = self._X_train[:, :n]
            self._X_val = self._X_val[:, :n]
            self._X_test = self._X_test[:, :n]
            self._X_predict = self._X_predict[:, :n]
        return self

    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._X_train, self._y_train

    def get_val_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._X_val, self._y_val

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._X_test, self._y_test

    def get_predict_data(self) -> np.ndarray:
        return self._X_predict

    def get_metadata(self, split: str = "train") -> pd.DataFrame:
        metadata_map = {
            "train": self._meta_train,
            "val": self._meta_val,
            "test": self._meta_test,
            "predict": self._meta_predict,
        }
        return metadata_map[split].copy()

    def decode_labels(self, indices) -> List[str]:
        return [self.class_names[str(int(i))] for i in indices]

    def get_info(self) -> Dict:
        return {
            "n_train": len(self._X_train),
            "n_val": len(self._X_val),
            "n_test": len(self._X_test),
            "n_predict": len(self._X_predict),
            "n_features": len(self.feature_cols),
            "n_classes": self.n_classes,
            "feature_names": self.feature_cols,
            "classes": list(self.class_names.values()),
        }

    def get_train_dataset(self, batch_size: int = 32) -> tf.data.Dataset:
        X, y = self.get_train_data()
        return (
            tf.data.Dataset.from_tensor_slices((X, y.astype(np.int32)))
            .shuffle(len(X))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

    def get_val_dataset(self, batch_size: int = 32) -> tf.data.Dataset:
        X, y = self.get_val_data()
        return (
            tf.data.Dataset.from_tensor_slices((X, y.astype(np.int32)))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

    def get_test_dataset(self, batch_size: int = 32) -> tf.data.Dataset:
        X, y = self.get_test_data()
        return (
            tf.data.Dataset.from_tensor_slices((X, y.astype(np.int32)))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )


# Pytest fixtures
@pytest.fixture
def dummy_data_loader():
    """Fixture providing a DummyDataLoader instance."""
    return DummyDataLoader()


@pytest.fixture
def dummy_statements_dataset():
    """Fixture providing a DummyStatementsDataset instance."""
    return DummyStatementsDataset()


@pytest.fixture
def dummy_credit_dataset():
    """Fixture providing a DummyCreditDataset instance."""
    return DummyCreditDataset()
