import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    precision_recall_fscore_support,
)


class CreditRatingModel:
    """
    XGBoost ordinal regression model for credit rating prediction.
    Treats ratings as ordered categories (0=worst, n-1=best).
    """

    def __init__(
        self,
        n_classes: int,
        n_features: int,
        max_depth=4,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=3,
        gamma=0.5,
        random_state=42,
        use_gpu=True,
    ):
        self.n_classes = n_classes
        self.n_features = n_features
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.random_state = random_state
        # self.early_stopping_rounds = early_stopping_rounds
        self.use_gpu = use_gpu

        self.model: Optional[xgb.XGBRegressor] = None
        self.history: Dict[str, List[float]] = {"train": [], "val": []}
        self.best_iteration: Optional[int] = None
        self.feature_importance: Optional[pd.DataFrame] = None

    def build(self) -> "CreditRatingModel":
        """Build XGBoost regressor for ordinal regression."""
        params = {
            "objective": "reg:squarederror",
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "min_child_weight": self.min_child_weight,
            "gamma": self.gamma,
            "random_state": self.random_state,
            "eval_metric": "mae",
            "device": "cuda" if self.use_gpu else "cpu",
        }

        self.model = xgb.XGBRegressor(**params)
        print(
            f"Ordinal regression model built with {self.n_classes} ordered classes, {self.n_features} features"
        )
        print(f"Device: {'GPU (CUDA)' if self.use_gpu else 'CPU'}")
        return self

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> "CreditRatingModel":
        """Train the model with early stopping."""
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")

        # Ensure data types are correct
        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        y_train = y_train.astype(np.float32)  # Regression target
        y_val = y_val.astype(np.float32)

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            sample_weight=sample_weight,
            verbose=verbose,
        )

        # Extract training history
        results = self.model.evals_result()
        self.history["train"] = results["validation_0"]["mae"]
        self.history["val"] = results["validation_1"]["mae"]

        # Get best iteration
        self.best_iteration = len(self.history["train"]) - 1

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels by rounding regression output."""
        if self.model is None:
            raise ValueError("Model not trained.")

        # Get continuous predictions and round to nearest class
        raw_pred = self.model.predict(X)
        # Clip to valid range [0, n_classes-1] and round
        return np.clip(np.round(raw_pred), 0, self.n_classes - 1).astype(int)

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Predict raw continuous values (before rounding)."""
        if self.model is None:
            raise ValueError("Model not trained.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Approximate class probabilities using distance from predicted value.
        Not true probabilities but useful for compatibility.
        """
        if self.model is None:
            raise ValueError("Model not trained.")

        raw_pred = self.model.predict(X)
        n_samples = len(raw_pred)
        proba = np.zeros((n_samples, self.n_classes))

        for i, pred in enumerate(raw_pred):
            # Clip to valid range
            pred = np.clip(pred, 0, self.n_classes - 1)
            # Assign probability based on distance to each class
            for c in range(self.n_classes):
                distance = abs(pred - c)
                proba[i, c] = np.exp(-distance)  # Exponential decay
            # Normalize to sum to 1
            proba[i] /= proba[i].sum()

        return proba

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_names: Optional[List[str]] = None,
        split_name: str = "test",
    ) -> Dict:
        """Evaluate model performance."""
        if self.model is None:
            raise ValueError("Model not trained.")

        y_pred = self.predict(X)
        y_raw = self.predict_raw(X)

        # Get unique classes in predictions
        unique_classes = np.unique(np.concatenate([y, y_pred]))

        # Filter class_names to only include classes present in data
        if class_names is not None and len(class_names) != len(unique_classes):
            class_names = [class_names[i] for i in unique_classes]

        # Basic metrics
        accuracy = accuracy_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        mae_raw = mean_absolute_error(y, y_raw)

        # Within-1 accuracy (prediction within 1 rating level)
        within_1_acc = np.mean(np.abs(y - y_pred) <= 1)

        precision, recall, f1, support = precision_recall_fscore_support(
            y, y_pred, average="weighted", zero_division=0
        )

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = (
            precision_recall_fscore_support(y, y_pred, average=None, zero_division=0)
        )

        metrics = {
            "accuracy": accuracy,
            "mae": mae,
            "mae_raw": mae_raw,
            "within_1_accuracy": within_1_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "precision_per_class": precision_per_class.tolist(),
            "recall_per_class": recall_per_class.tolist(),
            "f1_per_class": f1_per_class.tolist(),
        }

        print(f"\n{'=' * 60}")
        print(f"{split_name.upper()} SET EVALUATION (Ordinal Regression)")
        print(f"{'=' * 60}")
        print(f"Exact Accuracy:        {accuracy:.4f}")
        print(f"Within-1 Accuracy:     {within_1_acc:.4f}")
        print(f"MAE (rounded):         {mae:.4f}")
        print(f"MAE (continuous):      {mae_raw:.4f}")
        print(f"Weighted Precision:    {precision:.4f}")
        print(f"Weighted Recall:       {recall:.4f}")
        print(f"Weighted F1 Score:     {f1:.4f}")

        # Detailed classification report
        print(
            f"\n{classification_report(y, y_pred, labels=unique_classes, target_names=class_names, zero_division=0)}"
        )

        return metrics

    def compute_feature_importance(
        self, feature_names: List[str], importance_type: str = "weight"
    ) -> pd.DataFrame:
        """
        Compute and return feature importance.

        importance_type: 'weight', 'gain', or 'cover'
        """
        if self.model is None:
            raise ValueError("Model not trained.")

        importance = self.model.get_booster().get_score(importance_type=importance_type)

        # Convert to DataFrame
        importance_df = pd.DataFrame(
            [
                {"feature": feature_names[int(k.replace("f", ""))], "importance": v}
                for k, v in importance.items()
            ]
        )

        importance_df = importance_df.sort_values(
            "importance", ascending=False
        ).reset_index(drop=True)
        self.feature_importance = importance_df

        return importance_df

    def plot_training_history(self, save_path: Optional[Path] = None):
        """Plot training and validation MAE curves."""
        if not self.history["train"]:
            print("No training history available.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.history["train"], label="Train MAE", linewidth=2)
        plt.plot(self.history["val"], label="Val MAE", linewidth=2)

        if self.best_iteration is not None:
            plt.axvline(
                x=self.best_iteration,
                color="r",
                linestyle="--",
                label=f"Best Iteration ({self.best_iteration})",
            )

        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel("Mean Absolute Error", fontsize=12)
        plt.title(
            "Training History (Ordinal Regression)", fontsize=14, fontweight="bold"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        save_path: Optional[Path] = None,
        normalise: bool = False,
    ):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)

        if normalise:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            fmt = ".2f"
            title = "Normalised Confusion Matrix (Ordinal)"
        else:
            fmt = "d"
            title = "Confusion Matrix (Ordinal)"

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={"label": "Proportion" if normalise else "Count"},
        )
        plt.xlabel("Predicted", fontsize=12)
        plt.ylabel("True", fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

    def plot_feature_importance(
        self, top_n: int = 20, save_path: Optional[Path] = None
    ):
        """Plot top feature importances."""
        if self.feature_importance is None:
            print(
                "Feature importance not computed. Call compute_feature_importance() first."
            )
            return

        top_features = self.feature_importance.head(top_n)

        plt.figure(figsize=(10, max(6, top_n * 0.3)))
        plt.barh(range(len(top_features)), top_features["importance"])
        plt.yticks(range(len(top_features)), top_features["feature"])
        plt.xlabel("Importance", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

    def save(self, output_dir: Path = Path("/scratch/models/credit_rating")):
        """Save model and training artifacts."""
        if self.model is None:
            raise ValueError("Model not trained.")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save XGBoost model
        self.model.save_model(str(output_dir / "xgboost_model.json"))

        # Save training history and metadata
        metadata = {
            "model_type": "ordinal_regression",
            "n_classes": self.n_classes,
            "n_features": self.n_features,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "best_iteration": self.best_iteration,
            "history": self.history,
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save feature importance if available
        if self.feature_importance is not None:
            self.feature_importance.to_csv(
                output_dir / "feature_importance.csv", index=False
            )

        print(f"Model saved to {output_dir}")

    @classmethod
    def load(cls, model_dir: str) -> "CreditRatingModel":
        """Load saved model."""
        model_path = Path(model_dir)

        # Load metadata
        with open(model_path / "metadata.json", "r") as f:
            metadata = json.load(f)

        # Create instance
        instance = cls(
            n_classes=metadata["n_classes"],
            n_features=metadata["n_features"],
            max_depth=metadata["max_depth"],
            learning_rate=metadata["learning_rate"],
            n_estimators=metadata["n_estimators"],
        )

        # Build and load model
        instance.build()
        instance.model.load_model(str(model_path / "xgboost_model.json"))
        instance.history = metadata["history"]
        instance.best_iteration = metadata["best_iteration"]

        # Load feature importance if available
        importance_path = model_path / "feature_importance.csv"
        if importance_path.exists():
            instance.feature_importance = pd.read_csv(importance_path)

        print(f"Model loaded from {model_path}")
        return instance
