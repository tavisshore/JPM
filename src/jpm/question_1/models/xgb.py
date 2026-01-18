#!/usr/bin/env python3
"""
CreditRatingModel class for XGBoost-based credit rating prediction.
"""

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
    precision_recall_fscore_support,
    roc_auc_score,
)


class CreditRatingModel:
    """
    XGBoost model for credit rating prediction with training and evaluation.
    """

    def __init__(
        self,
        n_classes: int,
        n_features: int,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        early_stopping_rounds: int = 10,
        use_gpu: bool = False,
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
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.use_gpu = use_gpu

        self.model: Optional[xgb.XGBClassifier] = None
        self.history: Dict[str, List[float]] = {"train": [], "val": []}
        self.best_iteration: Optional[int] = None
        self.feature_importance: Optional[pd.DataFrame] = None

    def build(self) -> "CreditRatingModel":
        """Build XGBoost classifier."""
        params = {
            "objective": "multi:softprob",
            "num_class": self.n_classes,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "eval_metric": "mlogloss",
            "device": "cuda" if self.use_gpu else "cpu",  # Changed from tree_method
        }

        self.model = xgb.XGBClassifier(**params)
        print(f"Model built with {self.n_classes} classes, {self.n_features} features")
        print(f"Device: {'GPU (CUDA)' if self.use_gpu else 'CPU'}")
        return self

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        verbose: bool = True,
    ) -> "CreditRatingModel":
        """Train the model with early stopping."""
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")

        print("Training XGBoost model...")

        # Ensure data types are correct
        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        y_train = y_train.astype(np.int32)
        y_val = y_val.astype(np.int32)

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=verbose,
        )

        # Extract training history
        results = self.model.evals_result()
        self.history["train"] = results["validation_0"]["mlogloss"]
        self.history["val"] = results["validation_1"]["mlogloss"]

        # Get best iteration - use last iteration if no early stopping
        self.best_iteration = len(self.history["train"]) - 1

        print(f"Training complete. Iterations: {len(self.history['train'])}")
        print(f"Final train loss: {self.history['train'][-1]:.4f}")
        print(f"Final val loss: {self.history['val'][-1]:.4f}")
        print(f"Best val loss: {min(self.history['val']):.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self.model is None:
            raise ValueError("Model not trained.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model not trained.")
        return self.model.predict_proba(X)

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
        y_proba = self.predict_proba(X)

        # Get unique classes in predictions to match class_names
        unique_classes = np.unique(np.concatenate([y, y_pred]))

        # Filter class_names to only include classes present in data
        if class_names is not None and len(class_names) != len(unique_classes):
            class_names = [class_names[i] for i in unique_classes]

        # Basic metrics
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y, y_pred, average="weighted", zero_division=0
        )

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = (
            precision_recall_fscore_support(y, y_pred, average=None, zero_division=0)
        )

        # AUC (one-vs-rest for multiclass)
        try:
            auc = roc_auc_score(y, y_proba, multi_class="ovr", average="weighted")
        except ValueError:
            auc = None

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "precision_per_class": precision_per_class.tolist(),
            "recall_per_class": recall_per_class.tolist(),
            "f1_per_class": f1_per_class.tolist(),
        }

        print(f"\n{'=' * 60}")
        print(f"{split_name.upper()} SET EVALUATION")
        print(f"{'=' * 60}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        if auc:
            print(f"AUC:       {auc:.4f}")

        # Detailed classification report - use labels parameter
        print(
            f"\n{classification_report(y, y_pred, labels=unique_classes, target_names=class_names, zero_division=0)}"
        )

        # Confusion matrix
        cm = confusion_matrix(y, y_pred, labels=unique_classes)
        print("\nConfusion Matrix:")
        print(cm)

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

    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training and validation loss curves."""
        if not self.history["train"]:
            print("No training history available.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.history["train"], label="Train Loss", linewidth=2)
        plt.plot(self.history["val"], label="Val Loss", linewidth=2)

        if self.best_iteration is not None:
            plt.axvline(
                x=self.best_iteration,
                color="r",
                linestyle="--",
                label=f"Best Iteration ({self.best_iteration})",
            )

        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel("Log Loss", fontsize=12)
        plt.title("Training History", fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Training history plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        save_path: Optional[str] = None,
        normalize: bool = False,
    ):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            fmt = ".2f"
            title = "Normalized Confusion Matrix"
        else:
            fmt = "d"
            title = "Confusion Matrix"

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={"label": "Proportion" if normalize else "Count"},
        )
        plt.xlabel("Predicted", fontsize=12)
        plt.ylabel("True", fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_feature_importance(self, top_n: int = 20, save_path: Optional[str] = None):
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
            print(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def save(self, output_dir: str = "/scratch/models/credit_rating"):
        """Save model and training artifacts."""
        if self.model is None:
            raise ValueError("Model not trained.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save XGBoost model
        self.model.save_model(str(output_path / "xgboost_model.json"))

        # Save training history and metadata
        metadata = {
            "n_classes": self.n_classes,
            "n_features": self.n_features,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "best_iteration": self.best_iteration,
            "history": self.history,
        }

        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save feature importance if available
        if self.feature_importance is not None:
            self.feature_importance.to_csv(
                output_path / "feature_importance.csv", index=False
            )

        print(f"Model saved to {output_path}")

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


# Example usage
if __name__ == "__main__":
    from credit_dataset import CreditDataset

    # Load dataset
    dataset = CreditDataset(data_dir="/scratch/datasets/jpm")
    dataset.load()

    # Get data
    X_train, y_train = dataset.get_train_data()
    X_val, y_val = dataset.get_val_data()
    X_test, y_test = dataset.get_test_data()

    info = dataset.get_info()

    # Build and train model
    model = CreditRatingModel(
        n_classes=info["n_classes"],
        n_features=info["n_features"],
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        early_stopping_rounds=20,
    )

    model.build()
    model.train(X_train, y_train, X_val, y_val, verbose=True)

    # Evaluate
    test_metrics = model.evaluate(
        X_test, y_test, class_names=info["classes"], split_name="test"
    )

    # Feature importance
    feature_imp = model.compute_feature_importance(info["feature_names"])
    print("\nTop 10 Features:")
    print(feature_imp.head(10))

    # Plots
    output_dir = Path("/scratch/models/credit_rating")
    output_dir.mkdir(parents=True, exist_ok=True)

    model.plot_training_history(save_path=output_dir / "training_history.png")
    model.plot_confusion_matrix(
        y_test,
        model.predict(X_test),
        class_names=info["classes"],
        save_path=output_dir / "confusion_matrix.png",
    )
    model.plot_feature_importance(
        top_n=20, save_path=output_dir / "feature_importance.png"
    )

    # Make predictions on future quarters
    X_predict = dataset.get_predict_data()
    predictions = model.predict(X_predict)
    probabilities = model.predict_proba(X_predict)

    predicted_ratings = dataset.decode_labels(predictions)
    meta_predict = dataset.get_metadata("predict")

    results = meta_predict.copy()
    results["predicted_rating"] = predicted_ratings
    results["confidence"] = probabilities.max(axis=1)

    print("\nPredictions for future quarters:")
    print(results)

    # Save
    model.save()
    results.to_csv(output_dir / "predictions.csv", index=False)
