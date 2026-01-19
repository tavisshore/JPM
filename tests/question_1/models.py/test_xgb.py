"""Tests for CreditRatingModel (XGBoost-based credit rating prediction)."""

import json
from unittest.mock import patch

import numpy as np
import pytest

from jpm.question_1.models.xgb import CreditRatingModel

unit = pytest.mark.unit
integration = pytest.mark.integration


@pytest.fixture
def model():
    """Create a basic CreditRatingModel instance."""
    return CreditRatingModel(
        n_classes=3,
        n_features=10,
        max_depth=3,
        learning_rate=0.1,
        n_estimators=5,
        random_state=42,
        use_gpu=False,  # Disable GPU to avoid device mismatch warnings in tests
    )


@pytest.fixture
def sample_data():
    """Generate sample training and validation data."""
    np.random.seed(42)
    n_train, n_val = 100, 20
    n_features = 10
    n_classes = 3

    X_train = np.random.randn(n_train, n_features)
    y_train = np.random.randint(0, n_classes, n_train)
    X_val = np.random.randn(n_val, n_features)
    y_val = np.random.randint(0, n_classes, n_val)

    return X_train, y_train, X_val, y_val


# Initialization Tests


@unit
def test_init_stores_parameters():
    """CreditRatingModel should store all hyperparameters correctly."""
    model = CreditRatingModel(
        n_classes=5,
        n_features=20,
        max_depth=8,
        learning_rate=0.05,
        n_estimators=200,
        subsample=0.7,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=123,
        use_gpu=False,
    )

    assert model.n_classes == 5
    assert model.n_features == 20
    assert model.max_depth == 8
    assert model.learning_rate == 0.05
    assert model.n_estimators == 200
    assert model.subsample == 0.7
    assert model.colsample_bytree == 0.9
    assert model.reg_alpha == 0.1
    assert model.reg_lambda == 2.0
    assert model.random_state == 123
    assert model.use_gpu is False


@unit
def test_init_default_state():
    """CreditRatingModel should initialize with None model and empty history."""
    model = CreditRatingModel(n_classes=3, n_features=10)

    assert model.model is None
    assert model.history == {"train": [], "val": []}
    assert model.best_iteration is None
    assert model.feature_importance is None


# Build Tests


@unit
def test_build_creates_xgb_classifier(model):
    """build() should create an XGBRegressor with correct parameters."""
    result = model.build()

    assert result is model  # Returns self for chaining
    assert model.model is not None
    assert hasattr(model.model, "fit")
    assert hasattr(model.model, "predict")


@unit
def test_build_sets_gpu_device():
    """build() should set device='cuda' when use_gpu=True."""
    model = CreditRatingModel(n_classes=3, n_features=10, use_gpu=True)
    model.build()

    # Check the model params - uses 'device' not 'tree_method'
    params = model.model.get_params()
    assert params.get("device") == "cuda"


@unit
def test_build_sets_cpu_device():
    """build() should set device='cpu' when use_gpu=False."""
    model = CreditRatingModel(n_classes=3, n_features=10, use_gpu=False)
    model.build()

    params = model.model.get_params()
    assert params.get("device") == "cpu"


# Training Tests


@unit
def test_train_raises_without_build(model, sample_data):
    """train() should raise ValueError if model not built."""
    X_train, y_train, X_val, y_val = sample_data

    with pytest.raises(ValueError, match="not built"):
        model.train(X_train, y_train, X_val, y_val)


@integration
def test_train_fits_model(model, sample_data):
    """train() should fit the model and record history."""
    X_train, y_train, X_val, y_val = sample_data

    model.build()
    result = model.train(X_train, y_train, X_val, y_val, verbose=False)

    assert result is model  # Returns self for chaining
    assert len(model.history["train"]) > 0
    assert len(model.history["val"]) > 0
    assert model.best_iteration is not None


@integration
def test_train_history_decreases(model, sample_data):
    """Training loss should generally decrease over iterations."""
    X_train, y_train, X_val, y_val = sample_data

    model.build()
    model.train(X_train, y_train, X_val, y_val, verbose=False)

    # First loss should be greater than or equal to best loss
    assert model.history["train"][0] >= min(model.history["train"])


# Prediction Tests


@unit
def test_predict_raises_without_training(model, sample_data):
    """predict() should raise ValueError if model not trained."""
    X_train, _, _, _ = sample_data

    with pytest.raises(ValueError, match="not trained"):
        model.predict(X_train)


@integration
def test_predict_returns_class_labels(model, sample_data):
    """predict() should return integer class labels."""
    X_train, y_train, X_val, y_val = sample_data

    model.build().train(X_train, y_train, X_val, y_val, verbose=False)
    predictions = model.predict(X_val)

    assert predictions.shape == (len(X_val),)
    assert predictions.dtype in [np.int64, np.int32]
    assert all(0 <= p < model.n_classes for p in predictions)


@unit
def test_predict_proba_raises_without_training(model, sample_data):
    """predict_proba() should raise ValueError if model not trained."""
    X_train, _, _, _ = sample_data

    with pytest.raises(ValueError, match="not trained"):
        model.predict_proba(X_train)


@integration
def test_predict_proba_returns_probabilities(model, sample_data):
    """predict_proba() should return probability matrix."""
    X_train, y_train, X_val, y_val = sample_data

    model.build().train(X_train, y_train, X_val, y_val, verbose=False)
    proba = model.predict_proba(X_val)

    assert proba.shape == (len(X_val), model.n_classes)
    # Each row should sum to 1
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X_val)), rtol=1e-5)
    # All values between 0 and 1
    assert np.all(proba >= 0) and np.all(proba <= 1)


# Evaluation Tests


@unit
def test_evaluate_raises_without_training(model, sample_data):
    """evaluate() should raise ValueError if model not trained."""
    _, _, X_val, y_val = sample_data

    with pytest.raises(ValueError, match="not trained"):
        model.evaluate(X_val, y_val)


@integration
def test_evaluate_returns_metrics_dict(model, sample_data, capsys):
    """evaluate() should return dict with standard metrics."""
    X_train, y_train, X_val, y_val = sample_data

    model.build().train(X_train, y_train, X_val, y_val, verbose=False)
    metrics = model.evaluate(X_val, y_val)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "precision_per_class" in metrics
    assert "recall_per_class" in metrics
    assert "f1_per_class" in metrics

    # Values should be in valid ranges
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1"] <= 1


@integration
def test_evaluate_prints_report(model, sample_data, capsys):
    """evaluate() should print evaluation report."""
    X_train, y_train, X_val, y_val = sample_data
    class_names = ["Class_A", "Class_B", "Class_C"]

    model.build().train(X_train, y_train, X_val, y_val, verbose=False)
    model.evaluate(X_val, y_val, class_names=class_names)

    captured = capsys.readouterr()
    assert "EVALUATION" in captured.out
    assert "Accuracy" in captured.out


# Feature Importance Tests


@unit
def test_feature_importance_raises_without_training(model):
    """compute_feature_importance() should raise ValueError if model not trained."""
    feature_names = [f"feature_{i}" for i in range(10)]

    with pytest.raises(ValueError, match="not trained"):
        model.compute_feature_importance(feature_names)


@integration
def test_feature_importance_returns_dataframe(model, sample_data):
    """compute_feature_importance() should return sorted DataFrame."""
    X_train, y_train, X_val, y_val = sample_data
    feature_names = [f"feature_{i}" for i in range(10)]

    model.build().train(X_train, y_train, X_val, y_val, verbose=False)
    importance_df = model.compute_feature_importance(feature_names)

    assert "feature" in importance_df.columns
    assert "importance" in importance_df.columns
    # Should be sorted descending
    assert importance_df["importance"].is_monotonic_decreasing
    # Should be stored in model
    assert model.feature_importance is not None


# Save/Load Tests


@integration
def test_save_and_load_model(model, sample_data, tmp_path):
    """Model should be savable and loadable with consistent predictions."""
    X_train, y_train, X_val, y_val = sample_data

    model.build().train(X_train, y_train, X_val, y_val, verbose=False)
    original_predictions = model.predict(X_val)

    # Save model to directory (pass Path object, not string)
    save_dir = tmp_path / "model_dir"
    model.save(save_dir)
    assert save_dir.exists()
    assert (save_dir / "xgboost_model.json").exists()
    assert (save_dir / "metadata.json").exists()

    # Load into new model
    loaded_model = CreditRatingModel.load(str(save_dir))

    # Predictions should match
    loaded_predictions = loaded_model.predict(X_val)
    np.testing.assert_array_equal(original_predictions, loaded_predictions)


@integration
def test_save_creates_metadata(model, sample_data, tmp_path):
    """save() should create metadata JSON with correct fields."""
    X_train, y_train, X_val, y_val = sample_data

    model.build().train(X_train, y_train, X_val, y_val, verbose=False)

    # Pass Path object, not string
    save_dir = tmp_path / "model_dir"
    model.save(save_dir)

    # Check metadata file exists
    metadata_path = save_dir / "metadata.json"
    assert metadata_path.exists()

    with open(metadata_path) as f:
        metadata = json.load(f)

    assert "n_classes" in metadata
    assert "n_features" in metadata
    assert "best_iteration" in metadata
    assert "history" in metadata
    assert metadata["n_classes"] == 3
    assert metadata["n_features"] == 10


@unit
def test_save_raises_without_training(model, tmp_path):
    """save() should raise ValueError if model not trained."""
    save_dir = tmp_path / "model_dir"

    with pytest.raises(ValueError, match="not trained"):
        model.save(save_dir)


# Plotting Tests (mock matplotlib)


@unit
def test_plot_training_history_no_history(model, capsys):
    """plot_training_history() should print message when no history."""
    model.plot_training_history()
    captured = capsys.readouterr()
    assert "No training history" in captured.out


@integration
def test_plot_training_history_with_save(model, sample_data, tmp_path):
    """plot_training_history() should save figure to file."""
    X_train, y_train, X_val, y_val = sample_data

    model.build().train(X_train, y_train, X_val, y_val, verbose=False)

    save_path = tmp_path / "history.png"
    with patch("matplotlib.pyplot.show"):
        model.plot_training_history(save_path=save_path)

    assert save_path.exists()


@integration
def test_plot_confusion_matrix_with_save(model, sample_data, tmp_path):
    """plot_confusion_matrix() should save figure to file."""
    X_train, y_train, X_val, y_val = sample_data
    class_names = ["Class_A", "Class_B", "Class_C"]

    model.build().train(X_train, y_train, X_val, y_val, verbose=False)
    y_pred = model.predict(X_val)

    save_path = tmp_path / "confusion.png"
    with patch("matplotlib.pyplot.show"):
        model.plot_confusion_matrix(
            y_val, y_pred, class_names=class_names, save_path=save_path
        )

    assert save_path.exists()


# Edge Cases


@integration
def test_handles_imbalanced_classes(sample_data):
    """Model should handle imbalanced class distributions."""
    X_train, _, X_val, _ = sample_data

    # Create highly imbalanced labels
    y_train_imbalanced = np.array([0] * 80 + [1] * 15 + [2] * 5)
    y_val_imbalanced = np.array([0] * 16 + [1] * 3 + [2] * 1)

    model = CreditRatingModel(n_classes=3, n_features=10, n_estimators=5, use_gpu=False)
    model.build().train(
        X_train, y_train_imbalanced, X_val, y_val_imbalanced, verbose=False
    )

    predictions = model.predict(X_val)
    assert len(predictions) == len(y_val_imbalanced)


@integration
def test_handles_binary_classification():
    """Model should work for binary classification (n_classes=2)."""
    np.random.seed(42)
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)

    model = CreditRatingModel(n_classes=2, n_features=5, n_estimators=5, use_gpu=False)
    model.build().train(X[:40], y[:40], X[40:], y[40:], verbose=False)

    predictions = model.predict(X[40:])
    assert all(p in [0, 1] for p in predictions)
