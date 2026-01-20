"""
XGBoost Credit Rating Model Training Script.

Trains a gradient boosting classifier to predict credit ratings from financial ratios.
The script performs:

Training Process:
- Load historical credit rating data with financial ratios
- Split data into train/validation/test sets
- Train XGBoost classifier with hyperparameter tuning
- Evaluate model performance on test set

Outputs:
- Trained model with performance metrics (accuracy, F1 score)
- Feature importance analysis
- Confusion matrices (normalized and unnormalized)
- Training history plots
- Predictions on future quarters with confidence scores

The model maps financial health indicators to credit rating categories
(Prime, High, Medium, Low) for risk assessment. # Expanding later
"""

from jpm.config import Config, DataConfig, LLMConfig, LSTMConfig, XGBConfig
from jpm.question_1 import CreditDataset, CreditRatingModel, get_args, set_seed


def train(cfg: Config):
    print("\n[1/5] Loading dataset...")
    dataset = CreditDataset(
        data_dir=cfg.data.cache_dir,
        pattern="*_ratings.parquet",
        val_size=0.15,
        test_size=0.15,
        random_state=42,
        verbose=True,
    )
    dataset.load()

    print("\n[2/5] Dataset summary...")
    info = dataset.get_info()

    print("\n[3/5] Preparing training data...")
    X_train, y_train = dataset.get_train_data()
    X_val, y_val = dataset.get_val_data()
    X_test, y_test = dataset.get_test_data()

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    print("\n[4/5] Training XGBoost model...")
    model = CreditRatingModel(
        n_classes=info["n_classes"], n_features=info["n_features"], **cfg.xgb.to_dict()
    )
    model.build()
    model.train(X_train, y_train, X_val, y_val, verbose=False)

    print("\n[5/5] Evaluating model...")
    test_metrics = model.evaluate(
        X_test, y_test, class_names=info["classes"], split_name="test"
    )

    # Feature importance
    feature_imp = model.compute_feature_importance(info["feature_names"])

    # Generate plots
    model.plot_training_history(save_path=cfg.data.plots_dir / "training_history.png")
    model.plot_confusion_matrix(
        y_test,
        model.predict(X_test),
        class_names=info["classes"],
        save_path=cfg.data.plots_dir / "confusion_matrix.png",
    )
    model.plot_confusion_matrix(
        y_test,
        model.predict(X_test),
        class_names=info["classes"],
        normalise=True,
        save_path=cfg.data.plots_dir / "confusion_matrix_normalised.png",
    )
    model.plot_feature_importance(
        top_n=20, save_path=cfg.data.plots_dir / "feature_importance.png"
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

    # Add top 3 predictions with probabilities
    for i in range(min(3, probabilities.shape[1])):
        top_idx = probabilities.argsort(axis=1)[:, -(i + 1)]
        results[f"top_{i + 1}_rating"] = dataset.decode_labels(top_idx)
        results[f"top_{i + 1}_prob"] = probabilities[range(len(probabilities)), top_idx]

    # Save everything
    model.save(cfg.data.save_dir)
    results.to_csv(cfg.data.save_dir / "predictions.csv", index=False)
    feature_imp.to_csv(cfg.data.save_dir / "feature_importance.csv", index=False)

    # Save test predictions for analysis
    test_meta = dataset.get_metadata("test")
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)

    test_results = test_meta.copy()
    test_results["predicted_rating"] = dataset.decode_labels(test_preds)
    test_results["confidence"] = test_probs.max(axis=1)
    test_results["correct"] = (
        test_results["target_rating"] == test_results["predicted_rating"]
    )

    test_results.to_csv(cfg.data.save_dir / "test_predictions.csv", index=False)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model saved to: {cfg.data.save_dir}")
    print(f"Plots saved to: {cfg.data.plots_dir}")
    print(f"\nTest Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1 Score: {test_metrics['f1']:.4f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    set_seed(42)
    args = get_args()

    data_cfg = DataConfig.from_args(args)
    lstm_cfg = LSTMConfig.from_args(args)
    llm_cfg = LLMConfig.from_args(args)
    xgb_cfg = XGBConfig.from_args(args)
    config = Config(
        data=data_cfg,
        lstm=lstm_cfg,
        llm=llm_cfg,
        xgb=xgb_cfg,
    )

    train(config)
