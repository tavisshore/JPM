from pathlib import Path

from jpm.question_1 import CreditDataset, CreditRatingModel


def main():
    DATA_DIR = "/scratch/datasets/jpm/ratings"
    MODEL_DIR = "/scratch/projects/JPM/temp"
    PLOTS_DIR = Path(MODEL_DIR) / "plots"

    PARAMS = {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "early_stopping_rounds": 20,
        "use_gpu": True,  # Set to False if no GPU
        "random_state": 42,
    }
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CREDIT RATING PREDICTION - XGBoost Training")
    print("=" * 70)
    print("\n[1/5] Loading dataset...")
    dataset = CreditDataset(
        data_dir=DATA_DIR,
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
        n_classes=info["n_classes"], n_features=info["n_features"], **PARAMS
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
    model.plot_training_history(save_path=PLOTS_DIR / "training_history.png")
    model.plot_confusion_matrix(
        y_test,
        model.predict(X_test),
        class_names=info["classes"],
        save_path=PLOTS_DIR / "confusion_matrix.png",
    )
    model.plot_confusion_matrix(
        y_test,
        model.predict(X_test),
        class_names=info["classes"],
        normalize=True,
        save_path=PLOTS_DIR / "confusion_matrix_normalized.png",
    )
    model.plot_feature_importance(
        top_n=20, save_path=PLOTS_DIR / "feature_importance.png"
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
    model.save(MODEL_DIR)
    results.to_csv(Path(MODEL_DIR) / "predictions.csv", index=False)
    feature_imp.to_csv(Path(MODEL_DIR) / "feature_importance.csv", index=False)

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

    test_results.to_csv(Path(MODEL_DIR) / "test_predictions.csv", index=False)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model saved to: {MODEL_DIR}")
    print(f"Plots saved to: {PLOTS_DIR}")
    print(f"\nTest Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1 Score: {test_metrics['f1']:.4f}")
    if test_metrics["auc"]:
        print(f"Test AUC:      {test_metrics['auc']:.4f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
