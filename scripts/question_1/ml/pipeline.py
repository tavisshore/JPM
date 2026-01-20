"""
End-to-End Pipeline for Credit Rating Prediction.

This comprehensive pipeline integrates multiple components for credit rating analysis:

Workflow:
1. Train XGBoost model on financial ratios to predict credit ratings
2. Extract financial ratios from annual reports using LLM parsing
3. Predict credit ratings from extracted ratios
4. Optionally integrate with LSTM balance sheet forecasts

Components:
- XGBoost classifier trained on historical ratio-to-rating mappings
- LLM-based annual report parser for ratio extraction
- Feature engineering and model evaluation
- Prediction confidence scoring and analysis

The pipeline supports both historical model training and real-time report analysis
for credit risk assessment.
"""

import numpy as np

from jpm.config import Config, DataConfig, LLMConfig, LSTMConfig, XGBConfig, get_args
from jpm.question_1 import CreditDataset, CreditRatingModel, set_seed
from jpm.question_1.clients.llm_client import LLMClient


def train_xgb(cfg: Config, feature_names: list = None):
    print("\n[1/5] Loading dataset...")
    dataset = CreditDataset(
        data_dir=cfg.data.cache_dir,
        pattern="*_ratings.parquet",
        val_size=0.15,
        test_size=0.15,
        random_state=42,
        verbose=True,
    )
    dataset.load(feature_names=feature_names)

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

    return model


if __name__ == "__main__":
    set_seed(42)
    args = get_args()

    # Setup Configs
    data_cfg = DataConfig.from_args(args)
    lstm_cfg = LSTMConfig.from_args(args)
    llm_cfg = LLMConfig(
        provider="openai",
        model="gpt-4o-2024-08-06",
    )
    xgb_cfg = XGBConfig.from_args(args)
    config = Config(
        data=data_cfg,
        lstm=lstm_cfg,
        llm=llm_cfg,
        xgb=xgb_cfg,
    )

    # Feature names used across training and inference
    feature_cols = [
        "quick_ratio",
        "debt_to_equity",
        "debt_to_assets",
        "debt_to_capital",
        "debt_to_ebitda",
    ]

    # Train Ratios -> Credit Rating XGBoost model
    xgb_model = train_xgb(config, feature_names=feature_cols)

    # Parse Report -> Ratios -> Credit Rating
    client = LLMClient()
    report_path = config.data.get_report_path()
    report_pages = config.data.get_report_pages()

    if report_path and report_pages:
        data = client.parse_annual_report(config)

        if data:
            # Convert to feature array in correct order
            X_report = np.array([[data.get(col, 0.0) for col in feature_cols]]).astype(
                np.float32
            )

            # Predict credit rating from report ratios
            prediction = xgb_model.predict(X_report)
            probabilities = xgb_model.predict_proba(X_report)

            # Decode prediction
            class_names = {0: "Prime", 1: "High", 2: "Medium", 3: "Low"}
            predicted_rating = class_names.get(prediction[0], "Unknown")
            confidence = probabilities[0].max()

            print("\n" + "=" * 70)
            print("REPORT ANALYSIS COMPLETE")
            print("=" * 70)
            print(f"Ticker: {config.data.ticker}")
            print(f"Report Date: {data.get('report_date', 'N/A')}")
            print("\nExtracted Ratios:")
            for col in feature_cols:
                print(f"  {col}: {data.get(col, 'N/A')}")
            print(f"\nPredicted Credit Rating: {predicted_rating}")
            print(f"Confidence: {confidence:.2%}")
            print("=" * 70 + "\n")
