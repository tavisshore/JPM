from __future__ import annotations

import numpy as np
import xgboost as xgb

from jpm.question_1.config import Config
from jpm.question_1.data.ed import EdgarData
from jpm.question_1.models.metrics import (
    Metric,
    TickerResults,
    baseline_skill_scores,
    compute_baseline_predictions,
)
from jpm.question_1.vis import (
    build_baseline_rows,
    build_equity_rows,
    build_section_rows,
    make_row,
    print_table,
)


class XGBoostForecaster:
    """Wrapper around XGBoost for balance sheet forecasting."""

    def __init__(self, config: Config, data: EdgarData) -> None:
        self.config = config
        self.data = data
        self.model = None
        self.train_results = None
        self.val_results = None

        if not self.config.training.checkpoint_path.exists():
            self.config.training.checkpoint_path.mkdir(parents=True)

    def _prepare_data(self, ds) -> tuple[np.ndarray, np.ndarray]:
        """Convert TensorFlow dataset to numpy arrays for XGBoost."""
        x_batches = []
        y_batches = []
        for x_batch, y_batch in ds:
            x_batches.append(x_batch.numpy())
            y_batches.append(y_batch.numpy())

        if not x_batches:
            raise ValueError("No batches found in dataset")

        x = np.concatenate(x_batches, axis=0)
        y = np.concatenate(y_batches, axis=0)

        # Reshape x: (samples, timesteps, features) → (samples, timesteps*features)
        # XGBoost expects 2D input
        x_reshaped = x.reshape(x.shape[0], -1)

        return x_reshaped, y

    def fit(self, **kwargs):
        """Train the XGBoost model."""
        # Prepare training data
        x_train, y_train = self._prepare_data(self.data.train_dataset)

        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(x_train, label=y_train)

        # Prepare validation data if available
        eval_list = [(dtrain, "train")]
        if self.data.val_dataset is not None:
            x_val, y_val = self._prepare_data(self.data.val_dataset)
            dval = xgb.DMatrix(x_val, label=y_val)
            eval_list.append((dval, "val"))

        # XGBoost parameters
        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "max_depth": kwargs.get("max_depth", 6),
            "learning_rate": self.config.training.lr,
            "n_estimators": kwargs.get("n_estimators", 100),
            "subsample": kwargs.get("subsample", 0.8),
            "colsample_bytree": kwargs.get("colsample_bytree", 0.8),
            "verbosity": kwargs.get("verbose", 0),
        }

        # Train model
        evals_result = {}
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.config.training.epochs,
            evals=eval_list,
            evals_result=evals_result,
            verbose_eval=kwargs.get("verbose", 0) > 0,
        )

        # Save model
        model_path = self.config.training.checkpoint_path / "best_model.json"
        self.model.save_model(str(model_path))

        # Return history-like object for compatibility
        class History:
            def __init__(self, results):
                self.history = {
                    "loss": results.get("train", {}).get("rmse", []),
                    "val_loss": results.get("val", {}).get("rmse", []),
                }

        return History(evals_result)

    def predict(self, x):
        """Make predictions with the XGBoost model."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Reshape if necessary
        if len(x.shape) == 3:
            x = x.reshape(x.shape[0], -1)

        dmatrix = xgb.DMatrix(x)
        return self.model.predict(dmatrix)

    def save(self, path: str):
        """Save the model to disk."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        self.model.save_model(path)

    @classmethod
    def load(cls, path: str, config: Config, data: EdgarData) -> "XGBoostForecaster":
        """Load a saved model from disk."""
        obj = cls(config=config, data=data)
        obj.model = xgb.Booster()
        obj.model.load_model(path)
        return obj

    def evaluate(self, stage: str = "val") -> TickerResults:
        """Evaluate the model on train or validation data."""
        if stage not in {"val", "train"}:
            raise ValueError("stage must be 'val' or 'train'")

        ds = self.data.val_dataset if stage == "val" else self.data.train_dataset
        if ds is None:
            raise ValueError(f"{stage} dataset is not available for evaluation")

        history, y_gt = self._collect_batches(ds, stage)
        y_pred = self._predict_with_uncertainty(history)

        y_pred_unscaled, y_gt_unscaled, history_unscaled = self._unscale(
            y_pred, y_gt, history
        )

        feature_metrics, per_feature_std = self._compute_feature_metrics(
            y_pred_unscaled, y_gt_unscaled
        )

        ticker_results = self._build_results(
            y_pred_unscaled,
            y_gt_unscaled,
            feature_metrics,
            per_feature_std,
            history_unscaled,
        )

        if stage == "val":
            self.val_results = ticker_results
        else:
            self.train_results = ticker_results
        return ticker_results

    def _collect_batches(self, ds, stage: str) -> tuple[np.ndarray, np.ndarray]:
        """Collect batches from dataset into numpy arrays."""
        x_batches = []
        y_batches = []
        for x_batch, y_batch in ds:
            x_batches.append(x_batch.numpy())
            y_batches.append(y_batch.numpy())
        if not x_batches:
            raise ValueError(f"No batches found when evaluating {stage} dataset")
        history = np.concatenate(x_batches, axis=0)
        y_gt = np.concatenate(y_batches, axis=0)
        return history, y_gt

    def _predict_with_uncertainty(self, history: np.ndarray) -> np.ndarray:
        """Make predictions (no uncertainty for XGBoost by default)."""
        return self.predict(history)

    def _unscale(
        self, y_pred: np.ndarray, y_gt: np.ndarray, history: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unscale predictions and targets back to original scale."""
        y_pred_unscaled = y_pred * self.data.target_std + self.data.target_mean
        y_gt_unscaled = y_gt * self.data.target_std + self.data.target_mean
        history_unscaled = history * self.data.target_std + self.data.target_mean
        return y_pred_unscaled, y_gt_unscaled, history_unscaled

    def _compute_feature_metrics(
        self,
        y_pred_unscaled: np.ndarray,
        y_gt_unscaled: np.ndarray,
    ) -> tuple[dict[str, Metric], np.ndarray]:
        """Compute per-feature metrics."""
        per_feature_mae, per_feature_std = self._per_feature_errors(
            y_pred_unscaled, y_gt_unscaled
        )

        feature_metrics = {
            name: Metric(
                value=float(y_pred_unscaled[:, idx].mean()),
                mae=float(per_feature_mae[idx]),
                gt=float(y_gt_unscaled[:, idx].mean()),
                std=float(per_feature_std[idx]),
            )
            for name, idx in self.data.feat_to_idx.items()
        }
        return feature_metrics, per_feature_std

    def _per_feature_errors(
        self,
        y_pred_unscaled: np.ndarray,
        y_gt_unscaled: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-feature MAE and std."""
        err = y_pred_unscaled - y_gt_unscaled
        abs_err = np.abs(err)
        per_feature_mae = np.mean(abs_err, axis=0)
        per_feature_std = np.std(err, axis=0)
        return per_feature_mae, per_feature_std

    def _build_results(
        self,
        y_pred_unscaled: np.ndarray,
        y_gt_unscaled: np.ndarray,
        feature_metrics: dict[str, Metric],
        per_feature_std: np.ndarray,
        history_unscaled: np.ndarray,
    ) -> TickerResults:
        """Build TickerResults object from predictions and ground truth."""
        asset_idx = self.data.feature_mappings["assets"]
        liability_idx = self.data.feature_mappings["liabilities"]
        equity_idx = self.data.feature_mappings["equity"]

        assets_pred = np.sum(y_pred_unscaled[:, asset_idx], axis=-1)
        liabilities_pred = np.sum(y_pred_unscaled[:, liability_idx], axis=-1)
        equity_pred = np.sum(y_pred_unscaled[:, equity_idx], axis=-1)

        assets_gt = np.sum(y_gt_unscaled[:, asset_idx], axis=-1)
        liabilities_gt = np.sum(y_gt_unscaled[:, liability_idx], axis=-1)
        equity_gt = np.sum(y_gt_unscaled[:, equity_idx], axis=-1)

        mae_assets = np.mean(np.abs(assets_pred - assets_gt))
        mae_liabilities = np.mean(np.abs(liabilities_pred - liabilities_gt))
        mae_equity = np.mean(np.abs(equity_pred - equity_gt))

        baseline_results = baseline_skill_scores(
            y_true=y_gt_unscaled,
            model_pred=y_pred_unscaled,
            history=history_unscaled,
            seasonal_lag=min(4, history_unscaled.shape[1]),
        )

        net_income_results = self._net_income_baselines(
            y_pred_unscaled, y_gt_unscaled, history_unscaled
        )

        return TickerResults(
            assets=Metric(
                value=float(assets_pred.mean()),
                mae=float(mae_assets),
                gt=float(assets_gt.mean()),
            ),
            liabilities=Metric(
                value=float(liabilities_pred.mean()),
                mae=float(mae_liabilities),
                gt=float(liabilities_gt.mean()),
            ),
            equity=Metric(
                value=float(equity_pred.mean()),
                mae=float(mae_equity),
                gt=float(equity_gt.mean()),
            ),
            features=feature_metrics,
            model_mae=baseline_results["model_mae"],
            baseline_mae=baseline_results["baseline_mae"],
            skill=baseline_results["skill"],
            net_income_model_mae=net_income_results["model_mae"],
            net_income_baseline_mae=net_income_results["baseline_mae"],
            net_income_skill=net_income_results["skill"],
            net_income_pred=net_income_results["pred"],
            net_income_gt=net_income_results["gt"],
            net_income_baseline_pred=net_income_results["baseline_pred"],
            pred_std={
                name: float(per_feature_std[idx])
                for name, idx in self.data.feat_to_idx.items()
            },
        )

    def _net_income_baselines(
        self,
        y_pred_unscaled: np.ndarray,
        y_gt_unscaled: np.ndarray,
        history_unscaled: np.ndarray,
    ) -> dict[str, dict[str, float] | float]:
        """Compute net income baselines and metrics."""
        net_income_baseline_mae: dict[str, float] = {}
        net_income_skill: dict[str, float] = {}
        net_income_model_mae = 0.0
        net_income_pred = 0.0
        net_income_gt = 0.0
        net_income_baseline_pred: dict[str, float] = {}
        net_income_key = "Net Income (Loss)"

        ni_idx = self.data.feat_to_idx[net_income_key]
        net_income_pred = float(y_pred_unscaled[:, ni_idx].mean())
        net_income_gt = float(y_gt_unscaled[:, ni_idx].mean())
        net_income_model_mae = float(
            np.mean(np.abs(y_pred_unscaled[:, ni_idx] - y_gt_unscaled[:, ni_idx]))
        )

        baselines_pred = compute_baseline_predictions(
            history=history_unscaled,
            seasonal_lag=min(4, history_unscaled.shape[1]),
        )

        eps = 1e-12
        for name, pred in baselines_pred.items():
            mae = float(np.mean(np.abs(pred[:, ni_idx] - y_gt_unscaled[:, ni_idx])))
            net_income_baseline_mae[name] = mae
            denom = mae if mae > eps else eps
            net_income_skill[name] = 1.0 - net_income_model_mae / denom
            net_income_baseline_pred[name] = float(pred[:, ni_idx].mean())

        return {
            "baseline_mae": net_income_baseline_mae,
            "skill": net_income_skill,
            "model_mae": net_income_model_mae,
            "pred": net_income_pred,
            "gt": net_income_gt,
            "baseline_pred": net_income_baseline_pred,
        }

    def view_results(self, stage: str = "val") -> None:
        """Display evaluation results in formatted tables."""
        results = self.val_results if stage == "val" else self.train_results

        print(f"\033[1mResults for {stage} dataset ({self.config.data.ticker}):\033[0m")

        # Summary Table
        overall_rows = [
            make_row("Assets", results.assets),
            make_row("Liabilities", results.liabilities),
            make_row("Equity", results.equity),
        ]
        print_table("Overall", overall_rows)

        # Detailed per-feature tables
        assets_rows = build_section_rows(
            self.data.bs_structure["Assets"], results.features
        )
        print_table("Assets", assets_rows)

        liabilities_rows = build_section_rows(
            self.data.bs_structure["Liabilities"], results.features
        )
        print_table("Liabilities", liabilities_rows)

        equity_rows = build_equity_rows(
            self.data.bs_structure["Equity"], results.features
        )
        print_table("Equity", equity_rows)

        if results.baseline_mae:
            baseline_rows = build_baseline_rows(
                results.baseline_mae, results.skill, results.model_mae
            )
            print_table(
                "Baseline Comparison (Balance Sheet)",
                baseline_rows,
                headers=("Method", "MAE", "Error diff"),
            )

        print()


if __name__ == "__main__":
    from jpm.question_1 import (
        BalanceSheet,
        Config,
        DataConfig,
        EdgarData,
        IncomeStatement,
        LossConfig,
        ModelConfig,
        TrainingConfig,
        get_args,
        set_seed,
    )

    set_seed(42)
    args = get_args()

    data_cfg = DataConfig.from_args(args)
    model_cfg = ModelConfig.from_args(args)
    train_cfg = TrainingConfig.from_args(args)
    loss_cfg = LossConfig.from_args(args)

    config = Config(data=data_cfg, model=model_cfg, training=train_cfg, loss=loss_cfg)

    data = EdgarData(config=config)

    model = XGBoostForecaster(config=config, data=data)
    model.fit(verbose=1)

    model.evaluate(stage="train")
    validation_results = model.evaluate(stage="val")

    model.view_results(stage="val")

    # Pass outputs to BS Model
    bs = BalanceSheet(config=config, data=data, results=validation_results)
    bs.check_identity()

    # Income Statement to predict Net Income (Loss)
    i_s = IncomeStatement(config=config, data=data, results=validation_results)
    i_s.view()
