from __future__ import annotations

from typing import Dict

import numpy as np
import tensorflow as tf

from jpm.question_1.config import Config, ModelConfig
from jpm.question_1.data.ed import EdgarDataLoader
from jpm.question_1.models.losses import EnforceBalance, bs_loss
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


class LSTMForecaster:
    """Wrapper around a Keras LSTM for balance sheet forecasting."""

    def __init__(self, config: Config, data: EdgarDataLoader) -> None:
        self.config = config
        self.data = data
        self.model = self._build_model()
        self._compile_model()

        if not self.config.training.checkpoint_path.exists():
            self.config.training.checkpoint_path.mkdir(parents=True)

    def _build_model(self) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(
            shape=(self.config.data.lookback, self.data.num_features),
            dtype=tf.float64,
            name="inputs",
        )

        x = inputs
        for i in range(self.config.model.lstm_layers):
            return_sequences = i < self.config.model.lstm_layers - 1
            x = tf.keras.layers.LSTM(
                self.config.model.lstm_units,
                return_sequences=return_sequences,
                name=f"lstm_{i + 1}",
            )(x)

        if self.config.model.dropout > 0:
            x = tf.keras.layers.Dropout(self.config.model.dropout)(x)

        if self.config.model.dense_units is not None:
            x = tf.keras.layers.Dense(self.config.model.dense_units, activation="relu")(
                x
            )

        outputs = tf.keras.layers.Dense(len(self.data.targets), name="next_quarter")(x)

        if self.config.loss.enforce_balance:
            outputs = EnforceBalance(
                feature_mappings=self.data.feature_mappings,
                feature_means=self.data.target_mean,  # np.ndarray shape (F,)
                feature_stds=self.data.target_std,  # np.ndarray shape (F,)
                slack_name="accumulated_other_comprehensive_income_loss_net_of_tax",
                feature_names=self.data.bs_keys,
            )(outputs)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def _compile_model(self):
        loss_fn = bs_loss(
            feature_means=self.data.target_mean,
            feature_stds=self.data.target_std,
            feature_mappings=self.data.feature_mappings,
            config=self.config.loss,
        )

        self.model.compile(
            optimizer=self._build_optimizer(),
            loss=loss_fn,
            metrics=["mae"],
        )
        self.model.summary()

    def _build_optimizer(self):
        lr: float | tf.keras.optimizers.schedules.LearningRateSchedule = (
            self.config.training.lr
        )
        if self.config.training.scheduler == "cosine":
            lr = tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=self.config.training.lr,
                first_decay_steps=self.config.training.decay_steps,
            )
        return tf.keras.optimizers.Adam(learning_rate=lr)

    # Convenience wrappers so calling code doesn't touch raw Keras much
    def fit(self, **kwargs):
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.config.training.checkpoint_path
            / "best_model_ckpt.weights.h5",
            monitor="val_loss" if self.data.val_dataset is not None else "loss",
            save_best_only=True,
            save_weights_only=True,
        )

        return self.model.fit(
            self.data.train_dataset,
            validation_data=self.data.val_dataset,
            epochs=self.config.training.epochs,
            callbacks=[checkpoint_cb],
            **kwargs,
        )

    def predict(self, x):
        return self.model.predict(x)

    def save(self, path: str):
        self.model.save(path)

    @classmethod
    def load(cls, path: str, config: Config) -> "LSTMForecaster":
        obj = cls.__new__(cls)
        obj.config = config
        obj.model = tf.keras.models.load_model(path)
        return obj

    def evaluate(self, stage: str = "val") -> TickerResults:
        if stage not in {"val", "train"}:
            raise ValueError("stage must be 'val' or 'train'")

        ds = self.data.val_dataset if stage == "val" else self.data.train_dataset

        # GET GT to align predictions and baselines
        x_batches: list[np.ndarray] = []
        y_batches: list[np.ndarray] = []
        for x_batch, y_batch in ds:
            x_batches.append(x_batch.numpy())
            y_batches.append(y_batch.numpy())

        history = np.concatenate(x_batches, axis=0)
        y_gt = np.concatenate(y_batches, axis=0)

        y_pred = self.model.predict(history)

        # Unscale and construct full balance sheets
        y_pred_unscaled = y_pred * self.data.target_std + self.data.target_mean
        y_gt_unscaled = y_gt * self.data.target_std + self.data.target_mean
        history_unscaled = history * self.data.target_std + self.data.target_mean

        # Compute MAE on assets, liabilities, equity
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

        err = y_pred_unscaled - y_gt_unscaled
        abs_err = np.abs(err)
        per_feature_mae = np.mean(abs_err, axis=0)

        feature_metrics: Dict[str, Metric] = {
            name: Metric(
                value=float(y_pred_unscaled[:, idx].mean()),
                mae=float(per_feature_mae[idx]),
                gt=float(y_gt_unscaled[:, idx].mean()),
            )
            for name, idx in self.data.feat_to_idx.items()
        }

        baseline_results = baseline_skill_scores(
            y_true=y_gt_unscaled,
            model_pred=y_pred_unscaled,
            history=history_unscaled,
            seasonal_lag=min(4, history_unscaled.shape[1]),
        )

        # Net income specific baseline comparison, if available
        net_income_baseline_mae: Dict[str, float] = {}
        net_income_skill: Dict[str, float] = {}
        net_income_model_mae = 0.0
        net_income_pred = 0.0
        net_income_gt = 0.0
        net_income_baseline_pred: Dict[str, float] = {}
        net_income_key = "net_income_loss"
        if net_income_key in self.data.feat_to_idx:
            ni_idx = self.data.feat_to_idx[net_income_key]
            net_income_pred = float(y_pred_unscaled[:, ni_idx].mean())
            net_income_gt = float(y_gt_unscaled[:, ni_idx].mean())
            net_income_model_mae = float(
                np.mean(np.abs(y_pred_unscaled[:, ni_idx] - y_gt_unscaled[:, ni_idx]))
            )
            baselines_pred = compute_baseline_predictions(
                history_unscaled, seasonal_lag=min(4, history_unscaled.shape[1])
            )
            eps = 1e-12
            for name, pred in baselines_pred.items():
                mae = float(np.mean(np.abs(pred[:, ni_idx] - y_gt_unscaled[:, ni_idx])))
                net_income_baseline_mae[name] = mae
                denom = mae if mae > eps else eps
                net_income_skill[name] = 1.0 - net_income_model_mae / denom
                net_income_baseline_pred[name] = float(pred[:, ni_idx].mean())

        ticker_results = TickerResults(
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
            net_income_model_mae=net_income_model_mae,
            net_income_baseline_mae=net_income_baseline_mae,
            net_income_skill=net_income_skill,
            net_income_pred=net_income_pred,
            net_income_gt=net_income_gt,
            net_income_baseline_pred=net_income_baseline_pred,
        )

        # results_dict: Dict[str, TickerResults] = {
        # self.config.data.ticker: ticker_results
        # }

        if stage == "val":
            self.val_results = ticker_results
        else:
            self.train_results = ticker_results
        return ticker_results

    def view_results(self, stage: str = "val") -> None:
        results: TickerResults = (
            self.val_results if stage == "val" else self.train_results
        )
        ticker = self.config.data.ticker

        print(f"\033[1mResults for {stage} dataset ({ticker}):\033[0m")

        # Summary Table
        overall_rows: list[list[str]] = [
            make_row("Assets", results.assets),
            make_row("Liabilities", results.liabilities),
            make_row("Equity", results.equity),
        ]
        print_table("Overall", overall_rows)

        # Detailed per-feature tables
        assets_rows = build_section_rows(
            self.data.bs_structure["assets"], results.features
        )
        print_table("Assets", assets_rows)

        liabilities_rows = build_section_rows(
            self.data.bs_structure["liabilities"], results.features
        )
        print_table("Liabilities", liabilities_rows)

        equity_rows = build_equity_rows(
            self.data.bs_structure["equity"], results.features
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
    from jpm.question_1.models.balance_sheet import BalanceSheet
    from jpm.question_1.models.income_statement import IncomeStatement
    from src.jpm.question_1.config import (
        Config,
        DataConfig,
        LossConfig,
        ModelConfig,
        TrainingConfig,
    )
    from src.jpm.question_1.misc import train_args

    args = train_args()

    data_cfg = DataConfig.from_args(args)
    model_cfg = ModelConfig.from_args(args)
    train_cfg = TrainingConfig.from_args(args)
    loss_cfg = LossConfig.from_args(args)

    config = Config(data=data_cfg, model=model_cfg, training=train_cfg, loss=loss_cfg)

    data = EdgarDataLoader(config=config)

    model = LSTMForecaster(config=config, data=data)
    model.fit()

    model.evaluate(stage="train")
    validation_results = model.evaluate(stage="val")

    model.view_results(stage="train")
    model.view_results(stage="val")

    # Pass outputs to BS Model
    bs = BalanceSheet(config=config, results=validation_results)
    bs.check_identity()

    # Income Statement to predict Net Income (Loss)
    i_s = IncomeStatement(config=config, results=validation_results)
    i_s.view()
