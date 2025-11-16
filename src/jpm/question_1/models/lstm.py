from __future__ import annotations

from typing import Dict

import numpy as np
import tensorflow as tf

from src.jpm.question_1.config import Config, ModelConfig
from src.jpm.question_1.data.ed import EdgarDataLoader
from src.jpm.question_1.models.losses import EnforceBalance, bs_loss
from src.jpm.question_1.models.metrics import Metric, TickerResults  # adjust path
from src.jpm.question_1.vis import (
    build_equity_rows,
    build_section_rows,
    make_row,
    print_table,
)


class LSTMForecaster:
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
            dtype=tf.float32,
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
        if self.config.training.scheduler == "exponential":
            lr = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.config.training.lr,
                decay_steps=self.config.training.decay_steps,
                decay_rate=self.config.training.decay_rate,
                staircase=True,
            )
        elif self.config.training.scheduler == "cosine":
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
        y_pred = self.predict(
            self.data.val_dataset if stage == "val" else self.data.train_dataset
        )
        ds = self.data.val_dataset if stage == "val" else self.data.train_dataset
        y_gt = np.concatenate([y.numpy() for _, y in ds], axis=0)

        # Unscale and construct full balance sheets
        y_pred_unscaled = y_pred * self.data.target_std + self.data.target_mean
        y_gt_unscaled = y_gt * self.data.target_std + self.data.target_mean

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

        # Mean absolute ground truth magnitude (denominator for % error)
        mean_assets_gt = np.mean(np.abs(assets_gt))
        mean_liabilities_gt = np.mean(np.abs(liabilities_gt))
        mean_equity_gt = np.mean(np.abs(equity_gt))

        # Percentage errors
        pct_assets = mae_assets / (np.abs(mean_assets_gt) + 1e-9)
        pct_liabilities = mae_liabilities / (mean_liabilities_gt + 1e-9)
        pct_equity = mae_equity / (mean_equity_gt + 1e-9)

        # One value oscillates around zero, causing spikes in pct error
        eps = 1e-6
        abs_err = np.abs(y_pred_unscaled - y_gt_unscaled)
        per_feature_mae = np.mean(abs_err, axis=0)
        per_feature_mean_gt = np.mean(np.abs(y_gt_unscaled), axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            per_feature_pct_error = per_feature_mae / per_feature_mean_gt
            per_feature_pct_error = np.where(
                per_feature_mean_gt > eps,
                per_feature_pct_error,
                0.0,  # would NaN or 0 be better?
            )

        feature_metrics: Dict[str, Metric] = {
            name: Metric(
                value=float(y_pred_unscaled[:, idx].mean()),
                mae=float(per_feature_mae[idx]),
                pct=float(per_feature_pct_error[idx]),
            )
            for name, idx in self.data.feat_to_idx.items()
        }

        ticker_results = TickerResults(
            assets=Metric(
                value=float(assets_pred.mean()),
                mae=float(mae_assets),
                pct=float(pct_assets),
            ),
            liabilities=Metric(
                value=float(liabilities_pred.mean()),
                mae=float(mae_liabilities),
                pct=float(pct_liabilities),
            ),
            equity=Metric(
                value=float(equity_pred.mean()),
                mae=float(mae_equity),
                pct=float(pct_equity),
            ),
            features=feature_metrics,
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
        # t_res: TickerResults = results[ticker]

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

        print()


if __name__ == "__main__":
    from src.jpm.question_1.config import (
        Config,
        DataConfig,
        LossConfig,
        ModelConfig,
        TrainingConfig,
    )
    from src.jpm.question_1.misc import train_args
    from src.jpm.question_1.models.bs import BalanceSheet

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
    print(f"\nIdentity: {bs.check_identity()}")
