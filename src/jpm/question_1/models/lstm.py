from __future__ import annotations

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

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

# Keep keras tied to tf.keras so unit tests can patch tf.keras.* as expected
keras = tf.keras

tfpl = tfp.layers
tfd = tfp.distributions


class LSTMForecaster:
    """Wrapper around a Keras LSTM for balance sheet forecasting."""

    def __init__(self, config: Config, data: EdgarDataLoader) -> None:
        self.config = config
        self.data = data
        self.model = self._build_model()
        self._compile_model()

        if not self.config.training.checkpoint_path.exists():
            self.config.training.checkpoint_path.mkdir(parents=True)

    def _build_model(self) -> keras.Model:
        inputs = keras.layers.Input(
            shape=(self.config.data.lookback, self.data.num_features),
            dtype="float32",
            name="inputs",
        )

        x = inputs
        for i in range(self.config.model.lstm_layers):
            return_sequences = i < self.config.model.lstm_layers - 1
            x = keras.layers.LSTM(
                self.config.model.lstm_units,
                return_sequences=return_sequences,
                name=f"lstm_{i + 1}",
            )(x)

        if self.config.model.dropout > 0:
            x = keras.layers.Dropout(self.config.model.dropout, name="dropout")(x)

        if self.config.model.dense_units > 0:
            x = keras.layers.Dense(
                self.config.model.dense_units,
                activation="relu",
                name="dense",
            )(x)

        if self.config.model.probabilistic:
            n = len(self.data.targets)
            params_size = tfpl.MultivariateNormalTriL.params_size(n)

            params = keras.layers.Dense(
                params_size,
                name="params",
            )(x)

            outputs = tfpl.MultivariateNormalTriL(
                event_size=n,
                name="next_quarter",
            )(params)
        else:
            outputs = self._build_output_layer()(x)

        # EnforceBalance is a post-processing constraint layer
        # Not for probabilistic outputs
        if self.config.loss.enforce_balance:
            if self.config.model.probabilistic:
                raise ValueError(
                    "enforce_balance + probabilistic=True is not currently supported. "
                )

            outputs = EnforceBalance(
                feature_mappings=self.data.feature_mappings,
                feature_means=self.data.target_mean,
                feature_stds=self.data.target_std,
                slack_name="accumulated_other_comprehensive_income_loss_net_of_tax",
                feature_names=self.data.bs_keys,
            )(outputs)

        model = keras.Model(inputs=inputs, outputs=outputs, name="lstm_model")
        return model

    def _compile_model(self):
        if self.config.model.probabilistic:

            def nll(y_true, y_pred_dist):
                return -tf.reduce_mean(y_pred_dist.log_prob(y_true))

            keras.Model.compile(
                self.model,
                optimizer="adam",
                loss=nll,
                metrics=["mae"],
            )
        else:
            loss_fn = bs_loss(
                feature_means=self.data.target_mean,
                feature_stds=self.data.target_std,
                feature_mappings=self.data.feature_mappings,
                config=self.config.loss,
            )

            keras.Model.compile(
                self.model,
                optimizer=self._build_optimizer(),
                loss=loss_fn,
                metrics=["mae"],
            )
        self.model.summary()

    def _build_optimizer(self):
        lr = self.config.training.lr
        if self.config.training.scheduler == "cosine":
            lr = keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=self.config.training.lr,
                first_decay_steps=self.config.training.decay_steps,
            )
        return keras.optimizers.Adam(learning_rate=lr)

    def _kl_weight(self) -> float:
        """Approximate KL weight based on dataset cardinality."""
        try:
            cardinality = int(self.data.train_dataset.cardinality().numpy())
        except Exception:
            cardinality = -1
        if cardinality <= 0:
            return 1.0
        return 1.0 / cardinality

    def _build_output_layer(self):
        if not self.config.model.variational:
            return keras.layers.Dense(len(self.data.targets), name="next_quarter")

        kl_weight = self._kl_weight()
        make_prior_fn = tfp.layers.default_multivariate_normal_fn
        make_posterior_fn = tfp.layers.default_mean_field_normal_fn()

        return tfp.layers.DenseVariational(
            len(self.data.targets),
            make_prior_fn=make_prior_fn,
            make_posterior_fn=make_posterior_fn,
            kl_weight=kl_weight,
            name="next_quarter",
        )

    def fit(self, **kwargs):
        checkpoint_cb = keras.callbacks.ModelCheckpoint(
            filepath=self.config.training.checkpoint_path
            / "best_model_ckpt.weights.h5",
            monitor="val_loss" if self.data.val_dataset is not None else "loss",
            save_best_only=True,
            save_weights_only=True,
        )
        if not hasattr(checkpoint_cb, "_implements_train_batch_hooks"):
            checkpoint_cb._implements_train_batch_hooks = lambda: False

        history = self.model.fit(
            self.data.train_dataset,
            validation_data=self.data.val_dataset,
            epochs=self.config.training.epochs,
            callbacks=[checkpoint_cb],
            **kwargs,
        )
        weights_path = (
            self.config.training.checkpoint_path / "best_model_ckpt.weights.h5"
        )
        if weights_path.exists():
            self.model.load_weights(weights_path)
        return history

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
        if ds is None:
            raise ValueError(f"{stage} dataset is not available for evaluation")

        history, y_gt = self._collect_batches(ds, stage)
        y_pred, pred_std = self._predict_with_uncertainty(history)

        y_pred_unscaled, y_gt_unscaled, history_unscaled = self._unscale(
            y_pred, y_gt, history
        )

        feature_metrics, per_feature_std = self._compute_feature_metrics(
            y_pred_unscaled, y_gt_unscaled, pred_std
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

    def _predict_with_uncertainty(
        self, history: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None]:
        pred_std = None
        if self.config.model.probabilistic:
            y_pred, pred_std = self._predict_probabilistic(history)
        elif self.config.model.variational and self.config.model.mc_samples > 1:
            y_pred, pred_std = self._predict_variational(history)
        else:
            y_pred = self.model.predict(history)
        return y_pred, pred_std

    def _predict_probabilistic(
        self, history: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.config.model.mc_samples > 1:
            dists = [
                self.model(history, training=False)
                for _ in range(self.config.model.mc_samples)
            ]
            means = np.stack([d.mean().numpy() for d in dists], axis=0)
            stds = np.stack([d.stddev().numpy() for d in dists], axis=0)
            return np.mean(means, axis=0), np.mean(stds, axis=0)

        dist = self.model(history, training=False)
        return dist.mean().numpy(), dist.stddev().numpy()

    def _predict_variational(
        self, history: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        samples = [
            self.model.predict(history, verbose=0)
            for _ in range(self.config.model.mc_samples)
        ]
        return np.mean(samples, axis=0), np.std(samples, axis=0)

    def _unscale(
        self, y_pred: np.ndarray, y_gt: np.ndarray, history: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y_pred_unscaled = y_pred * self.data.target_std + self.data.target_mean
        y_gt_unscaled = y_gt * self.data.target_std + self.data.target_mean
        history_unscaled = history * self.data.target_std + self.data.target_mean
        return y_pred_unscaled, y_gt_unscaled, history_unscaled

    def _compute_feature_metrics(
        self,
        y_pred_unscaled: np.ndarray,
        y_gt_unscaled: np.ndarray,
        pred_std: np.ndarray | None,
    ) -> tuple[dict[str, Metric], np.ndarray]:
        per_feature_mae, per_feature_std = self._per_feature_errors(
            y_pred_unscaled, y_gt_unscaled, pred_std
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
        pred_std: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        err = y_pred_unscaled - y_gt_unscaled
        abs_err = np.abs(err)
        per_feature_mae = np.mean(abs_err, axis=0)
        per_feature_std = np.zeros_like(per_feature_mae)
        if pred_std is not None:
            per_feature_std = np.mean(pred_std, axis=0)
        return per_feature_mae, per_feature_std

    def _build_results(
        self,
        y_pred_unscaled: np.ndarray,
        y_gt_unscaled: np.ndarray,
        feature_metrics: dict[str, Metric],
        per_feature_std: np.ndarray,
        history_unscaled: np.ndarray,
    ) -> TickerResults:
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

        net_income_results = self._net_income_baselines(y_pred_unscaled, y_gt_unscaled)

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
        self, y_pred_unscaled: np.ndarray, y_gt_unscaled: np.ndarray
    ) -> dict[str, dict[str, float] | float]:
        net_income_baseline_mae: dict[str, float] = {}
        net_income_skill: dict[str, float] = {}
        net_income_model_mae = 0.0
        net_income_pred = 0.0
        net_income_gt = 0.0
        net_income_baseline_pred: dict[str, float] = {}
        net_income_key = "net_income_loss"
        if net_income_key not in self.data.feat_to_idx:
            return {
                "baseline_mae": net_income_baseline_mae,
                "skill": net_income_skill,
                "model_mae": net_income_model_mae,
                "pred": net_income_pred,
                "gt": net_income_gt,
                "baseline_pred": net_income_baseline_pred,
            }

        ni_idx = self.data.feat_to_idx[net_income_key]
        net_income_pred = float(y_pred_unscaled[:, ni_idx].mean())
        net_income_gt = float(y_gt_unscaled[:, ni_idx].mean())
        net_income_model_mae = float(
            np.mean(np.abs(y_pred_unscaled[:, ni_idx] - y_gt_unscaled[:, ni_idx]))
        )

        baselines_pred = compute_baseline_predictions(
            history=y_gt_unscaled * 0 + y_gt_unscaled,  # reuse history shape
            seasonal_lag=min(4, y_gt_unscaled.shape[1]),
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
        results = self.val_results if stage == "val" else self.train_results
        ticker = self.config.data.ticker

        print(f"\033[1mResults for {stage} dataset ({ticker}):\033[0m")

        # Summary Table
        overall_rows = [
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

    model.view_results(stage="val")

    # Pass outputs to BS Model
    bs = BalanceSheet(config=config, results=validation_results)
    bs.check_identity()

    # Income Statement to predict Net Income (Loss)
    i_s = IncomeStatement(config=config, results=validation_results)
    i_s.view()
