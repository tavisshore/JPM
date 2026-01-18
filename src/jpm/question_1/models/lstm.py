from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from jpm.question_1.clients.llm_client import LLMClient
from jpm.question_1.config import Config, LLMConfig, ModelConfig
from jpm.question_1.data import EdgarData, StatementsDataset
from jpm.question_1.misc import RATINGS_MAPPINGS, set_seed
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

RATINGS_MAPPINGS = RATINGS_MAPPINGS

# Keep keras tied to tf.keras so unit tests can patch tf.keras.* as expected
keras = tf.keras

tfpl = tfp.layers
tfd = tfp.distributions


class LSTMForecaster:
    """Wrapper around a Keras LSTM for balance sheet forecasting."""

    def __init__(
        self, config: Config, data: EdgarData, dataset: StatementsDataset
    ) -> None:
        self.config = config
        self.data = data
        self.dataset = dataset
        self.model = self._build_model()
        self._compile_model()

        if not self.config.training.checkpoint_path.exists():
            self.config.training.checkpoint_path.mkdir(parents=True)

    def _build_model(self) -> keras.Model:
        inputs = keras.layers.Input(
            shape=(self.config.data.lookback, self.dataset.num_features),
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
                feature_mappings=self.dataset.feature_mappings,
                feature_means=self.dataset.target_mean,
                feature_stds=self.dataset.target_std,
                feature_names=self.dataset.name_to_target_idx,
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
                feature_means=self.dataset.target_mean,
                feature_stds=self.dataset.target_std,
                feature_mappings=self.dataset.feature_mappings,
                config=self.config.loss,
            )

            keras.Model.compile(
                self.model,
                optimizer=self._build_optimizer(),
                loss=loss_fn,
                metrics=["mae"],
            )
        # self.model.summary()  # Disabled for batch processing

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
            cardinality = int(self.dataset.train_dataset.cardinality().numpy())
        except Exception:
            cardinality = -1
        if cardinality <= 0:
            return 1.0
        return 1.0 / cardinality

    def _build_output_layer(self):
        if not self.config.model.variational:
            return keras.layers.Dense(
                len(self.dataset.tgt_indices), name="next_quarter"
            )

        kl_weight = self._kl_weight()
        make_prior_fn = tfp.layers.default_multivariate_normal_fn
        make_posterior_fn = tfp.layers.default_mean_field_normal_fn()

        return tfp.layers.DenseVariational(
            len(self.dataset.tgt_indices),
            make_prior_fn=make_prior_fn,
            make_posterior_fn=make_posterior_fn,
            kl_weight=kl_weight,
            name="next_quarter",
        )

    def fit(self, **kwargs):
        checkpoint_cb = keras.callbacks.ModelCheckpoint(
            filepath=self.config.training.checkpoint_path
            / "best_model_ckpt.weights.h5",
            monitor="val_loss" if self.dataset.val_dataset is not None else "loss",
            save_best_only=True,
            save_weights_only=True,
        )
        if not hasattr(checkpoint_cb, "_implements_train_batch_hooks"):
            checkpoint_cb._implements_train_batch_hooks = lambda: False

        history = self.model.fit(
            self.dataset.train_dataset,
            validation_data=self.dataset.val_dataset,
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

    def evaluate(
        self, stage: str = "val", llm_config: LLMConfig | None = None
    ) -> TickerResults:
        if stage not in {"val", "train"}:
            raise ValueError("stage must be 'val' or 'train'")

        ds = self.dataset.val_dataset if stage == "val" else self.dataset.train_dataset
        if ds is None:
            raise ValueError(f"{stage} dataset is not available for evaluation")

        history, y_gt = self._collect_batches(ds, stage)
        y_pred, pred_std = self._predict_with_uncertainty(history)

        y_pred_unscaled, y_gt_unscaled, history_unscaled = self._unscale(
            y_pred, y_gt, history
        )

        if llm_config:
            # Add in LLM client forecast here, and then average the two - comparing
            # There's no training involved so only for val
            history_df = pd.DataFrame(
                history_unscaled[0],
                columns=self.data.targets,
                index=self._prediction_index(
                    stage, history_unscaled[0].shape[0], for_history=True
                ),
                copy=False,
            )

            y_pred_unscaled_df = pd.DataFrame(
                y_pred_unscaled,
                columns=self.data.targets,
                index=self._prediction_index(stage, y_pred_unscaled.shape[0]),
                copy=False,
            )

            llm_client = LLMClient()

            if llm_config.adjust:
                llm_estimation = llm_client.forecast_next_quarter(
                    history=history_df, prediction=y_pred_unscaled_df, cfg=llm_config
                )
            else:
                llm_estimation = llm_client.forecast_next_quarter(
                    history=history_df, cfg=llm_config
                )

            y_pred_unscaled = y_pred_unscaled_df.apply(pd.to_numeric, errors="coerce")
            llm_estimation = llm_estimation.apply(pd.to_numeric, errors="coerce")

            # Ensure both indices are in pandas datetime format
            y_pred_unscaled.index = pd.to_datetime(y_pred_unscaled.index)
            llm_estimation.index = pd.to_datetime(llm_estimation.index)

            # # If not adjusting - avg the predictions
            if llm_config.adjust:
                y_pred_unscaled = llm_estimation.values
            else:
                y_pred_unscaled = (y_pred_unscaled.add(llm_estimation)).div(2).values

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
            y_pred = self.model.predict(history, verbose=0)
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
        y_pred_unscaled = y_pred * self.dataset.target_std + self.dataset.target_mean
        y_gt_unscaled = y_gt * self.dataset.target_std + self.dataset.target_mean
        history_unscaled = history * self.dataset.target_std + self.dataset.target_mean
        return y_pred_unscaled, y_gt_unscaled, history_unscaled

    def _prediction_index(self, stage: str, count: int, for_history: bool = False):
        if count <= 0 or not hasattr(self.data, "_get_timestamp_index"):
            return None

        try:
            ts_index = self.data._get_timestamp_index()
        except Exception:
            return None

        lookback = self.config.data.lookback
        horizon = self.config.data.horizon
        max_start = len(ts_index) - lookback - horizon + 1
        if max_start <= 0:
            return None

        first_window_start = (
            max_start - self.config.data.withhold_periods if stage == "val" else 0
        )
        if first_window_start < 0 or first_window_start > max_start:
            return None

        if for_history:
            start_pos = first_window_start
            end_pos = start_pos + count
        else:
            start_pos = first_window_start + lookback + horizon - 1
            end_pos = start_pos + count

        if start_pos < 0 or end_pos > len(ts_index):
            return None
        return ts_index[start_pos:end_pos]

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
            for name, idx in self.dataset.feat_to_idx.items()
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
        asset_idx = self.dataset.feature_mappings["assets"]
        liability_idx = self.dataset.feature_mappings["liabilities"]
        equity_idx = self.dataset.feature_mappings["equity"]

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
                for name, idx in self.dataset.feat_to_idx.items()
            },
        )

    def _net_income_baselines(
        self,
        y_pred_unscaled: np.ndarray,
        y_gt_unscaled: np.ndarray,
        history_unscaled: np.ndarray,
    ) -> dict[str, dict[str, float] | float]:
        net_income_baseline_mae: dict[str, float] = {}
        net_income_skill: dict[str, float] = {}
        net_income_model_mae = 0.0
        net_income_pred = 0.0
        net_income_gt = 0.0
        net_income_baseline_pred: dict[str, float] = {}
        net_income_key = "Net Income"

        ni_idx = self.dataset.feat_to_idx[net_income_key]
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
    from jpm.question_1.models.balance_sheet import BalanceSheet
    from jpm.question_1.models.income_statement import IncomeStatement
    from src.jpm.question_1.config import (
        Config,
        DataConfig,
        LossConfig,
        ModelConfig,
        TrainingConfig,
    )
    from src.jpm.question_1.misc import set_seed, train_args

    set_seed(42)
    args = train_args()

    data_cfg = DataConfig.from_args(args)
    model_cfg = ModelConfig.from_args(args)
    train_cfg = TrainingConfig.from_args(args)
    loss_cfg = LossConfig.from_args(args)

    config = Config(data=data_cfg, model=model_cfg, training=train_cfg, loss=loss_cfg)

    data = EdgarData(config=config)

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
