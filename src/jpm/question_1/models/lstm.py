from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from jpm.config.question_1 import Config, LLMConfig
from jpm.question_1.clients.llm_client import LLMClient
from jpm.question_1.data import EdgarData, StatementsDataset
from jpm.question_1.misc import RATINGS_MAPPINGS, format_money, set_seed
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
    colour,
    colour_mae,
    fmt,
    make_row,
    print_table,
)

RATINGS_MAPPINGS = RATINGS_MAPPINGS

# keras has to stick to tf.keras so pytests can patch tf.keras.* as expected
keras = tf.keras

tfpl = tfp.layers
tfd = tfp.distributions


class MultivariateNormalTriLLayer(keras.layers.Layer):
    """Custom layer for multivariate normal distribution output compatible with Keras 3.

    This layer outputs distribution parameters as a tensor, which can be converted
    to a distribution object during inference.
    """

    def __init__(self, event_size, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.event_size = event_size
        self.params_size = tfpl.MultivariateNormalTriL.params_size(event_size)

        # Create dense layer for distribution parameters
        self.dense = keras.layers.Dense(
            self.params_size, name=f"{name}_params" if name else "params"
        )

    def call(self, inputs, training=None):
        """Forward pass that outputs distribution parameters."""
        return self.dense(inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "event_size": self.event_size,
            }
        )
        return config


class DenseVariationalLayer(keras.layers.Layer):
    """Custom variational dense layer compatible with Keras 3.

    This layer uses variational inference with trainable weight distributions
    instead of point estimates. Compatible with Keras 3 functional API.
    """

    def __init__(self, units, kl_weight=1.0, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.units = units
        self.kl_weight = kl_weight

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.prior_mean_weights = 0.0
        self.prior_std_weights = 1.0
        self.prior_mean_bias = 0.0
        self.prior_std_bias = 1.0

        # Weight distribution parameters
        self.w_mean = self.add_weight(
            name="w_mean",
            shape=(input_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.w_log_std = self.add_weight(
            name="w_log_std",
            shape=(input_dim, self.units),
            initializer=keras.initializers.Constant(-5.0),
            trainable=True,
        )

        # Bias distribution parameters
        self.b_mean = self.add_weight(
            name="b_mean",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )
        self.b_log_std = self.add_weight(
            name="b_log_std",
            shape=(self.units,),
            initializer=keras.initializers.Constant(-5.0),
            trainable=True,
        )

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Forward pass with reparameterization trick."""
        w_std = tf.exp(self.w_log_std)
        b_std = tf.exp(self.b_log_std)

        if training:
            w_noise = tf.random.normal(shape=tf.shape(self.w_mean))
            b_noise = tf.random.normal(shape=tf.shape(self.b_mean))

            w = self.w_mean + w_std * w_noise
            b = self.b_mean + b_std * b_noise

            kl_loss = self._compute_kl_divergence(
                self.w_mean, w_std, self.b_mean, b_std
            )
            self.add_loss(self.kl_weight * kl_loss)
        else:
            w = self.w_mean
            b = self.b_mean

        output = tf.matmul(inputs, w) + b
        return output

    def _compute_kl_divergence(self, w_mean, w_std, b_mean, b_std):
        """Compute KL divergence between posterior and prior."""
        kl_w = 0.5 * tf.reduce_sum(
            tf.square(w_std) + tf.square(w_mean) - 1.0 - 2.0 * tf.math.log(w_std + 1e-8)
        )

        kl_b = 0.5 * tf.reduce_sum(
            tf.square(b_std) + tf.square(b_mean) - 1.0 - 2.0 * tf.math.log(b_std + 1e-8)
        )

        return kl_w + kl_b

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "kl_weight": self.kl_weight,
            }
        )
        return config


class LSTMForecaster:
    """Wrapper around a Keras LSTM for balance sheet forecasting."""

    def __init__(
        self, config: Config, data: EdgarData, dataset: StatementsDataset
    ) -> None:
        """
        Initialise the LSTM forecaster.

        Parameters
        ----------
        config : Config
            Configuration object containing LSTM and data settings.
        data : EdgarData
            Edgar data object containing financial statement data.
        dataset : StatementsDataset
            Dataset object with train/val/test splits and feature mappings.
        """
        self.config = config
        self.data = data
        self.dataset = dataset
        self.model = self._build_model()
        self._compile_model()

        if not self.config.lstm.checkpoint_path.exists():
            self.config.lstm.checkpoint_path.mkdir(parents=True)

    def _build_model(self) -> keras.Model:
        """
        Build the LSTM model architecture.

        Constructs a sequential LSTM model with configurable layers, dropout,
        and output layer type (deterministic, probabilistic, or variational).
        Optionally includes an EnforceBalance constraint layer.

        Returns
        -------
        keras.Model
            Compiled Keras model ready for training.
        """
        inputs = keras.layers.Input(
            shape=(self.config.data.lookback, self.dataset.num_features),
            dtype="float32",
            name="inputs",
        )

        x = inputs
        for i in range(self.config.lstm.lstm_layers):
            return_sequences = i < self.config.lstm.lstm_layers - 1
            x = keras.layers.LSTM(
                self.config.lstm.lstm_units,
                return_sequences=return_sequences,
                name=f"lstm_{i + 1}",
            )(x)

        if self.config.lstm.dropout > 0:
            x = keras.layers.Dropout(self.config.lstm.dropout, name="dropout")(x)

        if self.config.lstm.dense_units > 0:
            x = keras.layers.Dense(
                self.config.lstm.dense_units,
                activation="relu",
                name="dense",
            )(x)

        if self.config.lstm.probabilistic:
            n = len(self.dataset.targets)

            outputs = MultivariateNormalTriLLayer(
                event_size=n,
                name="next_quarter",
            )(x)
        else:
            outputs = self._build_output_layer()(x)

        if self.config.lstm.enforce_balance:
            if self.config.lstm.probabilistic:
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
        """
        Compile the model with loss function and optimiser.

        Configures the model with either negative log likelihood (for
        probabilistic models) or custom balance sheet loss function
        (for deterministic models).
        """
        if self.config.lstm.probabilistic:
            n = len(self.dataset.targets)

            def nll(y_true, y_pred_params):
                """Negative log likelihood loss for probabilistic model."""
                loc = y_pred_params[..., :n]
                scale_params = y_pred_params[..., n:]

                scale_tril = tfp.bijectors.FillScaleTriL(
                    diag_bijector=tfp.bijectors.Softplus(low=1e-3)
                )(scale_params)

                dist = tfd.MultivariateNormalTriL(loc=loc, scale_tril=scale_tril)
                return -tf.reduce_mean(dist.log_prob(y_true))

            keras.Model.compile(
                self.model,
                optimizer="adam",
                loss=nll,
            )
        else:
            loss_fn = bs_loss(
                feature_means=self.dataset.target_mean,
                feature_stds=self.dataset.target_std,
                feature_mappings=self.dataset.feature_mappings,
                config=self.config.lstm,
            )

            keras.Model.compile(
                self.model,
                optimizer=self._build_optimiser(),
                loss=loss_fn,
                metrics=["mae"],
            )

    def _build_optimiser(self):
        """
        Create optimiser with optional learning rate schedule.

        Returns
        -------
        keras.optimizers.Optimizer
            Adam optimiser with constant or cosine decay learning rate.
        """
        lr = self.config.lstm.lr
        if self.config.lstm.scheduler == "cosine":
            lr = keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=self.config.lstm.lr,
                first_decay_steps=self.config.lstm.decay_steps,
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
        """
        Build the output layer (dense or variational).

        Returns
        -------
        keras.layers.Layer
            Either a standard Dense layer or custom DenseVariational layer with
            Bayesian inference capabilities (Keras 3 compatible).
        """
        if not self.config.lstm.variational:
            return keras.layers.Dense(
                len(self.dataset.tgt_indices), name="next_quarter"
            )

        kl_weight = self._kl_weight()
        return DenseVariationalLayer(
            units=len(self.dataset.tgt_indices),
            kl_weight=kl_weight,
            name="next_quarter",
        )

    def fit(self, **kwargs):
        """
        Train the model on the training dataset.

        Fits the model using the training dataset and validates on the
        validation dataset if available. Saves the best model weights
        based on validation loss.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to model.fit().

        Returns
        -------
        keras.callbacks.History
            Training history object containing loss and metrics per epoch.
        """
        checkpoint_cb = keras.callbacks.ModelCheckpoint(
            filepath=self.config.lstm.checkpoint_path / "best_model_ckpt.weights.h5",
            monitor="val_loss" if self.dataset.val_dataset is not None else "loss",
            save_best_only=True,
            save_weights_only=True,
        )
        if not hasattr(checkpoint_cb, "_implements_train_batch_hooks"):
            checkpoint_cb._implements_train_batch_hooks = lambda: False

        history = self.model.fit(
            self.dataset.train_dataset,
            validation_data=self.dataset.val_dataset,
            epochs=self.config.lstm.epochs,
            callbacks=[checkpoint_cb],
            **kwargs,
        )
        weights_path = self.config.lstm.checkpoint_path / "best_model_ckpt.weights.h5"
        if weights_path.exists():
            self.model.load_weights(weights_path)
        return history

    def predict(self, x: np.ndarray | None = None) -> TickerResults:
        """
        Generate predictions for future periods (no ground truth).

        Parameters
        ----------
        x : np.ndarray, optional
            Input window of shape (1, lookback, features). If None, uses
            the dataset's predict_dataset (final lookback window).

        Returns
        -------
        TickerResults
            Prediction results with forecasted values. MAE and ground truth
            fields are set to 0 since no actuals are available.
        """
        if x is None:
            x = self.dataset.X_predict

        y_pred, pred_std = self._predict_with_uncertainty(x)

        # Unscale predictions
        y_pred_unscaled = y_pred * self.dataset.target_std + self.dataset.target_mean

        per_feature_std = np.zeros(y_pred_unscaled.shape[-1])
        if pred_std is not None:
            pred_std_unscaled = pred_std * self.dataset.target_std
            per_feature_std = np.mean(pred_std_unscaled, axis=0)

        feature_metrics = {
            name: Metric(
                value=float(y_pred_unscaled[:, idx].mean()),
                mae=0.0,
                gt=0.0,
                std=float(per_feature_std[idx]) if pred_std is not None else 0.0,
            )
            for name, idx in self.dataset.feat_to_idx.items()
        }

        asset_idx = self.dataset.feature_mappings["assets"]
        liability_idx = self.dataset.feature_mappings["liabilities"]
        equity_idx = self.dataset.feature_mappings["equity"]

        assets_pred = float(np.sum(y_pred_unscaled[:, asset_idx], axis=-1).mean())
        liabilities_pred = float(
            np.sum(y_pred_unscaled[:, liability_idx], axis=-1).mean()
        )
        equity_pred = float(np.sum(y_pred_unscaled[:, equity_idx], axis=-1).mean())

        ni_idx = self.dataset.feat_to_idx.get("Net Income")
        net_income_pred = (
            float(y_pred_unscaled[:, ni_idx].mean()) if ni_idx is not None else 0.0
        )

        ticker_results = TickerResults(
            assets=Metric(value=assets_pred),
            liabilities=Metric(value=liabilities_pred),
            equity=Metric(value=equity_pred),
            features=feature_metrics,
            net_income_pred=net_income_pred,
            pred_std={
                name: float(per_feature_std[idx])
                for name, idx in self.dataset.feat_to_idx.items()
            },
        )

        self.predict_results = ticker_results
        return ticker_results

    def save(self, path: str):
        """
        Save the model to disk.

        Parameters
        ----------
        path : str
            File path where the model will be saved.
        """
        self.model.save(path)

    @classmethod
    def load(cls, path: str, config: Config) -> "LSTMForecaster":
        """
        Load a saved model from disk.

        Parameters
        ----------
        path : str
            File path to the saved model.
        config : Config
            Configuration object for the loaded model.

        Returns
        -------
        LSTMForecaster
            Loaded LSTMForecaster instance with restored weights.
        """
        obj = cls.__new__(cls)
        obj.config = config
        obj.model = tf.keras.models.load_model(path)
        return obj

    def evaluate(
        self, stage: str = "val", llm_config: LLMConfig | None = None
    ) -> TickerResults:
        """
        Evaluate the model and compute metrics.

        Evaluates the model on the specified dataset stage (train or val),
        computes predictions, and calculates various performance metrics
        including MAE, baseline comparisons, and skill scores.

        Parameters
        ----------
        stage : str, default="val"
            Dataset stage to evaluate on, either "val" or "train".
        llm_config : LLMConfig, optional
            Configuration for LLM-based forecast adjustments. If provided,
            uses LLM to refine or generate predictions.

        Returns
        -------
        TickerResults
            Comprehensive evaluation results including predictions, errors,
            baseline comparisons, and aggregate metrics.
        """
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

        pred_std_unscaled = None
        if pred_std is not None:
            pred_std_unscaled = pred_std * self.dataset.target_std

        if llm_config:
            history_df = pd.DataFrame(
                history_unscaled[0],
                columns=self.dataset.targets,
                index=self._prediction_index(
                    stage, history_unscaled[0].shape[0], for_history=True
                ),
                copy=False,
            )

            y_pred_unscaled_df = pd.DataFrame(
                y_pred_unscaled,
                columns=self.dataset.targets,
                index=self._prediction_index(stage, y_pred_unscaled.shape[0]),
                copy=False,
            )

            llm_client = LLMClient()

            if llm_config.adjust:
                llm_estimation = llm_client.forecast_next_quarter(
                    history=history_df,
                    prediction=y_pred_unscaled_df,
                    cfg=llm_config,
                    verbose=True,
                )
            else:
                llm_estimation = llm_client.forecast_next_quarter(
                    history=history_df, cfg=llm_config, verbose=True
                )

            y_pred_unscaled_df = y_pred_unscaled_df.apply(
                pd.to_numeric, errors="coerce"
            )
            llm_estimation = llm_estimation.apply(pd.to_numeric, errors="coerce")

            # Ensure both indices are in pandas datetime format
            y_pred_unscaled_df.index = pd.to_datetime(y_pred_unscaled_df.index)
            llm_estimation.index = pd.to_datetime(llm_estimation.index)

            # LLM returns single row, broadcast to match validation set size
            if llm_config.adjust:
                # Replace all predictions with LLM estimation (broadcast single row)
                y_pred_unscaled = np.tile(
                    llm_estimation.values, (y_pred_unscaled.shape[0], 1)
                )
            else:
                # Average LSTM and LLM (broadcast single row)
                llm_broadcast = np.tile(
                    llm_estimation.values, (y_pred_unscaled.shape[0], 1)
                )
                y_pred_unscaled = (y_pred_unscaled + llm_broadcast) / 2.0

        feature_metrics, per_feature_std = self._compute_feature_metrics(
            y_pred_unscaled, y_gt_unscaled, pred_std_unscaled
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
        """
        Collect all batches from a dataset into arrays.

        Parameters
        ----------
        ds : tf.data.Dataset
            Dataset to collect batches from.
        stage : str
            Stage name for error messages ("train" or "val").

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple of (history, y_gt) where history has shape
            (n_samples, lookback, features) and y_gt has shape
            (n_samples, n_targets).
        """
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
        """
        Make predictions with uncertainty estimates.

        Routes to appropriate prediction method based on model configuration
        (probabilistic, variational, or deterministic).

        Parameters
        ----------
        history : np.ndarray
            Input sequences of shape (n_samples, lookback, features).

        Returns
        -------
        tuple[np.ndarray, np.ndarray | None]
            Tuple of (predictions, std) where predictions has shape
            (n_samples, n_targets) and std is either None (deterministic)
            or has the same shape as predictions.
        """
        pred_std = None
        if self.config.lstm.probabilistic:
            y_pred, pred_std = self._predict_probabilistic(history)
        elif self.config.lstm.variational and self.config.lstm.mc_samples > 1:
            y_pred, pred_std = self._predict_variational(history)
        else:
            y_pred = self.model.predict(history, verbose=0)
        return y_pred, pred_std

    def _predict_probabilistic(
        self, history: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Make probabilistic predictions using distribution outputs.

        For probabilistic models, extracts mean and standard deviation from
        the output distribution. Optionally uses Monte Carlo sampling for
        additional uncertainty quantification.

        Parameters
        ----------
        history : np.ndarray
            Input sequences of shape (n_samples, lookback, features).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple of (mean_predictions, std_predictions) both with shape
            (n_samples, n_targets).
        """
        n = len(self.dataset.targets)

        def params_to_dist(params):
            """Convert parameter tensor to distribution."""
            loc = params[..., :n]
            scale_params = params[..., n:]
            scale_tril = tfp.bijectors.FillScaleTriL(
                diag_bijector=tfp.bijectors.Softplus(low=1e-3)
            )(scale_params)
            return tfd.MultivariateNormalTriL(loc=loc, scale_tril=scale_tril)

        if self.config.lstm.mc_samples > 1:
            param_outputs = [
                self.model(history, training=False)
                for _ in range(self.config.lstm.mc_samples)
            ]
            dists = [params_to_dist(p) for p in param_outputs]
            means = np.stack([d.mean().numpy() for d in dists], axis=0)
            stds = np.stack([d.stddev().numpy() for d in dists], axis=0)
            return np.mean(means, axis=0), np.mean(stds, axis=0)

        params = self.model(history, training=False)
        dist = params_to_dist(params)
        return dist.mean().numpy(), dist.stddev().numpy()

    def _predict_variational(
        self, history: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Make variational predictions using Monte Carlo sampling.

        Performs multiple forward passes with weight sampling enabled to estimate
        prediction uncertainty via Monte Carlo sampling from the variational posterior.

        Parameters
        ----------
        history : np.ndarray
            Input sequences of shape (n_samples, lookback, features).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple of (mean_predictions, std_predictions) both with shape
            (n_samples, n_targets).
        """
        samples = [
            self.model(history, training=True).numpy()
            for _ in range(self.config.lstm.mc_samples)
        ]
        return np.mean(samples, axis=0), np.std(samples, axis=0)

    def _unscale(
        self, y_pred: np.ndarray, y_gt: np.ndarray, history: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Unscale normalised predictions back to original scale.

        Reverses standardisation by multiplying by standard deviation
        and adding the mean.

        Parameters
        ----------
        y_pred : np.ndarray
            Normalised predictions of shape (n_samples, n_targets).
        y_gt : np.ndarray
            Normalised ground truth of shape (n_samples, n_targets).
        history : np.ndarray
            Normalised history of shape (n_samples, lookback, features).

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Tuple of (y_pred_unscaled, y_gt_unscaled, history_unscaled).
        """
        y_pred_unscaled = y_pred * self.dataset.target_std + self.dataset.target_mean
        y_gt_unscaled = y_gt * self.dataset.target_std + self.dataset.target_mean
        history_unscaled = history * self.dataset.target_std + self.dataset.target_mean
        return y_pred_unscaled, y_gt_unscaled, history_unscaled

    def _prediction_index(self, stage: str, count: int, for_history: bool = False):
        """
        Create datetime index for predictions.

        Generates a pandas DatetimeIndex aligned with the prediction
        or history timestamps based on the data's time series index.

        Parameters
        ----------
        stage : str
            Dataset stage ("train" or "val") to determine time window.
        count : int
            Number of time steps to include in the index.
        for_history : bool, default=False
            If True, creates index for historical data; otherwise for predictions.

        Returns
        -------
        pd.DatetimeIndex or None
            DatetimeIndex for the predictions or None if unavailable.
        """
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
        """
        Compute per-feature metrics.

        Calculates mean absolute error, predicted values, ground truth,
        and standard deviations for each target feature.

        Parameters
        ----------
        y_pred_unscaled : np.ndarray
            Unscaled predictions of shape (n_samples, n_targets).
        y_gt_unscaled : np.ndarray
            Unscaled ground truth of shape (n_samples, n_targets).
        pred_std : np.ndarray or None
            Prediction standard deviations, shape (n_samples, n_targets).

        Returns
        -------
        tuple[dict[str, Metric], np.ndarray]
            Tuple of (feature_metrics, per_feature_std) where feature_metrics
            maps feature names to Metric objects and per_feature_std is an
            array of shape (n_targets,).
        """
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
        """
        Calculate per-feature errors and standard deviations.

        Parameters
        ----------
        y_pred_unscaled : np.ndarray
            Unscaled predictions of shape (n_samples, n_targets).
        y_gt_unscaled : np.ndarray
            Unscaled ground truth of shape (n_samples, n_targets).
        pred_std : np.ndarray or None
            Prediction standard deviations, shape (n_samples, n_targets).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple of (per_feature_mae, per_feature_std) both with shape
            (n_targets,).
        """
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
        """
        Build TickerResults object from predictions and ground truth.

        Aggregates per-feature predictions into balance sheet categories
        (assets, liabilities, equity) and computes baseline comparisons.

        Parameters
        ----------
        y_pred_unscaled : np.ndarray
            Unscaled predictions of shape (n_samples, n_targets).
        y_gt_unscaled : np.ndarray
            Unscaled ground truth of shape (n_samples, n_targets).
        feature_metrics : dict[str, Metric]
            Per-feature metrics mapping feature names to Metric objects.
        per_feature_std : np.ndarray
            Per-feature standard deviations, shape (n_targets,).
        history_unscaled : np.ndarray
            Unscaled history of shape (n_samples, lookback, features).

        Returns
        -------
        TickerResults
            Complete evaluation results with aggregated metrics.
        """
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
        """
        Compute net income baseline metrics.

        Compares model predictions for Net Income against baseline methods
        (e.g., persistence, seasonal naive) and computes skill scores.

        Parameters
        ----------
        y_pred_unscaled : np.ndarray
            Unscaled predictions of shape (n_samples, n_targets).
        y_gt_unscaled : np.ndarray
            Unscaled ground truth of shape (n_samples, n_targets).
        history_unscaled : np.ndarray
            Unscaled history of shape (n_samples, lookback, features).

        Returns
        -------
        dict[str, dict[str, float] | float]
            Dictionary containing baseline_mae, skill, model_mae, pred,
            gt, and baseline_pred for Net Income.
        """
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
        """
        Display formatted results table.

        Prints comprehensive evaluation results including overall metrics,
        per-feature breakdowns, and baseline comparisons.

        Parameters
        ----------
        stage : str, default="val"
            Dataset stage to display results for ("val" or "train").
        """
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
            self.dataset.bs_structure["Assets"], results.features
        )
        print_table("Assets", assets_rows)

        liabilities_rows = build_section_rows(
            self.dataset.bs_structure["Liabilities"], results.features
        )
        print_table("Liabilities", liabilities_rows)

        equity_rows = build_equity_rows(
            self.dataset.bs_structure["Equity"], results.features
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

    def sample_predictions(
        self, history: np.ndarray, n_samples: int = 100
    ) -> np.ndarray:
        """
        Generate multiple samples from the predictive distribution.

        For probabilistic models, samples from the output distribution.
        For variational models, uses Monte Carlo dropout sampling.

        Parameters
        ----------
        history : np.ndarray
            Input sequences of shape (n_samples, lookback, features).
        n_samples : int, default=100
            Number of samples to generate from the predictive distribution.

        Returns
        -------
        np.ndarray
            Samples with shape (n_samples, batch_size, n_targets).

        Raises
        ------
        ValueError
            If model is not probabilistic or variational.
        """
        if not (self.config.lstm.probabilistic or self.config.lstm.variational):
            raise ValueError(
                "sample_predictions requires a probabilistic or variational model"
            )

        if self.config.lstm.probabilistic:
            n = len(self.dataset.targets)
            params = self.model(history, training=False)

            loc = params[..., :n]
            scale_params = params[..., n:]
            scale_tril = tfp.bijectors.FillScaleTriL(
                diag_bijector=tfp.bijectors.Softplus(low=1e-3)
            )(scale_params)
            dist = tfd.MultivariateNormalTriL(loc=loc, scale_tril=scale_tril)

            samples = dist.sample(n_samples).numpy()
        else:
            samples = np.stack(
                [self.model(history, training=True).numpy() for _ in range(n_samples)],
                axis=0,
            )
        return samples

    def predict_quantiles(
        self, history: np.ndarray, quantiles: list[float] | None = None
    ) -> dict[float, np.ndarray]:
        """
        Compute quantile predictions from the predictive distribution.

        Parameters
        ----------
        history : np.ndarray
            Input sequences of shape (n_samples, lookback, features).
        quantiles : list[float], optional
            List of quantiles to compute (e.g., [0.05, 0.25, 0.5, 0.75, 0.95]).
            Default: [0.05, 0.25, 0.5, 0.75, 0.95].

        Returns
        -------
        dict[float, np.ndarray]
            Dictionary mapping quantile values to predictions of shape
            (n_samples, n_targets).

        Raises
        ------
        ValueError
            If model is not probabilistic or variational.
        """
        if quantiles is None:
            quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]

        if not (self.config.lstm.probabilistic or self.config.lstm.variational):
            raise ValueError(
                "predict_quantiles requires a probabilistic or variational model"
            )

        samples = self.sample_predictions(history, n_samples=1000)

        quantile_preds = {}
        for q in quantiles:
            quantile_preds[q] = np.percentile(samples, q * 100, axis=0)

        return quantile_preds

    def get_prediction_intervals(
        self,
        history: np.ndarray,
        confidence_levels: list[float] | None = None,
    ) -> dict[float, tuple[np.ndarray, np.ndarray]]:
        """
        Compute prediction intervals at various confidence levels.

        Parameters
        ----------
        history : np.ndarray
            Input sequences of shape (n_samples, lookback, features).
        confidence_levels : list[float], optional
            List of confidence levels (e.g., [0.68, 0.95, 0.99]).
            Default: [0.68, 0.95, 0.99].

        Returns
        -------
        dict[float, tuple[np.ndarray, np.ndarray]]
            Dictionary mapping confidence levels to (lower, upper) tuples,
            each with shape (n_samples, n_targets).

        Raises
        ------
        ValueError
            If model is not probabilistic or variational.
        """
        if confidence_levels is None:
            confidence_levels = [0.68, 0.95, 0.99]

        if not (self.config.lstm.probabilistic or self.config.lstm.variational):
            raise ValueError(
                "get_prediction_intervals requires a probabilistic or variational model"
            )

        intervals = {}
        for level in confidence_levels:
            alpha = (1 - level) / 2
            lower_q = alpha
            upper_q = 1 - alpha

            quantiles = self.predict_quantiles(history, [lower_q, upper_q])
            intervals[level] = (quantiles[lower_q], quantiles[upper_q])

        return intervals

    def compute_calibration_metrics(
        self, stage: str = "val"
    ) -> dict[str, float | dict]:
        """
        Compute calibration metrics for probabilistic predictions.

        Evaluates whether the predicted uncertainties are well-calibrated by
        checking if the empirical coverage matches the nominal coverage.

        Parameters
        ----------
        stage : str, default="val"
            Dataset stage to evaluate on ("val" or "train").

        Returns
        -------
        dict[str, float | dict]
            Dictionary containing:
            - coverage: dict mapping confidence levels to empirical coverage
            - calibration_error: mean absolute difference between nominal and empirical coverage
            - sharpness: average prediction interval width
            - crps: continuous ranked probability score (if available)

        Raises
        ------
        ValueError
            If model is not probabilistic or stage is invalid.
        """
        if not self.config.lstm.probabilistic:
            raise ValueError(
                "compute_calibration_metrics requires a probabilistic model"
            )

        if stage not in {"val", "train"}:
            raise ValueError("stage must be 'val' or 'train'")

        ds = self.dataset.val_dataset if stage == "val" else self.dataset.train_dataset
        if ds is None:
            raise ValueError(f"{stage} dataset is not available")

        history, y_gt = self._collect_batches(ds, stage)

        confidence_levels = [0.50, 0.68, 0.80, 0.90, 0.95, 0.99]
        intervals = self.get_prediction_intervals(history, confidence_levels)

        coverage = {}
        interval_widths = []

        for level, (lower, upper) in intervals.items():
            in_interval = (y_gt >= lower) & (y_gt <= upper)
            empirical_coverage = np.mean(in_interval)
            coverage[level] = float(empirical_coverage)

            interval_widths.append(np.mean(upper - lower))

        calibration_error = np.mean(
            [abs(coverage[level] - level) for level in confidence_levels]
        )

        sharpness = float(np.mean(intervals[0.95][1] - intervals[0.95][0]))

        crps = None
        try:
            samples = self.sample_predictions(history, n_samples=200)
            mean_abs_error = np.mean(np.abs(samples - y_gt[np.newaxis, :, :]), axis=0)
            mean_spread = np.mean(np.abs(samples[:100] - samples[100:]), axis=0) / 2
            crps_per_feature = np.mean(mean_abs_error - mean_spread, axis=0)
            crps = float(np.mean(crps_per_feature))
        except Exception:
            pass

        return {
            "coverage": coverage,
            "calibration_error": float(calibration_error),
            "sharpness": sharpness,
            "crps": crps,
        }

    def plot_prediction_intervals(
        self,
        feature_name: str,
        stage: str = "val",
        confidence_levels: list[float] | None = None,
        max_periods: int = 20,
    ) -> None:
        """
        Plot prediction intervals for a specific feature.

        Creates a visualisation showing the ground truth, predictions, and
        uncertainty bands at different confidence levels.

        Parameters
        ----------
        feature_name : str
            Name of the feature to plot (must be in dataset.targets).
        stage : str, default="val"
            Dataset stage to plot ("val" or "train").
        confidence_levels : list[float], optional
            Confidence levels to plot. Default: [0.68, 0.95].
        max_periods : int, default=20
            Maximum number of time periods to display.

        Raises
        ------
        ValueError
            If model is not probabilistic, feature not found, or stage is invalid.
        """
        if not self.config.lstm.probabilistic:
            raise ValueError("plot_prediction_intervals requires a probabilistic model")

        if feature_name not in self.dataset.targets:
            raise ValueError(f"Feature '{feature_name}' not found in targets")

        if stage not in {"val", "train"}:
            raise ValueError("stage must be 'val' or 'train'")

        if confidence_levels is None:
            confidence_levels = [0.68, 0.95]

        import matplotlib.pyplot as plt

        ds = self.dataset.val_dataset if stage == "val" else self.dataset.train_dataset
        history, y_gt = self._collect_batches(ds, stage)

        n_periods = min(max_periods, y_gt.shape[0])
        history = history[:n_periods]
        y_gt = y_gt[:n_periods]

        y_pred, pred_std = self._predict_with_uncertainty(history)
        intervals = self.get_prediction_intervals(history, confidence_levels)

        y_pred_unscaled = y_pred * self.dataset.target_std + self.dataset.target_mean
        y_gt_unscaled = y_gt * self.dataset.target_std + self.dataset.target_mean

        feat_idx = self.dataset.feat_to_idx[feature_name]

        pred = y_pred_unscaled[:, feat_idx]
        gt = y_gt_unscaled[:, feat_idx]

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(n_periods)

        ax.plot(x, gt, "o-", label="Ground Truth", color="black", linewidth=2)
        ax.plot(x, pred, "s-", label="Prediction", color="blue", linewidth=2)

        colors = ["lightblue", "lightcoral", "lightyellow"]
        for i, level in enumerate(sorted(confidence_levels, reverse=True)):
            lower, upper = intervals[level]
            lower_unscaled = lower * self.dataset.target_std + self.dataset.target_mean
            upper_unscaled = upper * self.dataset.target_std + self.dataset.target_mean

            ax.fill_between(
                x,
                lower_unscaled[:, feat_idx],
                upper_unscaled[:, feat_idx],
                alpha=0.3,
                color=colors[i % len(colors)],
                label=f"{level * 100:.0f}% CI",
            )

        ax.set_xlabel("Time Period")
        ax.set_ylabel(f"{feature_name} (Unscaled)")
        ax.set_title(
            f"Probabilistic Forecast: {feature_name} ({stage} set)\n"
            f"{self.config.data.ticker}"
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_uncertainty_heatmap(self, stage: str = "val") -> None:
        """
        Plot a heatmap of prediction uncertainties across all features.

        Visualises which features have the highest prediction uncertainty.

        Parameters
        ----------
        stage : str, default="val"
            Dataset stage to plot ("val" or "train").

        Raises
        ------
        ValueError
            If model is not probabilistic or stage is invalid.
        """
        if not self.config.lstm.probabilistic:
            raise ValueError("plot_uncertainty_heatmap requires a probabilistic model")

        if stage not in {"val", "train"}:
            raise ValueError("stage must be 'val' or 'train'")

        import matplotlib.pyplot as plt
        import seaborn as sns

        results = self.val_results if stage == "val" else self.train_results

        feature_names = []
        uncertainties = []

        for name in self.dataset.targets:
            if name in results.pred_std:
                feature_names.append(name)
                uncertainties.append(results.pred_std[name])

        data = np.array(uncertainties).reshape(-1, 1)

        fig, ax = plt.subplots(figsize=(10, max(8, len(feature_names) * 0.3)))
        sns.heatmap(
            data,
            yticklabels=feature_names,
            xticklabels=["Std Dev"],
            annot=True,
            fmt=".2e",
            cmap="YlOrRd",
            cbar_kws={"label": "Prediction Std Dev"},
            ax=ax,
        )

        ax.set_title(
            f"Prediction Uncertainty by Feature ({stage} set)\n{self.config.data.ticker}"
        )
        plt.tight_layout()
        plt.show()

    def view_calibration_results(self, stage: str = "val") -> None:
        """
        Display calibration diagnostics in a formatted table.

        Shows empirical coverage vs nominal coverage, calibration error,
        sharpness, and CRPS metrics.

        Parameters
        ----------
        stage : str, default="val"
            Dataset stage to evaluate ("val" or "train").

        Raises
        ------
        ValueError
            If model is not probabilistic or stage is invalid.
        """
        if not self.config.lstm.probabilistic:
            print("Model is not probabilistic. Use view_results() instead.")
            return

        print(
            f"\033[1mCalibration Diagnostics for {stage} dataset ({self.config.data.ticker}):\033[0m\n"
        )

        metrics = self.compute_calibration_metrics(stage)

        coverage_rows = []
        for level in sorted(metrics["coverage"].keys()):
            nominal = level
            empirical = metrics["coverage"][level]
            diff = empirical - nominal

            if abs(diff) < 0.05:
                diff_str = colour(f"{diff:+.3f}", "green")
            elif abs(diff) < 0.10:
                diff_str = colour(f"{diff:+.3f}", "yellow")
            else:
                diff_str = colour(f"{diff:+.3f}", "red")

            coverage_rows.append(
                [
                    f"{nominal * 100:.0f}%",
                    f"{empirical * 100:.1f}%",
                    diff_str,
                ]
            )

        print_table(
            "Prediction Interval Coverage",
            coverage_rows,
            headers=("Nominal Coverage", "Empirical Coverage", "Difference"),
        )

        print("\nOverall Calibration Metrics:")
        print(f"  Calibration Error (MAE): {colour_mae(metrics['calibration_error'])}")
        print(f"  Sharpness (95% PI width): {colour_mae(metrics['sharpness'])}")
        if metrics["crps"] is not None:
            print(f"  CRPS: {colour_mae(metrics['crps'])}")

        print("\nInterpretation:")
        print(
            "  - Well-calibrated: Empirical coverage ≈ Nominal coverage (difference < 5%)"
        )
        print("  - Sharpness: Lower is better (narrower prediction intervals)")
        print("  - CRPS: Lower is better (better probabilistic forecasts)")
        print()

    def view_probabilistic_results(
        self, stage: str = "val", confidence_level: float = 0.95
    ) -> None:
        """
        Display probabilistic forecast results with uncertainty intervals.

        Shows predictions with confidence intervals for probabilistic models,
        including per-feature uncertainty estimates and distribution statistics.

        Parameters
        ----------
        stage : str, default="val"
            Dataset stage to display results for ("val" or "train").
        confidence_level : float, default=0.95
            Confidence level for prediction intervals (e.g., 0.95 for 95% CI).
        """
        if not self.config.lstm.probabilistic:
            print("Model is not probabilistic. Use view_results() instead.")
            return

        results = self.val_results if stage == "val" else self.train_results

        print(
            f"\033[1mProbabilistic Results for {stage} dataset ({self.config.data.ticker}):\033[0m"
        )
        print(f"Confidence Level: {confidence_level * 100:.0f}%\n")

        from scipy import stats

        z = stats.norm.ppf((1 + confidence_level) / 2)

        def make_prob_row(category: str, metric: Metric, std: float) -> list[str]:
            """Build row with prediction, CI, and ground truth."""
            ci_lower = metric.value - z * std
            ci_upper = metric.value + z * std
            return [
                category,
                format_money(metric.gt),
                format_money(metric.value),
                f"{format_money(ci_lower)} to {format_money(ci_upper)}",
                colour_mae(metric.mae),
            ]

        asset_idx = self.dataset.feature_mappings["assets"]
        liability_idx = self.dataset.feature_mappings["liabilities"]
        equity_idx = self.dataset.feature_mappings["equity"]

        assets_std = np.sqrt(
            sum(
                results.pred_std[name] ** 2
                for name in self.dataset.targets
                if self.dataset.feat_to_idx.get(name) in asset_idx
            )
        )
        liabilities_std = np.sqrt(
            sum(
                results.pred_std[name] ** 2
                for name in self.dataset.targets
                if self.dataset.feat_to_idx.get(name) in liability_idx
            )
        )
        equity_std = np.sqrt(
            sum(
                results.pred_std[name] ** 2
                for name in self.dataset.targets
                if self.dataset.feat_to_idx.get(name) in equity_idx
            )
        )

        overall_rows = [
            make_prob_row("Assets", results.assets, assets_std),
            make_prob_row("Liabilities", results.liabilities, liabilities_std),
            make_prob_row("Equity", results.equity, equity_std),
        ]
        print_table(
            "Overall with Confidence Intervals",
            overall_rows,
            headers=(
                "Category",
                "Ground Truth",
                "Predicted",
                f"{confidence_level * 100:.0f}% CI",
                "Error",
            ),
        )

        def build_prob_section_rows(
            sections: dict[str, list[str]],
            feature_stats: dict[str, Metric],
        ) -> list[list[str]]:
            """Create rows with uncertainty for features."""
            rows = []
            for feat in sections:
                m = feature_stats.get(feat)
                if m is None:
                    continue
                std = results.pred_std.get(feat, 0.0)
                ci_lower = m.value - z * std
                ci_upper = m.value + z * std
                rows.append(
                    [
                        fmt(feat),
                        format_money(m.gt),
                        format_money(m.value),
                        f"{format_money(ci_lower)} to {format_money(ci_upper)}",
                        colour_mae(m.mae),
                    ]
                )
            return rows

        assets_rows = build_prob_section_rows(
            self.dataset.bs_structure["Assets"], results.features
        )
        print_table(
            "Assets with Confidence Intervals",
            assets_rows,
            headers=(
                "Category",
                "Ground Truth",
                "Predicted",
                f"{confidence_level * 100:.0f}% CI",
                "Error",
            ),
        )

        liabilities_rows = build_prob_section_rows(
            self.dataset.bs_structure["Liabilities"], results.features
        )
        print_table(
            "Liabilities with Confidence Intervals",
            liabilities_rows,
            headers=(
                "Category",
                "Ground Truth",
                "Predicted",
                f"{confidence_level * 100:.0f}% CI",
                "Error",
            ),
        )

        equity_rows = build_prob_section_rows(
            self.dataset.bs_structure["Equity"], results.features
        )
        print_table(
            "Equity with Confidence Intervals",
            equity_rows,
            headers=(
                "Category",
                "Ground Truth",
                "Predicted",
                f"{confidence_level * 100:.0f}% CI",
                "Error",
            ),
        )

        print("\nUncertainty Statistics:")
        avg_std = np.mean(list(results.pred_std.values()))
        max_std_feature = max(results.pred_std.items(), key=lambda x: x[1])
        min_std_feature = min(results.pred_std.items(), key=lambda x: x[1])

        print(f"  Average Prediction Std Dev: {colour_mae(avg_std)}")
        print(
            f"  Highest Uncertainty: {fmt(max_std_feature[0])} ({colour_mae(max_std_feature[1])})"
        )
        print(
            f"  Lowest Uncertainty: {fmt(min_std_feature[0])} ({colour_mae(min_std_feature[1])})"
        )

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
    from jpm.config.question_1 import Config, DataConfig, LSTMConfig, StatementsDataset
    from jpm.question_1.models import BalanceSheet, IncomeStatement
    from src.jpm.question_1.misc import get_args, set_seed

    set_seed(42)
    args = get_args()

    data_cfg = DataConfig.from_args(args)
    lstm_cfg = LSTMConfig.from_args(args)

    config = Config(data=data_cfg, lstm=lstm_cfg)

    data = EdgarData(config=config)
    dataset = StatementsDataset(config=config, data=data)

    model = LSTMForecaster(config=config, data=data, dataset=dataset)
    model.fit()

    model.evaluate(stage="train")
    validation_results = model.evaluate(stage="val")

    model.view_results(stage="val")

    # Pass outputs to BS Model
    bs = BalanceSheet(
        config=config, data=data, dataset=dataset, results=validation_results
    )
    bs.check_identity()

    # Income Statement to predict Net Income (Loss)
    i_s = IncomeStatement(
        config=config, data=data, dataset=dataset, results=validation_results
    )
    i_s.view()
