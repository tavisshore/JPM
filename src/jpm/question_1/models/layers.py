from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp

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


class SeasonalWeightLogger(keras.callbacks.Callback):
    """Callback to log learned seasonal weight values during training.

    This callback monitors the seasonal weight parameter(s) in the TemporalAttention
    layer and logs their values at the end of each epoch. Useful for understanding
    how the model learns to weight seasonal vs non-seasonal timesteps.

    Parameters
    ----------
    layer_name : str, optional
        Name of the TemporalAttention layer to monitor. Default is "temporal_attention".
    """

    def __init__(self, layer_name: str = "temporal_attention"):
        super().__init__()
        self.layer_name = layer_name
        self.seasonal_layer = None
        self.weight_history = []

    def on_train_begin(self, logs=None):
        """Find the TemporalAttention layer at the start of training."""
        for layer in self.model.layers:
            if layer.name == self.layer_name:
                self.seasonal_layer = layer
                break

        if self.seasonal_layer is None:
            print(
                f"Warning: Could not find layer '{self.layer_name}' for weight monitoring"
            )

    def on_epoch_end(self, epoch, logs=None, verbose=0):
        """Log seasonal weight value(s) at the end of each epoch."""
        if self.seasonal_layer is None:
            return

        # Get the seasonal weight value(s)
        weight_value = self.seasonal_layer.seasonal_weight.numpy()
        self.weight_history.append(weight_value.copy())

        # Format output based on whether it's per-feature or global
        if weight_value.size == 1:
            # Single global weight
            logs = logs or {}
            logs["seasonal_weight"] = float(weight_value.item())
            if (epoch % 25 == 0 or epoch < 5) and verbose > 0:
                print(f"\nEpoch {epoch + 1}: Seasonal Weight = {weight_value[0]:.4f}")
        else:
            # Per-feature weights
            logs = logs or {}
            logs["seasonal_weight_mean"] = float(weight_value.mean())
            logs["seasonal_weight_std"] = float(weight_value.std())
            if (epoch % 25 == 0 or epoch < 5) and verbose > 0:
                print(
                    f"\nEpoch {epoch + 1}: Seasonal Weights - "
                    f"Mean: {weight_value.mean():.4f}, "
                    f"Std: {weight_value.std():.4f}, "
                    f"Range: [{weight_value.min():.4f}, {weight_value.max():.4f}]"
                )

    def get_weight(self):
        """Get the history of seasonal weight values over epochs."""
        return self.weight_history


class TemporalAttention(keras.layers.Layer):
    """Learnable attention over timesteps with seasonal bias.

    This layer learns to weight different timesteps in the input sequence,
    with special emphasis on seasonal timesteps. Instead of applying a fixed
    seasonal weight during preprocessing, this layer learns the optimal
    weighting during training.

    Parameters
    ----------
    seasonal_lag : int
        The lag between seasonal timesteps (e.g., 4 for quarterly data).
    initial_weight : float, optional
        Initial value for the seasonal weight parameter. Default is 2.0.
    per_feature : bool, optional
        If True, learns separate seasonal weights for each feature.
        If False, learns a single seasonal weight applied to all features.
        Default is False.
    min_weight : float, optional
        Minimum allowed value for seasonal weight. Default is 0.1.
    max_weight : float, optional
        Maximum allowed value for seasonal weight. Default is 10.0.
    name : str, optional
        Name for the layer.
    """

    def __init__(
        self,
        seasonal_lag: int,
        initial_weight: float = 2.0,
        per_feature: bool = False,
        min_weight: float = 0.1,
        max_weight: float = 10.0,
        name: str = "temporal_attention",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.seasonal_lag = seasonal_lag
        self.initial_weight = initial_weight
        self.per_feature = per_feature
        self.min_weight = min_weight
        self.max_weight = max_weight

    def build(self, input_shape):
        """Build the layer's trainable weights.

        Parameters
        ----------
        input_shape : tuple
            Shape of input tensor (batch, timesteps, features).
        """
        if self.per_feature:
            # Learn one weight per feature
            weight_shape = (input_shape[-1],)
        else:
            # Learn a single weight for all features
            weight_shape = (1,)

        self.seasonal_weight = self.add_weight(
            name="seasonal_weight",
            shape=weight_shape,
            initializer=keras.initializers.Constant(self.initial_weight),
            constraint=keras.constraints.MinMaxNorm(
                min_value=self.min_weight, max_value=self.max_weight, rate=1.0
            ),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        """Apply learnable seasonal weighting to inputs.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor of shape (batch, timesteps, features).

        Returns
        -------
        tf.Tensor
            Weighted input tensor of the same shape.
        """
        # Get dimensions
        lookback = tf.shape(inputs)[1]

        # Create seasonal mask: True for seasonal timesteps
        timesteps = tf.range(lookback)
        # Seasonal positions are at lookback-1, lookback-1-seasonal_lag, etc.
        seasonal_mask = tf.equal(
            tf.math.floormod(lookback - 1 - timesteps, self.seasonal_lag), 0
        )
        seasonal_mask = tf.cast(seasonal_mask, inputs.dtype)

        # Create weight tensor: seasonal_weight for seasonal timesteps, 1.0 otherwise
        if self.per_feature:
            # Shape: (timesteps, features)
            weights = tf.where(
                seasonal_mask[:, None] > 0,
                self.seasonal_weight[None, :],
                tf.ones_like(self.seasonal_weight)[None, :],
            )
        else:
            # Shape: (timesteps, 1) - will broadcast over features
            weights = tf.where(seasonal_mask[:, None] > 0, self.seasonal_weight, 1.0)

        # Apply weights: (batch, timesteps, features) * (timesteps, features)
        return inputs * weights[None, :, :]

    def get_config(self):
        """Return layer configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "seasonal_lag": self.seasonal_lag,
                "initial_weight": self.initial_weight,
                "per_feature": self.per_feature,
                "min_weight": self.min_weight,
                "max_weight": self.max_weight,
            }
        )
        return config
