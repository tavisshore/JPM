from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf
from tensorflow.keras import Model, layers


@dataclass
class LSTMConfig:
    lookback: int
    num_features: int
    lstm_units: int = 128
    lstm_layers: int = 1
    dense_units: int = 128
    dropout: float = 0.1
    learning_rate: float = 1e-3


class LSTMForecaster:
    def __init__(self, config: LSTMConfig):
        self.config = config
        self.model = self._build_model(config)
        self._compile_model()

    def _build_model(self, cfg: LSTMConfig) -> Model:
        inputs = layers.Input(shape=(cfg.lookback, cfg.num_features))

        x = inputs
        # Optional: stack multiple LSTM layers
        for i in range(cfg.lstm_layers):
            return_sequences = i < cfg.lstm_layers - 1
            x = layers.LSTM(
                cfg.lstm_units,
                return_sequences=return_sequences,
                name=f"lstm_{i + 1}",
            )(x)

        if cfg.dropout > 0:
            x = layers.Dropout(cfg.dropout)(x)

        if cfg.dense_units is not None:
            x = layers.Dense(cfg.dense_units, activation="relu")(x)

        # Output: next quarter's full feature vector (F)
        outputs = layers.Dense(cfg.num_features, name="next_step")(x)

        return Model(inputs=inputs, outputs=outputs, name="lstm_forecaster")

    def _compile_model(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(self.config.learning_rate),
            loss="mse",
            metrics=["mae"],
        )

    # Convenience wrappers so calling code doesn't touch raw Keras much
    def fit(self, train_ds, val_ds=None, epochs: int = 50, **kwargs):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss" if val_ds is not None else "loss",
                patience=5,
                restore_best_weights=True,
            )
        ]
        return self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            **kwargs,
        )

    def predict(self, x):
        return self.model.predict(x)

    def save(self, path: str):
        self.model.save(path)

    @classmethod
    def load(cls, path: str, config: LSTMConfig) -> "LSTMForecaster":
        obj = cls.__new__(cls)
        obj.config = config
        obj.model = tf.keras.models.load_model(path)
        return obj
