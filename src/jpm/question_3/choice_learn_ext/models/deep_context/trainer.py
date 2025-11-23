# src/jpm/question_3/choice_learn_ext/models/deep_context/trainer.py

from __future__ import annotations
from typing import Dict, Optional
import tensorflow as tf
from choice_learn_ext.models.deep_context.deep_halo_core import DeepHalo


class Trainer:
    """
    Simple training wrapper around DeepHalo / DeepContextChoiceModel.
    """

    def __init__(self, model: DeepHalo, lr: float = 1e-3):
        self.model = model
        # On Apple Silicon, legacy optimizer is usually faster/more stable
        self.optimizer = tf.keras.optimizers.legacy.Adam(lr)

    @tf.function
    def train_step(self, batch: Dict[str, tf.Tensor]) -> tf.Tensor:
        with tf.GradientTape() as tape:
            loss = self.model.nll(batch, training=True)
            # tiny L2 regularization
            reg = 1e-6 * tf.add_n(
                [tf.nn.l2_loss(v) for v in self.model.trainable_variables]
            )
            loss = loss + reg
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def fit_arrays(
        self,
        available: tf.Tensor,      # (N, J)
        choices: tf.Tensor,        # (N,)
        item_ids: Optional[tf.Tensor] = None,  # (N, J) if featureless
        X: Optional[tf.Tensor] = None,         # (N, J, d_x) if feature-based
        batch_size: int = 256,
        epochs: int = 20,
        verbose: int = 1,
    ) -> None:
        model = self.model
        cfg = model.cfg

        ds_inputs: Dict[str, tf.Tensor] = {
            "available": available,
            "choice": choices,
        }
        if cfg.featureless:
            if item_ids is None:
                raise ValueError("item_ids is required when featureless=True.")
            ds_inputs["item_ids"] = item_ids
        else:
            if X is None:
                raise ValueError("X is required when featureless=False.")
            ds_inputs["X"] = X

        ds = (
            tf.data.Dataset.from_tensor_slices(ds_inputs)
            .shuffle(4096)
            .batch(batch_size)
            .prefetch(2)
        )

        for ep in range(1, epochs + 1):
            running = 0.0
            steps = 0
            for batch in ds:
                loss = self.train_step(batch)
                running += float(loss.numpy()) #THis makes it slower
                steps += 1
            if verbose:
                print(f"Epoch {ep:03d}  NLL: {running / max(steps, 1):.4f}")

    def predict_probs(
        self,
        available: tf.Tensor,
        item_ids: Optional[tf.Tensor] = None,
        X: Optional[tf.Tensor] = None,
        batch_size: int = 512,
    ) -> tf.Tensor:
        model = self.model
        cfg = model.cfg

        ds_inputs: Dict[str, tf.Tensor] = {"available": available}
        if cfg.featureless:
            ds_inputs["item_ids"] = item_ids
        else:
            ds_inputs["X"] = X

        ds = tf.data.Dataset.from_tensor_slices(ds_inputs).batch(batch_size)
        probs_list = []
        for batch in ds:
            out = model(batch, training=False)
            probs_list.append(tf.exp(out["log_probs"]))
        return tf.concat(probs_list, axis=0)
