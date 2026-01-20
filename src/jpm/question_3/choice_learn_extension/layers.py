# src/jpm/question_3/choice_learn_ext/models/deep_context/layers.py

from __future__ import annotations

from typing import Dict

import tensorflow as tf

from jpm.config import DeepHaloConfig
from jpm.question_3.choice_learn_extension.utils import masked_mean


class BaseEncoder(tf.keras.layers.Layer):
    """
    Shared encoder χ(x_j) -> z_j^{(0)}.

    - If cfg.featureless: Embedding(vocab_size, d_embed) on item_ids
    - Else: MLP on item features X (B, J, d_x)
    """

    def __init__(self, cfg: DeepHaloConfig, name: str = "base_encoder"):
        super().__init__(name=name)
        self.cfg = cfg

        if cfg.featureless:
            if cfg.vocab_size is None:
                raise ValueError("cfg.vocab_size must be set when featureless=True.")
            self.embedding = tf.keras.layers.Embedding(
                input_dim=cfg.vocab_size,
                output_dim=cfg.d_embed,
                name="item_embedding",
            )
            # identity: embeddings already have dimension d_embed
            self.mlp = tf.keras.layers.Lambda(lambda x: x, name="identity_mlp")
        else:
            if cfg.d_x is None:
                raise ValueError("cfg.d_x must be set when featureless=False.")
            self.embedding = None
            self.mlp = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(cfg.d_embed, activation="gelu"),
                    tf.keras.layers.Dense(cfg.d_embed, activation=None),
                ],
                name="feature_mlp",
            )

    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        Returns z0: (B, J, d_embed)
        """
        if self.cfg.featureless:
            item_ids = tf.cast(inputs["item_ids"], tf.int32)  # (B, J)
            z = self.embedding(item_ids)  # (B, J, d_embed)
            return self.mlp(z, training=training)
        else:
            X = tf.convert_to_tensor(inputs["X"])  # (B, J, d_x)
            B = tf.shape(X)[0]
            J = tf.shape(X)[1]
            X_flat = tf.reshape(X, [B * J, self.cfg.d_x])
            z_flat = self.mlp(X_flat, training=training)  # (B*J, d_embed)
            return tf.reshape(z_flat, [B, J, self.cfg.d_embed])


class HaloBlock(tf.keras.layers.Layer):
    """
    One DeepHalo-style context layer.

    For each layer ℓ:
      - Compute context summary c = masked_mean(z_prev)
      - For each item j, each head h:
          u_{j,h} = φ_{ℓ,h}([z_phi_in_j, c])
      - Aggregate heads and residual:
          z_j_out = z_j_prev + (1/H) * Σ_h u_{j,h}
    """

    def __init__(self, cfg: DeepHaloConfig, name: str = "halo_block"):
        super().__init__(name=name)
        self.cfg = cfg

        self.phi_heads = []
        for h in range(cfg.n_heads):
            mlp = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(
                        cfg.d_embed,
                        activation="gelu",
                        name=f"{name}_h{h}_dense1",
                    ),
                    tf.keras.layers.Dropout(cfg.dropout),
                    tf.keras.layers.Dense(
                        cfg.d_embed,
                        activation=None,
                        name=f"{name}_h{h}_dense2",
                    ),
                ],
                name=f"{name}_phi_head_{h}",
            )
            self.phi_heads.append(mlp)

        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1, name=f"{name}_ln")

    def call(
        self,
        z_prev: tf.Tensor,  # (B, J, d)
        z_base: tf.Tensor,  # (B, J, d)
        avail: tf.Tensor,  # (B, J)
        training: bool = False,
    ) -> tf.Tensor:
        B = tf.shape(z_prev)[0]
        J = tf.shape(z_prev)[1]
        d = self.cfg.d_embed

        # Which representation feeds into φ
        if self.cfg.residual_variant == "fixed_base":
            z_phi_in = z_base  # (B, J, d)
        else:
            z_phi_in = z_prev  # (B, J, d)

        # Global context c: (B, d)
        c = masked_mean(z_prev, mask=avail, axis=1)  # (B, d)

        # Broadcast context to all items: (B, 1, d) -> (B, J, d)
        c_exp = tf.expand_dims(c, axis=1)  # (B, 1, d)
        c_exp = tf.tile(c_exp, [1, J, 1])  # (B, J, d)

        # φ inputs: concat [z_phi_in_j, c] -> (B, J, 2d)
        phi_inputs = tf.concat([z_phi_in, c_exp], axis=-1)

        # Flatten to apply MLPs
        phi_flat = tf.reshape(phi_inputs, [B * J, 2 * d])
        # each item is updated by a nonlinear function of its own embedding
        # plus the context summary.
        upd = 0.0
        for mlp in self.phi_heads:
            upd_flat = mlp(phi_flat, training=training)  # (B*J, d)
            upd_h = tf.reshape(upd_flat, [B, J, d])
            upd += upd_h

        upd = upd / float(self.cfg.n_heads)
        z_next = z_prev + upd
        z_next = self.layer_norm(z_next, training=training)
        return z_next

    """       In standard mode: no difference.
            In fixed_base mode: you’re using current layer representation to
            build context and base representation in φ inputs.
    """
