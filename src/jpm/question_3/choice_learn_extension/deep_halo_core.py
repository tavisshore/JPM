# src/jpm/question_3/choice_learn_ext/models/deep_context/model.py

from __future__ import annotations

from typing import Dict

import tensorflow as tf

from jpm.config import DeepHaloConfig
from jpm.question_3.choice_learn_extension.layers import BaseEncoder, HaloBlock
from jpm.question_3.choice_learn_extension.utils import masked_log_softmax


class DeepHalo(tf.keras.Model):
    """
    Deep context-dependent choice model.

    Inputs dict must contain:
        - "available": (B, J)  float32/bool
        - if cfg.featureless: "item_ids": (B, J) int32
        - else: "X": (B, J, d_x) float32
        - "choice": (B,) int32  (only needed for nll)
    """

    def __init__(self, cfg: DeepHaloConfig, name: str = "DeepHalo"):
        super().__init__(name=name)
        self.cfg = cfg

        self.base_encoder = BaseEncoder(cfg, name="base_encoder")
        self.blocks = [
            HaloBlock(cfg, name=f"halo_block_l{layer_idx}")
            for layer_idx in range(cfg.n_layers)
        ]
        self.beta = tf.keras.layers.Dense(1, use_bias=False, name="beta_final")

    def call(
        self, inputs: Dict[str, tf.Tensor], training: bool = False
    ) -> Dict[str, tf.Tensor]:
        avail = tf.cast(inputs["available"], tf.float32)  # (B, J)

        z0 = self.base_encoder(inputs, training=training)  # (B, J, d)
        z = z0

        for block in self.blocks:
            z = block(z_prev=z, z_base=z0, avail=avail, training=training)

        u = self.beta(z)  # (B, J, 1)
        u = tf.squeeze(u, axis=-1)  # (B, J)

        log_probs = masked_log_softmax(u, mask=avail, axis=1)

        return {"utilities": u, "log_probs": log_probs}

    def nll(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        Mean negative log-likelihood of observed choices.
        """
        outputs = self.call(inputs, training=training)
        logP = outputs["log_probs"]  # (B, J)
        choices = tf.cast(inputs["choice"], tf.int32)
        B = tf.shape(logP)[0]
        idx = tf.stack([tf.range(B, dtype=tf.int32), choices], axis=1)
        chosen_logp = tf.gather_nd(logP, idx)  # (B,)
        return -tf.reduce_mean(chosen_logp)


class DeepContextChoiceModel(DeepHalo):
    """
    Convenience wrapper used in tests and in the public wrapper.
    Allows configuring embedding dim, number of blocks, heads, and residual variant.
    """

    def __init__(
        self,
        num_items: int,
        d_embed: int = 16,
        n_blocks: int = 2,
        n_heads: int = 2,
        residual_variant: str = "standard",
        dropout: float = 0.0,
        name: str = "DeepContext",
    ):
        cfg = DeepHaloConfig(
            d_embed=d_embed,
            n_heads=n_heads,
            n_layers=n_blocks,
            residual_variant=residual_variant,
            featureless=True,
            vocab_size=num_items,
            dropout=dropout,
        )
        super().__init__(cfg, name=name)
