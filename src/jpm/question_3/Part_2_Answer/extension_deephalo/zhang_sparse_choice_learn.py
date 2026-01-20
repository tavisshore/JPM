"""zhang_sparse_choice_learn.py

Zhang-Sparse (Lu25-aligned) estimator implemented using DeepHalo (Zhang 2025)
with a Lu(2025)-style sparse market×product shock layer.

Key idea
--------
DeepHalo provides context-dependent utilities u_halo(t,j).
Lu(2025) suggests unobserved shocks decompose as xi_{t j} = mu_t + d_{t j} with d sparse.

We implement:
u_{t j} = u_halo(t,j) + 1{j!=0} * ( mu_t + d_{t j} )

and estimate by MAP:
minimize mean-NLL
  + lambda * mean(|d|)
  + (1/(2*mu_sd^2)) * mean(mu^2)

This file is intentionally "repo-consistent":
- It does NOT depend on choice-learn's internal BaseModel API.
- It uses ChoiceDataset as a data container, then trains manually.

Run
---
python choice_learn_extension/zhang_sparse_choice_learn.py
"""

from __future__ import annotations

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf

from choice_learn.data import ChoiceDataset


# -----------------------------------------------------------------------------
# Import DeepHalo
# -----------------------------------------------------------------------------

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

try:
    from deep_halo_core import DeepHalo
    from config import DeepHaloConfig
except ImportError:
    print("Warning: DeepHalo not found, using placeholder.")

    class DeepHaloConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class DeepHalo(tf.keras.Model):
        def __init__(self, config):
            super().__init__()
            self.embedding = tf.keras.layers.Embedding(
                config.vocab_size, config.d_embed
            )
            self.dense = tf.keras.layers.Dense(1)

        def call(self, inputs, training=False):
            emb = self.embedding(inputs["item_ids"])
            u = tf.squeeze(self.dense(emb), -1)
            return {"utilities": u}

# -----------------------------------------------------------------------------
# Conditional Logit baseline
# -----------------------------------------------------------------------------

class ConditionalLogit(tf.keras.Model):
    def __init__(self, num_items: int):
        super().__init__()
        self.beta = tf.Variable(tf.zeros([num_items], tf.float32))

    def call(self, inputs):
        B = tf.shape(inputs["item_ids"])[0]
        u = tf.tile(self.beta[None, :], [B, 1])
        avail = tf.cast(inputs["available"], tf.float32)
        u_masked = tf.where(avail > 0.5, u, -1e9)
        log_probs = tf.nn.log_softmax(u_masked, axis=1)
        return log_probs

    def nll(self, inputs):
        logP = self.call(inputs)
        y = tf.cast(inputs["choice"], tf.int32)
        idx = tf.stack([tf.range(tf.shape(y)[0]), y], axis=1)
        return -tf.reduce_mean(tf.gather_nd(logP, idx))

# -----------------------------------------------------------------------------
# Zhang-Sparse configuration
# -----------------------------------------------------------------------------

@dataclass
class ZhangSparseConfig:
    d_embed: int = 16
    n_blocks: int = 2
    n_heads: int = 2
    residual_variant: str = "fixed_base"
    dropout: float = 0.0

    lr: float = 5e-4
    batch_size: int = 256
    epochs: int = 10

    l1_strength: float = 5.0
    mu_sd: float = 2.0
    tau_detect: float = 0.25

    seed: int = 123

# -----------------------------------------------------------------------------
# Zhang-Sparse DeepHalo model
# -----------------------------------------------------------------------------

class ZhangSparseDeepHalo(tf.keras.Model):
    def __init__(self, num_items: int, T: int, J_inside: int, cfg: ZhangSparseConfig):
        super().__init__()
        self.cfg = cfg
        self.J_inside = J_inside
        self.T = T

        halo_cfg = DeepHaloConfig(
            d_embed=cfg.d_embed,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_blocks,
            residual_variant=cfg.residual_variant,
            featureless=True,
            vocab_size=num_items,
            dropout=cfg.dropout,
        )
        self.halo = DeepHalo(halo_cfg)

        self.mu = tf.Variable(tf.zeros([T], tf.float32))
        self.d = tf.Variable(tf.zeros([T, J_inside], tf.float32))

    def call(self, inputs, training=False):
        u = self.halo(inputs, training=training)["utilities"]  # (B, n_items)

        market_id = tf.reshape(
            tf.cast(inputs["market_id"], tf.int32), [-1]
        )

        mu_t = tf.gather(self.mu, market_id)          # (B,)
        d_t = tf.gather(self.d, market_id)            # (B, J_inside)

        # center d within market (recommended)
        d_t = d_t - tf.reduce_mean(d_t, axis=1, keepdims=True)

        d_pad = tf.concat(
            [tf.zeros_like(d_t[:, :1]), d_t], axis=1
        )

        inside_mask = tf.concat(
            [
                tf.zeros([tf.shape(u)[0], 1], u.dtype),
                tf.ones([tf.shape(u)[0], tf.shape(u)[1] - 1], u.dtype),
            ],
            axis=1,
        )

        u_aug = u + inside_mask * mu_t[:, None] + d_pad

        avail = tf.cast(inputs["available"], tf.float32)
        u_masked = tf.where(avail > 0.5, u_aug, -1e9)
        log_probs = tf.nn.log_softmax(u_masked, axis=1)

        return log_probs

    def nll(self, inputs, training=False):
        logP = self.call(inputs, training=training)
        y = tf.cast(inputs["choice"], tf.int32)
        idx = tf.stack([tf.range(tf.shape(y)[0]), y], axis=1)
        return -tf.reduce_mean(tf.gather_nd(logP, idx))

    def map_objective(self, inputs, training=False):
        nll = self.nll(inputs, training)
        l1 = self.cfg.l1_strength * tf.reduce_mean(tf.abs(self.d))
        mu_ridge = 0.5 * tf.reduce_mean(tf.square(self.mu)) / (self.cfg.mu_sd ** 2)
        return nll + l1 + mu_ridge

# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------

class ZhangSparseTrainer:
    def __init__(self, model, lr):
        self.model = model
        self.opt = tf.keras.optimizers.legacy.Adam(lr)

    @tf.function
    def train_step(self, batch):
        with tf.GradientTape() as tape:
            loss = self.model.map_objective(batch, training=True)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def fit(self, data, batch_size, epochs):
        N = len(data["choice"])
        idx = np.arange(N)
        for ep in range(1, epochs + 1):
            np.random.shuffle(idx)
            losses = []
            for s in range(0, N, batch_size):
                b = idx[s:s + batch_size]
                batch = {k: tf.convert_to_tensor(v[b]) for k, v in data.items()}
                losses.append(float(self.train_step(batch)))
            print(f"Epoch {ep:03d} | MAP loss {np.mean(losses):.4f}")

# -----------------------------------------------------------------------------
# Dataset conversion
# -----------------------------------------------------------------------------

def choice_dataset_to_tensors(dataset: ChoiceDataset, n_items: int):
    # In some choice-learn versions, shared_features_by_choice can be a tuple
    shared = dataset.shared_features_by_choice
    if isinstance(shared, (tuple, list)):
        shared = shared[0]  # usually the actual array is the first element

    shared = np.asarray(shared)
    choices = np.asarray(dataset.choices)

    N = choices.shape[0]

    # market_id from shared feature column 0
    if shared.ndim == 1:
        market_id = shared.astype(np.int32)
    else:
        market_id = shared[:, 0].astype(np.int32)

    return {
        "item_ids": np.tile(np.arange(n_items, dtype=np.int32)[None, :], (N, 1)),
        "available": np.ones((N, n_items), dtype=np.float32),
        "choice": choices.astype(np.int32).reshape(-1),
        "market_id": market_id.reshape(-1),
    }


# -----------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------

def simulate(T, J_inside, N_t, seed):
    rng = np.random.default_rng(seed)
    n_items = J_inside + 1
    N = T * N_t

    market_id = np.repeat(np.arange(T), N_t)
    item_ids = np.tile(np.arange(n_items)[None, :], (N, 1))
    available = np.ones((N, n_items), np.float32)

    halo = DeepHalo(
        DeepHaloConfig(
            d_embed=16,
            n_heads=2,
            n_layers=2,
            residual_variant="fixed_base",
            featureless=True,
            vocab_size=n_items,
            dropout=0.0,
        )
    )

    u_halo = halo(
        {"item_ids": tf.constant(item_ids), "available": tf.constant(available)},
        training=False,
    )["utilities"].numpy()

    mu_true = rng.normal(0, 2.0, T)
    d_true = np.zeros((T, J_inside))
    for t in range(T):
        idx = rng.choice(J_inside, J_inside // 2, replace=False)
        d_true[t, idx] = rng.normal(0, 1.0, len(idx))

    d_pad = np.concatenate([np.zeros((T, 1)), d_true], axis=1)
    u = u_halo + mu_true[market_id][:, None] + d_pad[market_id]

    P = np.exp(u - u.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    y = np.array([rng.choice(n_items, p=P[i]) for i in range(N)])

    dataset = ChoiceDataset(
        shared_features_by_choice=market_id[:, None],
        items_features_by_choice=np.zeros((N, n_items, 1)),
        choices=y,
    )

    return dataset

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    np.random.seed(123)
    tf.random.set_seed(123)

    T, J_inside, N_t = 25, 15, 1000
    dataset = simulate(T, J_inside, N_t, seed=123)
    n_items = J_inside + 1

    data = choice_dataset_to_tensors(dataset, n_items)

    cfg = ZhangSparseConfig()
    model = ZhangSparseDeepHalo(n_items, T, J_inside, cfg)
    trainer = ZhangSparseTrainer(model, cfg.lr)

    trainer.fit(data, cfg.batch_size, cfg.epochs)

    print("Done.")

if __name__ == "__main__":
    main()
