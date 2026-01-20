"""
Lu & Shimizu (2025) style estimator (closer-to-paper approximation)

Key differences vs classic BLP:
- Treats market-product shocks xi_{jt} as parameters (no IV requirement)
- Imposes a sparsity structure by decomposing xi_{jt} = mu_t + d_{jt}
  and shrinking deviations d_{jt} toward 0 (many products share the same
  market-level baseline mu_t).

This implementation performs MAP estimation:
    maximize  log p(data | beta, sigma, mu, d) + log p(mu) + log p(d)

Data likelihood:
- Uses a multinomial pseudo-likelihood on aggregate market shares.
  If a market has size N_t and observed shares s_obs, we construct counts
  y_j = round(N_t * s_obs_j), y_0 = N_t - sum_j y_j.
  Then:
      log L_t = sum_j y_j log s_hat_j + y_0 log s_hat_0
  where s_hat is the model-predicted RC logit shares under (beta, sigma, xi).

Priors (MAP penalties):
- mu_t ~ Normal(0, mu_sd^2)     (ridge penalty)
- d_{jt} ~ Laplace(0, b)        (L1 penalty; encourages sparsity of deviations)

Notes:
- This is a MAP (penalized likelihood) approximation to Bayesian shrinkage.
- It does NOT use BLP inversion or IVs.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def _simulate_shares_given_delta(
    delta: tf.Tensor,
    p: tf.Tensor,
    sigma: tf.Tensor,
    nu_draws: tf.Tensor,
) -> tf.Tensor:
    """Random-coeff logit shares for one market, inside goods only.

    Args:
        delta: [J] mean utility of inside goods.
        p: [J] prices.
        sigma: scalar random-coefficient std dev.
        nu_draws: [R] standard normal draws.

    Returns:
        s_hat: [J] predicted inside shares.
    """
    mu = tf.expand_dims(nu_draws, 1) * tf.expand_dims(p, 0) * sigma  # [R, J]
    util = tf.expand_dims(delta, 0) + mu  # [R, J]
    expu = tf.exp(util)
    denom = 1.0 + tf.reduce_sum(expu, axis=1, keepdims=True)  # include outside option
    s_r = expu / denom
    return tf.reduce_mean(s_r, axis=0)  # [J]


@dataclass
class Lu25MapConfig:
    # Monte Carlo draws for integrating random coefficients
    R: int = 200

    # Optimizer
    steps: int = 1200
    lr: float = 0.05
    seed: int = 123

    # Sparsity strength on deviations d (L1 penalty weight)
    # Larger => more shrinkage => more d's driven toward 0
    l1_strength: float = 8.0

    # Prior strength on mu_t (market baseline shocks)
    mu_sd: float = 2.0

    # Numerical stability
    eps: float = 1e-12

    # Pseudo market size for multinomial likelihood (if market dict has no N)
    default_market_size: int = 5000

    # Detection threshold for deviations (MAP has no inclusion probabilities)
    tau_detect: float = 0.25


def estimate_lu25_map(markets: List[Dict], cfg: Lu25MapConfig | None = None) -> Dict:
    """MAP estimator for (beta, sigma, mu_t, d_{jt}) using aggregate shares likelihood.

    Args:
        markets: list of dicts, each with keys:
            - 's': [J] observed inside shares
            - 'p': [J] prices
            - 'w': [J] observed characteristic
            - optional 'N': market size (int)
        cfg: configuration

    Returns:
        dict with sigma_hat, beta_hat, mu_hat, d_hat, xi_hat, gamma_hat and diagnostics.
    """
    if cfg is None:
        cfg = Lu25MapConfig()

    tf.random.set_seed(cfg.seed)

    T = len(markets)
    if T == 0:
        raise ValueError("markets is empty")

    J_list = [len(m["p"]) for m in markets]
    if len(set(J_list)) != 1:
        raise ValueError("This implementation assumes fixed J across markets.")
    J = J_list[0]

    # Build per-market tensors (lists first)
    s_obs_list: List[tf.Tensor] = []
    p_list: List[tf.Tensor] = []
    w_list: List[tf.Tensor] = []
    N_list: List[int] = []

    for m in markets:
        s_obs_list.append(tf.convert_to_tensor(m["s"], dtype=tf.float64))
        p_list.append(tf.convert_to_tensor(m["p"], dtype=tf.float64))
        w_list.append(tf.convert_to_tensor(m["w"], dtype=tf.float64))
        N_list.append(int(m.get("N", cfg.default_market_size)))

    # Pre-draw nu per market (market-specific seed)
    nu_draws_list: List[tf.Tensor] = []
    for t in range(T):
        nu = tfd.Normal(0.0, 1.0).sample(cfg.R, seed=cfg.seed + t)
        nu_draws_list.append(tf.cast(nu, tf.float64))

    # Stack lists into tensors for tf.function-safe indexing
    S_obs = tf.stack(s_obs_list, axis=0)  # [T, J]
    P = tf.stack(p_list, axis=0)  # [T, J]
    W = tf.stack(w_list, axis=0)  # [T, J]
    NU = tf.stack(nu_draws_list, axis=0)  # [T, R]
    N_tf = tf.constant(N_list, dtype=tf.float64)  # [T]

    # Parameters
    # beta = [intercept, beta_p, beta_w]
    beta = tf.Variable(tf.zeros([3], dtype=tf.float64), name="beta")
    log_sigma = tf.Variable(
        tf.math.log(tf.constant(1.0, dtype=tf.float64)), name="log_sigma"
    )

    mu = tf.Variable(tf.zeros([T], dtype=tf.float64), name="mu")  # market baselines
    d = tf.Variable(tf.zeros([T, J], dtype=tf.float64), name="d")  # deviations

    # Initialize beta via crude static logit inversion using average shares
    s_bar = tf.reduce_mean(S_obs, axis=0)  # [J]
    outside = tf.maximum(1.0 - tf.reduce_sum(s_bar), cfg.eps)
    delta_init = tf.math.log(tf.maximum(s_bar, cfg.eps)) - tf.math.log(outside)

    X_init = tf.stack(
        [
            tf.ones([J], tf.float64),
            tf.reduce_mean(P, axis=0),
            tf.reduce_mean(W, axis=0),
        ],
        axis=1,
    )  # [J, 3]

    beta_init = tf.linalg.lstsq(X_init, tf.expand_dims(delta_init, 1), fast=False)[:, 0]
    beta.assign(beta_init)
    log_sigma.assign(tf.math.log(tf.constant(1.0, dtype=tf.float64)))

    opt = tf.keras.optimizers.Adam(learning_rate=cfg.lr)

    history: List[Dict] = []

    # Graph-safe step: take tensors as explicit inputs
    @tf.function
    def step(P_in, W_in, S_in, NU_in, N_in):
        with tf.GradientTape() as tape:
            sigma = tf.exp(log_sigma)

            ll = tf.constant(0.0, dtype=tf.float64)

            for t in tf.range(T):
                p_t = P_in[t]
                w_t = W_in[t]
                s_obs_t = S_in[t]
                nu_t = NU_in[t]
                N_t = N_in[t]

                X_t = tf.stack([tf.ones([J], tf.float64), p_t, w_t], axis=1)  # [J, 3]
                delta_t = tf.linalg.matvec(X_t, beta) + mu[t] + d[t]

                s_hat_t = _simulate_shares_given_delta(delta_t, p_t, sigma, nu_t)
                s0_hat = tf.maximum(1.0 - tf.reduce_sum(s_hat_t), cfg.eps)

                # Pseudo counts
                y_t = tf.cast(tf.round(N_t * s_obs_t), tf.float64)
                y0_t = tf.maximum(N_t - tf.reduce_sum(y_t), 0.0)

                ll += tf.reduce_sum(y_t * tf.math.log(tf.maximum(s_hat_t, cfg.eps)))
                ll += y0_t * tf.math.log(s0_hat)

            # MAP penalties
            mu_pen = 0.5 * tf.reduce_sum((mu / cfg.mu_sd) ** 2)
            d_pen = cfg.l1_strength * tf.reduce_sum(tf.abs(d))

            neg_obj = -ll + mu_pen + d_pen

        grads = tape.gradient(neg_obj, [beta, log_sigma, mu, d])
        opt.apply_gradients(zip(grads, [beta, log_sigma, mu, d], strict=True))
        return neg_obj, ll, mu_pen, d_pen, tf.exp(log_sigma)

    for it in range(cfg.steps):
        neg_obj, ll, mu_pen, d_pen, sigma_val = step(P, W, S_obs, NU, N_tf)
        if (it % 50) == 0 or it == cfg.steps - 1:
            history.append(
                {
                    "iter": int(it),
                    "neg_log_post": float(neg_obj.numpy()),
                    "loglik": float(ll.numpy()),
                    "mu_pen": float(mu_pen.numpy()),
                    "d_pen": float(d_pen.numpy()),
                    "sigma": float(sigma_val.numpy()),
                }
            )

    # Extract estimates
    sigma_hat = float(tf.exp(log_sigma).numpy())
    beta_hat = beta.numpy().astype(float)

    mu_hat = mu.numpy().astype(float)  # [T]
    d_hat_mat = d.numpy().astype(float)  # [T, J]
    d_hat = d_hat_mat.reshape(-1)  # [T*J]
    xi_hat = (mu_hat[:, None] + d_hat_mat).reshape(-1)  # [T*J]

    # Sparsity diagnostics and detection
    d_abs = np.abs(d_hat)
    sparsity_1e2 = float(np.mean(d_abs < 1e-2))
    sparsity_1e3 = float(np.mean(d_abs < 1e-3))

    tau = float(cfg.tau_detect)
    gamma_hat = (d_abs > tau).astype(int)  # MAP detection proxy

    return {
        "sigma_hat": sigma_hat,
        "beta_hat": beta_hat,
        "mu_hat": mu_hat,
        "d_hat": d_hat,
        "xi_hat": xi_hat,
        "gamma_hat": gamma_hat,
        "history": history,
        "final_neg_log_post": history[-1]["neg_log_post"] if history else None,
        "sparsity": {
            "tau_for_detection": tau,
            "frac_|d|<1e-2": sparsity_1e2,
            "frac_|d|<1e-3": sparsity_1e3,
            "frac_detected_|d|>tau": float(np.mean(gamma_hat)),
        },
        "config": cfg.__dict__,
    }
