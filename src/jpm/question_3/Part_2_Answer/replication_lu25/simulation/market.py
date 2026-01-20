"""Single-market simulator for Lu & Shimizu (2025) Monte Carlo designs."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .config import SimConfig
from .dgp import generate_eta_alpha

tfd = tfp.distributions


def _simulate_shares_rc_logit(
    p: np.ndarray,
    w: np.ndarray,
    xi: np.ndarray,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate inside-good shares for one market under RC logit with an outside option."""
    # J = p.shape[0]

    p_tf = tf.convert_to_tensor(p, dtype=tf.float64)  # (J,)
    w_tf = tf.convert_to_tensor(w, dtype=tf.float64)
    xi_tf = tf.convert_to_tensor(xi, dtype=tf.float64)

    # Draw individual-specific price coefficients: beta_p_i ~ N(beta_p_star, sigma_star^2)
    beta_p_draws = tf.cast(
        tfd.Normal(cfg.beta_p_star, cfg.sigma_star).sample(
            cfg.R0, seed=int(rng.integers(1, 2**31 - 1))
        ),
        tf.float64,
    )  # (R0,)

    # mean utility part: beta_p_star * p + beta_w_star * w + xi
    delta = cfg.beta_p_star * p_tf + cfg.beta_w_star * w_tf + xi_tf  # (J,)

    # heterogeneous component: (beta_p_i - beta_p_star) * p
    mu = tf.expand_dims(beta_p_draws - cfg.beta_p_star, 1) * tf.expand_dims(
        p_tf, 0
    )  # (R0, J)

    util = tf.expand_dims(delta, 0) + mu  # (R0, J)
    expu = tf.exp(util)
    denom = 1.0 + tf.reduce_sum(expu, axis=1, keepdims=True)
    s_r = expu / denom
    s = tf.reduce_mean(s_r, axis=0)  # (J,)
    return s.numpy().astype(float)


def simulate_market(dgp: str, J: int, cfg: SimConfig, rng: np.random.Generator) -> dict:
    """Simulate a single market.

    Returns dict with keys:
        w, p, u, xi, eta, alpha, s
    """
    # Observed product characteristic w
    w = rng.uniform(cfg.w_low, cfg.w_high, size=J).astype(float)

    # Cost shock (valid IV in the simulation design)
    u = rng.normal(loc=0.0, scale=cfg.cost_sd, size=J).astype(float)

    # Demand shock components and endogenous price component
    eta_star, alpha_star = generate_eta_alpha(dgp, J, cfg, rng)

    # Demand shock level
    xi = (cfg.xi_bar_star + eta_star).astype(float)

    # Price equation (linear, with endogenous component alpha_star)
    p = (alpha_star + 0.3 * w + u).astype(float)

    # Shares
    s = _simulate_shares_rc_logit(p, w, xi, cfg, rng)

    return {
        "w": w,
        "p": p,
        "u": u,
        "xi": xi,
        "eta": eta_star.astype(float),
        "alpha": alpha_star.astype(float),
        "s": s,
    }
