"""
Implementation of BLP estimation for a random-coefficients logit demand model.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def _simulate_shares_given_delta(delta, p, sigma, nu_draws):
    """
    Compute Random-coeff logit shares for one market, inside goods only.

    delta: [J] mean utilities
    p:     [J] prices
    sigma: scalar
    nu_draws: [R] standard normal draws for random coefficient
    """
    # mu_rj = sigma * nu_r * p_j
    mu = tf.expand_dims(nu_draws, 1) * tf.expand_dims(p, 0) * sigma  # [R, J]
    util = tf.expand_dims(delta, 0) + mu                              # [R, J]

    expu = tf.exp(util)
    denom = 1.0 + tf.reduce_sum(expu, axis=1, keepdims=True)          # outside option included
    s_r = expu / denom                                                # [R, J]
    return tf.reduce_mean(s_r, axis=0)                                # [J]


def invert_delta_contraction(s_obs, p, sigma, R=200, max_iter=2000, tol=1e-10, seed=123):
    """
    Berry contraction mapping to find delta s.t. model shares match observed shares.
    s_obs: [J] observed inside shares
    p:     [J]
    """
    s_obs = tf.convert_to_tensor(s_obs, dtype=tf.float64)
    p = tf.convert_to_tensor(p, dtype=tf.float64)
    sigma = tf.convert_to_tensor(sigma, dtype=tf.float64)

    # fixed simulation draws per inversion for stability
    nu_draws = tfd.Normal(0.0, 1.0).sample(R, seed=int(seed))
    nu_draws = tf.cast(nu_draws, tf.float64)

    # initialize delta with simple logit inversion (no random coeff):
    s0 = tf.maximum(1.0 - tf.reduce_sum(s_obs), 1e-12)
    delta = tf.math.log(tf.maximum(s_obs, 1e-12)) - tf.math.log(s0)

    for _ in range(max_iter):
        s_hat = _simulate_shares_given_delta(delta, p, sigma, nu_draws)
        delta_new = delta + (tf.math.log(tf.maximum(s_obs, 1e-12)) - tf.math.log(tf.maximum(s_hat, 1e-12)))

        if tf.reduce_max(tf.abs(delta_new - delta)) < tol:
            delta = delta_new
            break
        delta = delta_new

    return delta  # [J]


def build_matrices(markets, iv_type="cost"):
    """
    markets: list of dicts, each has keys: s, p, w, maybe u
    Returns stacked arrays across all markets/products: y=delta, X, Z, along with market indexing.
    """
    deltas, Xs, Zs = [], [], []

    for m in markets:
        # delta is computed later (depends on sigma), so we only prep X,Z pieces here
        p = m["p"]
        w = m["w"]
        u = m.get("u", None)

        # X = [1, p, w]  (intercept is the "Int" reported in tables) 
        X = np.column_stack([np.ones_like(p), p, w])

        if iv_type == "cost":
            if u is None:
                raise ValueError("iv_type='cost' requires market['u'] (cost shock).")
            Z = np.column_stack([np.ones_like(p), w, w**2, u, u**2])
        elif iv_type == "nocost":
            Z = np.column_stack([np.ones_like(p), w, w**2, w**3, w**4])  
        else:
            raise ValueError("iv_type must be 'cost' or 'nocost'.")

        Xs.append(X)
        Zs.append(Z)

    X = np.vstack(Xs)
    Z = np.vstack(Zs)
    return X, Z


def iv_2sls_beta(delta_vec, X, Z):
    """
    2SLS: beta = (X' Pz X)^(-1) X' Pz delta
    """
    delta_vec = tf.convert_to_tensor(delta_vec, dtype=tf.float64)  # [N]
    X = tf.convert_to_tensor(X, dtype=tf.float64)                  # [N,k]
    Z = tf.convert_to_tensor(Z, dtype=tf.float64)                  # [N,l]

    # Pz = Z (Z'Z)^(-1) Z'
    ZTZ_inv = tf.linalg.inv(tf.matmul(Z, Z, transpose_a=True))
    PzX = tf.matmul(Z, tf.matmul(ZTZ_inv, tf.matmul(Z, X, transpose_a=True)))  # [N,k] via Z(Z'Z)^-1 Z'X
    XTPzX = tf.matmul(X, PzX, transpose_a=True)                                  # [k,k]
    XTPzY = tf.matmul(X, tf.matmul(Z, tf.matmul(ZTZ_inv, tf.matmul(Z, delta_vec[:,None], transpose_a=True))),
                      transpose_a=True)                                          # [k,1]

    beta = tf.linalg.solve(XTPzX, XTPzY)[:, 0]  # [k]
    return beta


def gmm_objective_for_sigma(sigma, markets, iv_type="cost", R=200):
    """
    One-step GMM objective with W = (Z'Z)^(-1).
    """
    # 1) invert all markets to get delta stacked
    delta_list = []
    for t, m in enumerate(markets):
        delta_t = invert_delta_contraction(m["s"], m["p"], sigma, R=R, seed=123+t)
        delta_list.append(delta_t.numpy())
    delta_vec = np.concatenate(delta_list, axis=0)  # [N=T*J]

    # 2) build X,Z
    X, Z = build_matrices(markets, iv_type=iv_type)

    # 3) 2SLS beta, residual xi
    beta = iv_2sls_beta(delta_vec, X, Z).numpy()
    xi = delta_vec - X @ beta  # [N]

    # 4) moments g = (1/N) Z' xi
    N = xi.shape[0]
    g = (Z.T @ xi) / N

    # 5) W = (Z'Z)^(-1) (one-step)
    W = np.linalg.inv(Z.T @ Z / N)
    obj = float(g.T @ W @ g)
    return obj, beta


def estimate_blp_sigma(markets, iv_type="cost", sigma_init=1.0, R=200):
    """
    Estimate BLP sigma via GMM.
    Function implemented: σ̂ = argmin_σ Q(σ) where Q(σ) = g(σ)' W g(σ)


    :param markets: List of simulated markets, each with s, p, w, u
    :param iv_type: Instrumental variable type ("cost" or "nocost")
    :param sigma_init: Initial value for sigma
    :param R: Number of replications
    """
    def _tf_value_and_grad(sigma_tf):
        sigma_val = float(sigma_tf.numpy()[0])
        obj, _ = gmm_objective_for_sigma(sigma_val, markets, iv_type=iv_type, R=R)
        # This objective is not fully differentiable through the contraction/2SLS as written (numpy breaks graph),
        # so we provide a gradient-free approach by wrapping in finite differences for now.
        # We'll upgrade to a TF-native implementation once it's working end-to-end.
        return obj

    # search space for σ, upper bound of 4, and lower bound 0.05
    grid = np.linspace(0.05, 4.0, 40)
    best = None
    for s in grid:
        obj, beta = gmm_objective_for_sigma(s, markets, iv_type=iv_type, R=R)
        if (best is None) or (obj < best[0]):
            best = (obj, s, beta)

    # local refine around best sigma
    s0 = best[1]
    refine = np.linspace(max(0.01, s0 - 0.25), s0 + 0.25, 30)

    # further refine search
    for s in refine:
        obj, beta = gmm_objective_for_sigma(s, markets, iv_type=iv_type, R=R)
        if obj < best[0]:
            best = (obj, s, beta)

    obj_hat, sigma_hat, beta_hat = best
    return sigma_hat, beta_hat, obj_hat