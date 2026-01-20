# shrinkage.py
import numpy as np
from estimators.blp import build_matrices, invert_delta_contraction


def _log_norm_pdf(x, var):
    # log N(x;0,var)
    return -0.5 * (np.log(2 * np.pi * var) + (x * x) / var)


def shrinkage_fit_beta_given_sigma(
    delta_vec,
    X,
    n_iter=200,
    burn=100,
    v0=0.05,  # spike variance (tuned to typical inversion noise scale)
    v1=1.0,  # slab variance (large)
    a_pi=1.0,
    b_pi=9.0,  # prior mean pi ~ 0.1 (sparse)
    beta_var=1e6,  # weak prior on beta
    seed=123,
):
    """
    Bayesian sparse-errors regression:
        delta = X beta + xi
        xi_n ~ N(0, v0) if gamma_n=0 else N(0,v1)
        gamma_n ~ Bern(pi), pi ~ Beta(a_pi,b_pi)

    Returns:
      beta_mean: posterior mean of beta after burn-in
      gamma_prob: posterior inclusion probabilities (mean gamma)
      score: average log posterior (rough diagnostic)
    """
    rng = np.random.default_rng(seed)
    N, k = X.shape

    # initialize
    beta = np.linalg.lstsq(X, delta_vec, rcond=None)[0]
    gamma = np.zeros(N, dtype=int)
    pi = a_pi / (a_pi + b_pi)

    beta_sum = np.zeros(k)
    gamma_sum = np.zeros(N)
    score_sum = 0.0
    kept = 0

    identity_k = np.eye(k)

    for it in range(n_iter):
        # residuals
        r = delta_vec - X @ beta

        # sample gamma_n independently
        # log odds for slab vs spike
        logp1 = np.log(pi + 1e-12) + _log_norm_pdf(r, v1)
        logp0 = np.log(1 - pi + 1e-12) + _log_norm_pdf(r, v0)
        m = np.maximum(logp1, logp0)
        p1 = np.exp(logp1 - m) / (np.exp(logp1 - m) + np.exp(logp0 - m))
        gamma = (rng.uniform(size=N) < p1).astype(int)

        # sample pi | gamma
        s = gamma.sum()
        pi = rng.beta(a_pi + s, b_pi + (N - s))

        # sample beta | gamma (weighted Gaussian regression)
        v = np.where(gamma == 1, v1, v0)  # variance per obs
        w = 1.0 / v  # precision weights
        Xw = X * w[:, None]
        Prec = (X.T @ Xw) + (1.0 / beta_var) * identity_k  # kxk
        mean = np.linalg.solve(Prec, X.T @ (w * delta_vec))
        cov = np.linalg.inv(Prec)

        beta = rng.multivariate_normal(mean, cov)

        # recompute residual after beta update for diagnostics
        r = delta_vec - X @ beta

        # optional scoring (rough): log p(r|gamma,pi) + log p(gamma|pi)
        # (skip constants from beta prior/cov, just diagnostic)
        loglik = _log_norm_pdf(r, v).sum()
        logprior_g = (
            gamma * np.log(pi + 1e-12) + (1 - gamma) * np.log(1 - pi + 1e-12)
        ).sum()
        score = loglik + logprior_g

        if it >= burn:
            beta_sum += beta
            gamma_sum += gamma
            score_sum += score
            kept += 1

    beta_mean = beta_sum / max(1, kept)
    gamma_prob = gamma_sum / max(1, kept)
    score_mean = score_sum / max(1, kept)
    return beta_mean, gamma_prob, float(score_mean)


def shrinkage_objective_for_sigma(sigma, markets, R=200, **kwargs):
    # 1) invert all markets to get delta_vec(sigma)
    delta_list = []
    for t, m in enumerate(markets):
        delta_t = invert_delta_contraction(m["s"], m["p"], sigma, R=R, seed=123 + t)
        delta_list.append(delta_t.numpy())
    delta_vec = np.concatenate(delta_list, axis=0)

    # 2) build X only (no Z needed)
    X, _ = build_matrices(markets, iv_type="nocost")  # X is same regardless
    beta_hat, gamma_prob, score = shrinkage_fit_beta_given_sigma(delta_vec, X, **kwargs)

    # We *maximize* score, so return it
    return score, beta_hat, gamma_prob


def estimate_shrinkage_sigma(markets, R=200, sigma_grid=None, **kwargs):
    if sigma_grid is None:
        sigma_grid = np.linspace(0.05, 4.0, 40)

    best = None
    for s in sigma_grid:
        score, beta_hat, gamma_prob = shrinkage_objective_for_sigma(
            s, markets, R=R, **kwargs
        )
        if (best is None) or (score > best[0]):
            best = (score, s, beta_hat, gamma_prob)

    # local refine
    s0 = best[1]
    refine = np.linspace(max(0.01, s0 - 0.25), s0 + 0.25, 25)
    for s in refine:
        score, beta_hat, gamma_prob = shrinkage_objective_for_sigma(
            s, markets, R=R, **kwargs
        )
        if score > best[0]:
            best = (score, s, beta_hat, gamma_prob)

    score_hat, sigma_hat, beta_hat, gamma_prob = best
    return sigma_hat, beta_hat, score_hat, gamma_prob
