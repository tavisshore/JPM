# Quick test for BLP estimation on simulated data based on DGP1 from the BLP paper.

from types import SimpleNamespace

import estimators.blp as blp
import numpy as np

# 1. Setup Configuration based on Paper (Section 4.1)
cfg = SimpleNamespace(
    T=25,  # Number of markets
    J=10,  # Number of products
    beta_p_true=-1.0,  #
    beta_w_true=0.5,  #
    sigma_true=1.5,  #
    xi_bar=-1.0,  # Mean demand shock
    R_sim=1000,  # Consumer draws for share simulation
)


def generate_test_data(cfg):
    markets = []
    # Fixed draws for share simulation
    nu_i = np.random.normal(0, 1, cfg.R_sim)

    for _ in range(cfg.T):
        # Exogenous characteristic w ~ U(1, 2)
        w = np.random.uniform(1, 2, cfg.J)

        # Cost shock u ~ N(0, 0.7^2)
        u = np.random.normal(0, 0.7, cfg.J)

        # Price p = 0.3w + u (DGP1 has alpha=0)
        p = 0.3 * w + u

        # Demand shock xi (simplified DGP1: no sparse eta for basic test)
        xi = np.full(cfg.J, cfg.xi_bar)

        # Compute Mean Utility: delta = xi + beta_w*w + beta_p*p
        delta = xi + cfg.beta_w_true * w + cfg.beta_p_true * p

        # Simulate observed shares
        # u_ijt = delta_jt + sigma * nu_i * p_jt + epsilon_ijt
        mu_ij = cfg.sigma_true * np.outer(nu_i, p)  # [R, J]
        util = delta + mu_ij  # [R, J]
        exp_u = np.exp(util)
        shares_r = exp_u / (1 + np.sum(exp_u, axis=1, keepdims=True))
        s_obs = np.mean(shares_r, axis=0)

        markets.append({"s": s_obs, "p": p, "w": w, "u": u})
    return markets


# 2. Run the Test
test_markets = generate_test_data(cfg)
print("Starting BLP Estimation...")

sigma_hat, beta_hat, obj = blp.estimate_blp_sigma(test_markets, iv_type="cost", R=200)

# 3. Evaluate Results
print("\n--- Recovery Results ---")
print("Parameter | True Value | Estimated | Bias")
print(
    f"Sigma     | {cfg.sigma_true:10.3f} | {sigma_hat:9.3f} | {sigma_hat - cfg.sigma_true:.3f}"
)
print(
    f"Intercept | {cfg.xi_bar:10.3f} | {beta_hat[0]:9.3f} | {beta_hat[0] - cfg.xi_bar:.3f}"
)
print(
    f"Beta_p    | {cfg.beta_p_true:10.3f} | {beta_hat[1]:9.3f} | {beta_hat[1] - cfg.beta_p_true:.3f}"
)
print(
    f"Beta_w    | {cfg.beta_w_true:10.3f} | {beta_hat[2]:9.3f} | {beta_hat[2] - cfg.beta_w_true:.3f}"
)
