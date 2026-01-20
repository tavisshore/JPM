"""DGP helpers for Lu & Shimizu (2025) style simulations.

We generate two product-level latent vectors for each market:

- eta_star[j]: deviation component of the demand shock (sparse in DGP1/2, dense in DGP3/4)
- alpha_star[j]: endogenous price component correlated with eta_star (strength controlled by cfg.alpha_scale)

These correspond to the paper's simulation design where endogeneity comes from correlation between
price and the demand shock via the shared component alpha_star(eta_star).
"""

from __future__ import annotations

import numpy as np

from .config import SimConfig


def generate_eta_alpha(
    dgp: str, J: int, cfg: SimConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (eta_star, alpha_star) for a single market.

    DGP definitions (aligned with the repo's config fields):

    - DGP1: sparse eta, no endogeneity (alpha = 0)
    - DGP2: sparse eta, endogenous prices (alpha proportional to sign(eta))
    - DGP3: dense eta ~ N(0, eta_dense_sd^2), no endogeneity (alpha = 0)
    - DGP4: dense eta, endogenous prices (alpha nonzero for large |eta|)

    Returns:
        eta_star: (J,) float array
        alpha_star: (J,) float array
    """
    dgp = dgp.upper()

    if dgp in ("DGP1", "DGP2"):
        # Sparse eta: first fraction are +/- values, rest are exactly 0
        eta = np.zeros(J, dtype=float)
        cutoff = int(np.floor(cfg.sparse_frac * J))
        if cutoff > 0:
            vals = np.asarray(cfg.eta_sparse_vals, dtype=float)
            # random signs among the non-zero set
            eta[:cutoff] = rng.choice(vals, size=cutoff, replace=True)

        alpha = np.zeros(J, dtype=float)
        if dgp == "DGP2":
            # correlated endogenous component
            alpha = cfg.alpha_scale * np.sign(eta)

    elif dgp in ("DGP3", "DGP4"):
        eta = rng.normal(loc=0.0, scale=cfg.eta_dense_sd, size=J).astype(float)

        alpha = np.zeros(J, dtype=float)
        if dgp == "DGP4":
            # only large eta values induce endogeneity
            alpha[eta >= cfg.eta_dense_sd] = cfg.alpha_scale
            alpha[eta <= -cfg.eta_dense_sd] = -cfg.alpha_scale
    else:
        raise ValueError(f"Unknown DGP: {dgp}")

    return eta, alpha
