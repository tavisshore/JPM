"""
Configuration object for the DeepHalo (Deep Context-Dependent Choice) model.
This dataclass specifies all architectural and training-related hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class DeepHaloConfig:
    # Embedding dimension for item representations
    d_embed: int = 32

    # Number of phi-interaction heads and stacked halo layers
    n_heads: int = 4
    n_layers: int = 2

    # Residual behavior:
    #   "standard":  phi receives z^{l-1}
    #   "fixed_base": phi receives z^{0}
    residual_variant: str = "standard"

    # Dropout rate for phi MLPs
    dropout: float = 0.0

    # Input mode:
    #   featureless=True  → item_ids are embedded using vocab_size
    #   featureless=False → raw item features X with dimension d_x
    featureless: bool = True
    vocab_size: Optional[int] = None

    # Required if featureless=False (feature-based items)
    d_x: Optional[int] = None


@dataclass(frozen=True)
class SimConfig:
    default_market_size: int = (
        1000  # Paper Section 4 uses N = 1000 consumers per market
    )

    # Structural parameters (paper Section 4.1)
    beta_p_star: float = -1.0  # mean price sensitivity
    beta_w_star: float = 0.5
    sigma_star: float = 1.5  # heterogeneity in price sensitivity
    xi_bar_star: float = -1.0

    # Market structure
    Nt: int = 1000  # true number of consumers per market
    R0: int = 200  # monte carlo size for integration

    # Product characteristics
    w_low: float = 1.0
    w_high: float = 2.0

    # Cost shock
    cost_sd: float = 0.7

    # Sparsity design
    sparse_frac: float = 0.4  # first 40% products non-zero
    eta_sparse_vals: tuple = (1.0, -1.0)

    # Approximate sparsity (DGP3/4)
    eta_dense_sd: float = 1.0 / 3.0

    # Endogeneity strength
    alpha_scale: float = 0.3


@dataclass
class StudyConfig:
    """Monte Carlo study config (grid + estimator settings)."""

    # Monte Carlo reps
    R_mc: int = 50

    # Base seed: replication r uses seed + r
    seed: int = 123

    # Shrinkage settings (must match what you used in your validated MC script)
    shrink_n_iter: int = 200
    shrink_burn: int = 100
    shrink_v0: float = 1e-4
    shrink_v1: float = 1.0

    # Optional Lu25 MAP settings (only used if HAS_LU25_MAP)
    lu_steps: int = 1200
    lu_lr: float = 0.05
    lu_l1_strength: float = 8.0
    lu_mu_sd: float = 2.0
    lu_tau_detect: float = 0.25
