from dataclasses import dataclass


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
