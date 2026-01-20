"""Dataset simulator for Lu & Shimizu (2025) replication harness."""

from __future__ import annotations

import numpy as np

from jpm.config import SimConfig
from jpm.question_3.replication_lu25.simulation.market import simulate_market


def simulate_dataset(
    dgp: str, T: int, J: int, cfg: SimConfig, seed: int = 123
) -> list[dict]:
    """Simulate a list of T markets."""
    rng = np.random.default_rng(seed)
    markets = [simulate_market(dgp, J, cfg, rng) for _ in range(T)]
    return markets
