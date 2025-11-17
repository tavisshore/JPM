from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class Metric:
    value: float
    mae: float
    pct: float


@dataclass
class TickerResults:
    """
    Per-ticker results:
    - aggregated sections: assets / liabilities / equity
    - per-feature metrics.
    """

    assets: Metric
    liabilities: Metric
    equity: Metric
    features: Dict[str, Metric]

    def feature_values(self) -> Dict[str, float]:
        return {name: m.value for name, m in self.features.items()}
