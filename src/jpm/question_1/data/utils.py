import math
import re
from typing import Optional


# yf names to snake case
def title_to_snake(s: str) -> str:
    s = re.sub(r"[^0-9A-Za-z]+", " ", s)
    parts = s.lower().strip().split()
    return "_".join(parts)


# handle None values
def nz(x: Optional[float]) -> float:
    return 0.0 if x is None else float(x)


# return the first non-None value
def first_non_none(*args: Optional[float]) -> Optional[float]:
    for a in args:
        if a is not None:
            return a
    return None


def _rel_diff(a: float, b: float) -> float:
    """Relative difference |a-b| / max(1, |a|, |b|)."""
    return abs(a - b) / max(1.0, abs(a), abs(b))


def _pct(x: float, denom: float) -> float:
    return 0.0 if denom == 0 else x / denom


def _isfinite(x: float) -> bool:
    return x is not None and math.isfinite(float(x))


def _sign(x: float) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)
