from __future__ import annotations

import math
import random
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf

Nested = Dict[str, Any] | List[str]


def find_subtree(d: Nested, target: str) -> Nested | None:
    """Locate the subtree whose key == target."""
    if isinstance(d, list):
        return None

    # Depth-first scan until we bump into the target key
    for k, v in d.items():
        if k == target:
            return v
        if isinstance(v, dict):
            out = find_subtree(v, target)
            if out is not None:
                return out

    return None


def collect_leaves(d: Nested) -> List[str]:
    """Return all leaf strings under a nested dict/list."""
    if isinstance(d, list):
        return d

    out = []
    # Flatten any nested mix of dicts and lists
    for v in d.values():
        out.extend(collect_leaves(v))
    return out


def get_leaf_values(d, sub_key: str | None = None) -> List[str]:
    """Return leaf strings, optionally from a subtree keyed by sub_key."""
    if sub_key:
        subtree = find_subtree(d, sub_key)
        if subtree is None:
            return []
        return collect_leaves(subtree)

    # Default: collect every terminal value in the nested structure
    return collect_leaves(d)


def get_leaf_keys(d: Nested) -> List[str]:
    """Return all leaf keys under a nested dict."""
    if isinstance(d, list):
        return []

    out = []
    for k, v in d.items():
        if isinstance(v, dict):
            out.extend(get_leaf_keys(v))
        elif isinstance(v, list):
            out.append(k)
    return out


def get_leaf_paths(d: Nested, prefix: tuple = ()) -> List[tuple]:
    """Return all paths from root to leaf values as tuples of keys."""
    if isinstance(d, list):
        return []

    out = []
    for k, v in d.items():
        current_path = prefix + (k,)
        if isinstance(v, dict):
            out.extend(get_leaf_paths(v, current_path))
        elif isinstance(v, list):
            out.append(current_path)
    return out


def to_tensor(x) -> tf.Tensor:
    """Convert input to a float32 tensor."""
    # Centralised conversion keeps dtype consistent
    return tf.convert_to_tensor(x, dtype=tf.float32)


def tf_sum(xs) -> tf.Tensor:
    """Sum a list of tensors."""
    # Prefer tf.add_n for performance on stacked tensors
    return tf.add_n(xs)


def set_seed(seed: int) -> None:
    """Set seeds across Python, NumPy, and TensorFlow for reproducibility."""
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def coerce_float(x) -> float:
    """Safely convert to float, returning 0.0 on failure or NaN."""
    if pd.isna(x):
        return 0.0
    try:
        return float(x)
    except Exception:
        return 0.0


def as_series(mapping: Dict[int, float], years) -> pd.Series:
    """Create a float64 series from a mapping over the provided index."""
    # Keeps ordering stable even when mapping misses some periods
    return pd.Series(
        [mapping.get(y, math.nan) for y in years], index=years, dtype="float64"
    )


def errs_below_tol(errs: Dict[str, tf.Tensor], tol: float = 1e-4) -> tf.Tensor:
    """Check all tensors are below a tolerance."""
    tol_t = tf.constant(tol, dtype=tf.float32)
    vals = [tf.convert_to_tensor(v, dtype=tf.float32) for v in errs.values()]
    # Stack before comparison so we only emit one boolean
    return tf.reduce_all(tf.math.less(tf.stack(vals), tol_t))


def format_money(n: float) -> str:
    """Format a numeric value into a compact money string."""
    abs_n = abs(n)

    if abs_n < 1_000:
        return f"${n}"

    if abs_n < 1_000_000:
        return f"${n / 1_000:.3g}k"

    if abs_n < 1_000_000_000:
        return f"${n / 1_000_000:.3g}mn"

    if abs_n < 1_000_000_000_000:
        return f"${n / 1_000_000_000:.3g}bn"

    return f"${n / 1_000_000_000_000:.3g}tn"
