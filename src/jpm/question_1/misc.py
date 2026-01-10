from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf

Nested = Dict[str, Any] | List[str]

# Encode ratings
RATINGS_MAPPINGS = {
    "AAA": 1,
    "AA+": 2,
    "AA": 3,
    "AA-": 4,
    "A+": 5,
    "A": 6,
    "A-": 7,
    "BBB+": 8,
    "BBB": 9,
    "BBB-": 10,
    "BB+": 11,
    "BB": 12,
    "BB-": 13,
    "B+": 14,
    "B": 15,
    "B-": 16,
    "CCC+": 17,
    "CCC": 18,
    "CCC-": 19,
    "CC": 20,
    "C": 21,
    "D": 22,
}


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


def get_args():
    """Build CLI args for training entrypoints."""
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--ticker", type=str, default="AAPL")
    p.add_argument("--cache_dir", type=str, default=None)  # required=True)
    p.add_argument("--target", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lookback", type=int, default=None)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--seasonal_weight", type=float, default=None)

    # Model params
    p.add_argument("--hidden_units", type=int, default=None)
    p.add_argument("--dense_units", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)

    # Training
    p.add_argument("--checkpoint_path", type=Path, default=Path("ckpts"))

    # BS identiy loss
    p.add_argument("--lambda_balance", type=float, default=None)
    p.add_argument("--enforce_balance", type=bool, default=None)
    p.add_argument("--learn_subtotals", type=bool, default=None)

    # LLM Ensemble
    p.add_argument("--use_llm", action="store_true")
    p.add_argument("--llm_provider", type=str, default=None)
    p.add_argument("--llm_model", type=str, default=None)
    p.add_argument("--llm_temperature", type=float, default=None)
    p.add_argument("--llm_max_tokens", type=int, default=None)
    p.add_argument("--adjust", type=bool, default=None)

    args = p.parse_args()

    # Resolve cache_dir: CLI arg > env var > hardcoded default
    if args.cache_dir is None:
        args.cache_dir = os.getenv("JPM_CACHE_DIR", "/scratch/datasets/jpm")

    return args
