from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import tensorflow as tf

Nested = Dict[str, Any] | List[str]


def find_subtree(d: Nested, target: str) -> Nested | None:
    """Locate the subtree whose key == target."""
    if isinstance(d, list):
        return None

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

    out: List[str] = []
    for v in d.values():
        out.extend(collect_leaves(v))
    return out


def get_leaf_values(d, sub_key: str | None = None) -> List[str]:
    if sub_key:
        subtree = find_subtree(d, sub_key)
        if subtree is None:
            return []
        return collect_leaves(subtree)

    return collect_leaves(d)


def to_tensor(x) -> tf.Tensor:
    return tf.convert_to_tensor(x, dtype=tf.float32)


def tf_sum(xs) -> tf.Tensor:
    return tf.add_n(xs)


def coerce_float(x) -> float:
    if pd.isna(x):
        return 0.0
    try:
        return float(x)
    except Exception:
        return 0.0


def as_series(mapping: Dict[int, float], years) -> pd.Series:
    return pd.Series(
        [mapping.get(y, math.nan) for y in years], index=years, dtype="float64"
    )


def errs_below_tol(errs: Dict[str, tf.Tensor], tol: float = 1e-4) -> tf.Tensor:
    tol_t = tf.constant(tol, dtype=tf.float32)
    vals = [tf.convert_to_tensor(v, dtype=tf.float32) for v in errs.values()]
    return tf.reduce_all(tf.math.less(tf.stack(vals), tol_t))


def format_money(n: float) -> str:
    abs_n = abs(n)

    if abs_n < 1_000:
        return f"${n}"

    if abs_n < 1_000_000:
        return f"${n/1_000:.3g}k"

    if abs_n < 1_000_000_000:
        return f"${n/1_000_000:.3g}mn"

    if abs_n < 1_000_000_000_000:
        return f"${n/1_000_000_000:.3g}bn"

    return f"${n/1_000_000_000_000:.3g}tn"


def train_args():
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--ticker", type=str, default="AAPL")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--target", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lookback", type=int, default=None)
    p.add_argument("--horizon", type=int, default=None)

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

    return p.parse_args()
