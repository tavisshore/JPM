import math
from typing import Any, Dict, List

import pandas as pd
import tensorflow as tf

Nested = Dict[str, Any] | List[str]


def _snake(s: str) -> str:
    return (
        s.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("__", "_")
    )


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


def get_leaf_values(d: Nested, sub_key: str | None = None) -> List[str]:
    if sub_key:
        subtree = find_subtree(d, sub_key)
        if subtree is None:
            return []
        return collect_leaves(subtree)

    return collect_leaves(d)


def to_tensor(x) -> tf.Tensor:
    return tf.convert_to_tensor(x, dtype=tf.float32)


def tf_sum(xs):
    return tf.add_n(xs)


def coerce_float(x) -> float:
    if pd.isna(x):
        return 0.0
    try:
        return float(x)
    except Exception:
        return 0.0


def as_series(mapping, years):
    return pd.Series(
        [mapping.get(y, math.nan) for y in years], index=years, dtype="float64"
    )


def errs_below_tol(errs: Dict[str, tf.Tensor], tol: float = 1e-4) -> tf.Tensor:
    tol_t = tf.constant(tol, dtype=tf.float32)
    vals = [tf.convert_to_tensor(v, dtype=tf.float32) for v in errs.values()]
    return tf.reduce_all(tf.math.less(tf.stack(vals), tol_t))
