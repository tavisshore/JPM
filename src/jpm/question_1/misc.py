import math
from typing import Dict

import pandas as pd
import tensorflow as tf


def _snake(s: str) -> str:
    return (
        s.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("__", "_")
    )


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
