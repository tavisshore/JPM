# src/jpm/question_3/choice_learn_ext/models/deep_context/utils.py

from __future__ import annotations

import tensorflow as tf



def masked_mean(x: tf.Tensor, mask: tf.Tensor, axis: int = 1) -> tf.Tensor:
    """
    Compute mean along `axis`, only where mask == 1.

    x:    (..., J, d)
    mask: (..., J)
    axis: axis over items (usually 1)

    Returns:
        (..., d)
    """
    x = tf.convert_to_tensor(x)
    mask = tf.cast(mask, x.dtype)  # (..., J)

    # Expand mask so it can multiply x
    while len(mask.shape) < len(x.shape):
        mask = tf.expand_dims(mask, axis=-1)  # (..., J, 1)

    # Sum over items
    num = tf.reduce_sum(x * mask, axis=axis)      # (..., d)
    den = tf.reduce_sum(mask, axis=axis)          # (..., 1)

    den = tf.maximum(den, tf.constant(1.0, dtype=x.dtype))
    # Broadcasting: (..., d) / (..., 1) -> (..., d)
    return num / den



def masked_log_softmax(logits: tf.Tensor, mask: tf.Tensor, axis: int = -1) -> tf.Tensor:
    """
    Log-softmax with a binary availability mask.

    logits: (B, J)
    mask:   (B, J)  1 if item is available, 0 otherwise

    Returns log-probabilities over available items; unavailable items
    effectively have log-prob ~ -inf.
    """
    logits = tf.convert_to_tensor(logits)
    mask = tf.cast(mask, tf.bool)


    very_neg = tf.constant(-1e9, dtype=logits.dtype)
    masked_logits = tf.where(mask, logits, very_neg)
    return tf.nn.log_softmax(masked_logits, axis=axis)
