import numpy as np
import tensorflow as tf

from jpm.question_3.choice_learn_extension.deep_halo_core import DeepContextChoiceModel


def test_masking_behavior():
    num_items = 5
    model = DeepContextChoiceModel(num_items=num_items)

    # Only items 0 and 2 available
    available = tf.constant([[1, 0, 1, 0, 0]], dtype=tf.float32)
    item_ids = tf.constant([[0, 1, 2, 3, 4]], dtype=tf.int32)

    out = model({"available": available, "item_ids": item_ids}, training=False)
    log_probs = out["log_probs"].numpy()[0]

    # Masked-out items must have huge negative log-probability
    for idx in [1, 3, 4]:
        assert log_probs[idx] < -1e5

    # Available items should be finite
    assert np.isfinite(log_probs[0])
    assert np.isfinite(log_probs[2])

    # Probabilities on available items should sum to ~1
    probs = np.exp(log_probs)
    assert np.allclose(probs[[0, 2]].sum(), 1.0, atol=1e-6)
