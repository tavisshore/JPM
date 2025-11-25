import tensorflow as tf
import numpy as np

from choice_learn_ext.models.deep_context.deep_halo_core import DeepContextChoiceModel


def test_permutation_equivariance():
    num_items = 6
    model = DeepContextChoiceModel(num_items=num_items)

    # base choice set
    available = tf.ones((1, num_items), dtype=tf.float32)
    item_ids = tf.range(num_items, dtype=tf.int32)[tf.newaxis, :]

    # Random permutation
    perm = np.random.permutation(num_items)
    inv_perm = np.argsort(perm)

    # Convert perm into a TF index
    perm_tf = tf.constant(perm, dtype=tf.int32)

    # Apply permutation
    item_ids_perm = tf.gather(item_ids, perm_tf, axis=1)
    available_perm = tf.gather(available, perm_tf, axis=1)

    batch_orig = {"available": available, "item_ids": item_ids}
    batch_perm = {"available": available_perm, "item_ids": item_ids_perm}

    out_orig = model(batch_orig, training=False)["log_probs"].numpy()
    out_perm = model(batch_perm, training=False)["log_probs"].numpy()

    # Undo permutation
    out_perm_unperm = out_perm[:, inv_perm]

    assert np.allclose(out_orig, out_perm_unperm, atol=1e-5)
