import tensorflow as tf

from jpm.question_3.choice_learn_ext.models.deep_context.deep_halo_core import (
    DeepContextChoiceModel,
)
from jpm.question_3.choice_learn_ext.models.deep_context.trainer import Trainer


def test_overfit_tiny_dataset():
    num_items = 3
    model = DeepContextChoiceModel(num_items=num_items)
    trainer = Trainer(model, lr=5e-2)

    # Single choice set {0,1,2}, always choose item 1
    N = 40
    available = tf.constant([[1, 1, 1]] * N, dtype=tf.float32)
    item_ids = tf.constant([[0, 1, 2]] * N, dtype=tf.int32)
    choices = tf.constant([1] * N, dtype=tf.int32)

    batch = {"available": available, "item_ids": item_ids, "choice": choices}

    # Train a bit
    for _ in range(150):
        loss = trainer.train_step(batch)

    out = model(batch, training=False)["log_probs"]
    probs = tf.exp(out).numpy()
    mean_p1 = probs[:, 1].mean()

    # We don't need perfection, just clear learning
    assert mean_p1 > 0.9
    assert float(loss.numpy()) < 0.5
