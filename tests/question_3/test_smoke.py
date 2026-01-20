# src/jpm/question_3/tests/test_smoke.py

import tensorflow as tf

from jpm.question_3.choice_learn_ext.models.deep_context.deep_halo_core import (
    DeepContextChoiceModel,
)
from jpm.question_3.choice_learn_ext.models.deep_context.trainer import Trainer


def test_smoke_run():
    num_items = 5
    model = DeepContextChoiceModel(num_items=num_items)
    trainer = Trainer(model, lr=1e-2)

    # Two tiny choice sets
    available = tf.constant([[1, 1, 1, 0, 0], [1, 0, 1, 0, 0]], dtype=tf.float32)
    item_ids = tf.constant([[0, 1, 2, 3, 4], [0, 2, 1, 3, 4]], dtype=tf.int32)
    choices = tf.constant([1, 0], dtype=tf.int32)

    batch = {
        "available": available,
        "item_ids": item_ids,
        "choice": choices,
    }

    loss = trainer.train_step(batch)
    # Just check it's a finite non-negative scalar
    loss_val = float(loss.numpy())
    assert loss_val >= 0.0
