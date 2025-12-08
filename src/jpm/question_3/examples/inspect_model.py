import os
import sys

import numpy as np
import tensorflow as tf
from choice_learn_ext.models.deep_context.model import DeepContextChoiceModel

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
QUESTION3_DIR = os.path.abspath(
    os.path.join(CURRENT_DIR, "..")
)  # .../src/jpm/question_3
if QUESTION3_DIR not in sys.path:
    sys.path.insert(0, QUESTION3_DIR)


def main():
    tf.random.set_seed(0)
    np.random.seed(0)

    num_items = 4  # imagine: 0=Pepsi, 1=Coke, 2=Sprite, 3=7-Up
    model = DeepContextChoiceModel(num_items=num_items)

    # define some simple choice sets
    # S1 = {Pepsi, Coke}
    # S2 = {Pepsi, Coke, Sprite}
    # S3 = {Pepsi, Coke, 7-Up}
    available = tf.constant(
        [
            [1, 1, 0, 0],  # S1
            [1, 1, 1, 0],  # S2
            [1, 1, 0, 1],  # S3
        ],
        dtype=tf.float32,
    )

    item_ids = tf.constant(
        [
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ],
        dtype=tf.int32,
    )

    outputs = model({"available": available, "item_ids": item_ids}, training=False)
    logits = outputs["utilities"].numpy()
    probs = np.exp(outputs["log_probs"].numpy())

    print("Logits:\n", logits)
    print("Probs:\n", probs)


if __name__ == "__main__":
    main()
