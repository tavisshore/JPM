import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
QUESTION3_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if QUESTION3_DIR not in sys.path:
    sys.path.insert(0, QUESTION3_DIR)

#from choice_learn_ext.models.deep_context.model import DeepContextChoiceModel
from choice_learn_ext.models.deep_context.deep_halo_core import DeepContextChoiceModel

from choice_learn_ext.models.deep_context.trainer import Trainer

# ------------------------------------------------------------
# 1. Attraction-effect setup
# ------------------------------------------------------------
# 3 items:
# 0 = A (target)
# 1 = B (dominating competitor)
# 2 = C (decoy dominated by B)
J = 3

# Ground-truth probabilities:
# S1 = {A,B}      : A and B roughly equal
# S2 = {A,B,C}    : decoy C pushes share toward B
rows = [
    ((0, 1),    [0.50, 0.50, 0.00]),  # no decoy
    ((0, 1, 2), [0.30, 0.70, 0.00]),  # with decoy favouring B
]


def build_sampled_dataset(rows, draws_per_row=5000, seed=0):
    rng = np.random.default_rng(seed)
    avail_list, choice_list = [], []

    for S, probs in rows:
        p = np.array(probs, dtype=np.float64)

        mask = np.zeros(J, dtype=np.float32)
        for j in S:
            mask[j] = 1.0

        p_masked = p.copy()
        p_masked[mask == 0] = 0.0
        p_masked = p_masked / p_masked.sum()

        choices = rng.choice(np.arange(J), size=draws_per_row, p=p_masked)
        for c in choices:
            avail_list.append(mask.copy())
            choice_list.append(c)

    available = np.stack(avail_list).astype(np.float32)
    choices = np.array(choice_list, dtype=np.int32)
    item_ids = np.tile(np.arange(J, dtype=np.int32), (available.shape[0], 1))
    return available, choices, item_ids


def plot_attraction_effect(probs_no_decoy, probs_decoy, out_path):
    labels = ["A (no decoy)", "A (with decoy)",
              "B (no decoy)", "B (with decoy)"]
    values = [
        probs_no_decoy[0],  # P(A | {A,B})
        probs_decoy[0],     # P(A | {A,B,C})
        probs_no_decoy[1],  # P(B | {A,B})
        probs_decoy[1],     # P(B | {A,B,C})
    ]

    x = np.arange(len(labels))
    plt.figure(figsize=(8, 4))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Choice probability")
    plt.title("Attraction effect (TF): impact of decoy C on A and B")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    tf.random.set_seed(0)
    np.random.seed(0)

    available, choices, item_ids = build_sampled_dataset(rows)
    print("Training data shape:")
    print("  available:", available.shape)
    print("  choices:  ", choices.shape)
    print("  item_ids: ", item_ids.shape)

    model = DeepContextChoiceModel(num_items=J)
    trainer = Trainer(model, lr=1e-2)

    trainer.fit_arrays(
        available=tf.convert_to_tensor(available),
        choices=tf.convert_to_tensor(choices),
        item_ids=tf.convert_to_tensor(item_ids),
        batch_size=1024,
        epochs=80,
        verbose=1,
    )

    # Evaluate on S1={A,B} and S2={A,B,C}
    avail_no_decoy = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)
    avail_decoy    = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)

    eval_available = tf.convert_to_tensor(
        np.vstack([avail_no_decoy, avail_decoy]), dtype=tf.float32
    )
    eval_item_ids = tf.convert_to_tensor(
        np.tile(np.arange(J, dtype=np.int32), (2, 1))
    )

    outputs = model(
        {"available": eval_available, "item_ids": eval_item_ids},
        training=False,
    )
    probs = tf.exp(outputs["log_probs"]).numpy()
    probs_no_decoy = probs[0]
    probs_decoy = probs[1]

    print("\nPredicted probabilities (TF):")
    print("  P(. | {A,B})   =", np.round(probs_no_decoy, 3))
    print("  P(. | {A,B,C}) =", np.round(probs_decoy, 3))

    delta_B = probs_decoy[1] - probs_no_decoy[1]
    print(f"\nAttraction effect on B: P(B | ABC) - P(B | AB) = {delta_B:.3f}")

    out_plot = os.path.join(CURRENT_DIR, "attraction_effect_probs_tf.png")
    plot_attraction_effect(probs_no_decoy, probs_decoy, out_plot)
    print("\nSaved TF bar plot to:", out_plot)


if __name__ == "__main__":
    main()
