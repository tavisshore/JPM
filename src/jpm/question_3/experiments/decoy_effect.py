"""
decoy_effect.py
-------------------------------------------------------------

Synthetic experiment to demonstrate the decoy (asymmetric dominance)
effect using the TensorFlow DeepHalo implementation
(DeepContextChoiceModel).

Setup:
  - 3 items:
        0 = A (target)
        1 = B (competitor)
        2 = C (decoy dominated by A)
  - Two choice sets:
        S1 = {A, B}
        S2 = {A, B, C}
    with "true" underlying probabilities:
        P(. | {A,B})   = [0.45, 0.55, 0.00]
        P(. | {A,B,C}) = [0.60, 0.40, 0.00]
  - The decoy C is designed to increase P(A) when added.

What this script does:
  1. Generates a synthetic dataset by sampling from the above
     distributions (5000 draws per choice set).
  2. Trains DeepContextChoiceModel on this data.
  3. Evaluates P(. | {A,B}) and P(. | {A,B,C}) from the fitted model.
  4. Prints the learned probabilities and the decoy effect
     Δ_A = P(A | {A,B,C}) - P(A | {A,B}).
  5. Saves a bar plot to:
        decoy_effect_probs.png

Run with:
    python src/jpm/question_3/experiments/decoy_effect.py
"""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Make sure choice_learn_ext is importable when running this as a script
# ---------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
QUESTION3_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if QUESTION3_DIR not in sys.path:
    sys.path.insert(0, QUESTION3_DIR)

#from choice_learn_ext.models.deep_context.deep_halo_core import DeepContextChoiceModel
from choice_learn_ext.models.deep_context.deep_halo_core import DeepContextChoiceModel

from choice_learn_ext.models.deep_context.trainer import Trainer

# ---------------------------------------------------------------------
# 1. Synthetic decoy setup
# ---------------------------------------------------------------------
# 3 items:
#   0 = A (target)
#   1 = B (competitor)
#   2 = C (decoy dominated by A)
J = 3

# Two choice sets:
#   S1 = {A, B}
#   S2 = {A, B, C}
# with "true" probabilities designed to create a decoy effect on A.
rows = [
    ((0, 1),    [0.45, 0.55, 0.00]),  # no decoy
    ((0, 1, 2), [0.60, 0.40, 0.00]),  # with decoy C
]


def build_sampled_dataset(rows, draws_per_row=5000, seed=0):
    """
    Build a synthetic dataset from specified choice sets and
    'true' probability vectors.

    For each row (S, probs):
      - zero out probabilities for items not in S
      - renormalize
      - sample draws_per_row choices

    Returns:
      available: (N, J) float32
      choices:   (N,)   int32
      item_ids:  (N, J) int32
    """
    rng = np.random.default_rng(seed)
    avail_list, choice_list = [], []

    for S, probs in rows:
        p = np.array(probs, dtype=np.float64)

        # Availability mask
        mask = np.zeros(J, dtype=np.float32)
        for j in S:
            mask[j] = 1.0

        # Mask probabilities and renormalize over available items
        p_masked = p.copy()
        p_masked[mask == 0] = 0.0
        p_masked = p_masked / p_masked.sum()

        # Sample choices for this set
        choices = rng.choice(np.arange(J), size=draws_per_row, p=p_masked)
        for c in choices:
            avail_list.append(mask.copy())
            choice_list.append(c)

    available = np.stack(avail_list).astype(np.float32)
    choices = np.array(choice_list, dtype=np.int32)
    item_ids = np.tile(np.arange(J, dtype=np.int32), (available.shape[0], 1))

    return available, choices, item_ids


def plot_decoy_effect(probs_no_decoy, probs_decoy, out_path):
    """
    Bar chart comparing P(A) and P(B) with and without the decoy.
    """
    labels = [
        "A (no decoy)",
        "A (with decoy)",
        "B (no decoy)",
        "B (with decoy)",
    ]
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
    plt.title("Decoy effect: impact of C on A and B")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    tf.random.set_seed(0)
    np.random.seed(0)

    # --------------------------------------------------------
    # 2. Build synthetic dataset
    # --------------------------------------------------------
    available, choices, item_ids = build_sampled_dataset(
        rows, draws_per_row=5000, seed=0
    )

    print("Training data shape:")
    print("  available:", available.shape)
    print("  choices:  ", choices.shape)
    print("  item_ids: ", item_ids.shape)

    # --------------------------------------------------------
    # 3. Instantiate and train model
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # 4. Evaluate probabilities on the two sets
    # --------------------------------------------------------
    # S1 = {A,B}
    avail_no_decoy = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)
    # S2 = {A,B,C}
    avail_decoy = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)

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

    print("\nPredicted probabilities:")
    print("  P(. | {A,B})     =", np.round(probs_no_decoy, 3))
    print("  P(. | {A,B,C})   =", np.round(probs_decoy, 3))

    # Check decoy effect on A
    delta_A = probs_decoy[0] - probs_no_decoy[0]
    print(f"\nDecoy effect on A: P(A | ABC) - P(A | AB) = {delta_A:.3f}")
    if delta_A > 0:
        print("  -> Decoy effect present (A becomes more likely when C is added).")
    else:
        print("  -> Decoy effect NOT present; training/config may need adjustment.")

    # --------------------------------------------------------
    # 5. Plot bar chart
    # --------------------------------------------------------
    out_plot = os.path.join(CURRENT_DIR, "decoy_effect_probs.png")
    plot_decoy_effect(probs_no_decoy, probs_decoy, out_plot)
    print("\nSaved bar plot to:", out_plot)


if __name__ == "__main__":
    main()
