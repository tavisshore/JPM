"""
reproduce_table1.py
-------------------------------------------------------------

Reproduces the “Table 1” synthetic experiment from
Zhang et al. (2025), Deep Context-Dependent Choice Model.

What this script does:
  1. Defines the 11 synthetic choice sets and associated
     target probabilities (exactly as in the paper).
  2. Generates a synthetic dataset by sampling from these
     probabilities (1000 draws per choice set).
  3. Trains the TensorFlow implementation of DeepHalo
     (DeepContextChoiceModel) on this dataset.
  4. Evaluates the learned probabilities on the same 11 sets.
  5. Saves:
        - table1_predictions.csv
        - heatmap_target.png
        - heatmap_predicted.png
  6. Prints target vs predicted probabilities.

Run with:
    python src/jpm/question_3/experiments/reproduce_table1.py

Outputs appear in the same folder as this script.
"""
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import csv

# ---------------------------------------------------------------------
# Make sure choice_learn_ext is importable when running this as a script
# ---------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))         
QUESTION3_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..")) 

if QUESTION3_DIR not in sys.path:
    sys.path.insert(0, QUESTION3_DIR)

from choice_learn_ext.models.deep_context.deep_halo_core import DeepContextChoiceModel
from choice_learn_ext.models.deep_context.trainer import Trainer



# ---------------------------------------------------------------------
# 1. Zhang-style synthetic table (4 items)
# ---------------------------------------------------------------------
J = 4  # items indexed 1..4 in paper but 0..3 internally

rows = [
    ((1, 2),         [0.98, 0.02, 0.00, 0.00]),
    ((1, 3),         [0.50, 0.00, 0.50, 0.00]),
    ((1, 4),         [0.50, 0.00, 0.00, 0.50]),
    ((2, 3),         [0.00, 0.50, 0.50, 0.00]),
    ((2, 4),         [0.00, 0.50, 0.00, 0.50]),
    ((3, 4),         [0.00, 0.00, 0.90, 0.10]),
    ((1, 2, 3),      [0.49, 0.01, 0.50, 0.00]),
    ((1, 2, 4),      [0.49, 0.01, 0.00, 0.50]),
    ((1, 3, 4),      [0.50, 0.00, 0.45, 0.05]),
    ((2, 3, 4),      [0.00, 0.50, 0.45, 0.05]),
    ((1, 2, 3, 4),   [0.49, 0.01, 0.45, 0.05]),
]


def build_sampled_dataset(rows, draws_per_row=1000, seed=0):
    """
    Replicates the authors' synthetic generator.

    For each (choice set S, distribution p):
      - zero out items not in S
      - renormalize
      - draw `draws_per_row` samples
    """
    rng = np.random.default_rng(seed)
    avail_list, choice_list = [], []

    for S, probs in rows:
        S0 = [i - 1 for i in S]  # convert to 0-based

        # Build mask for presence
        mask = np.zeros(J, dtype=np.float32)
        for j in S0:
            mask[j] = 1.0

        # Mask raw probs and renormalize
        p = np.array(probs, dtype=np.float64)
        p_masked = p * mask
        p_masked /= p_masked.sum()

        # Draw samples
        draws = rng.choice(np.arange(J), size=draws_per_row, p=p_masked)
        for c in draws:
            avail_list.append(mask.copy())
            choice_list.append(c)

    available = np.stack(avail_list).astype(np.float32)
    choices = np.array(choice_list, dtype=np.int32)
    item_ids = np.tile(np.arange(J, dtype=np.int32), (available.shape[0], 1))

    return available, choices, item_ids


def make_heatmap(matrix, title, path):
    plt.figure(figsize=(6, 4))
    plt.imshow(matrix, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Items (1–4)")
    plt.ylabel("Choice Sets")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    tf.random.set_seed(0)
    np.random.seed(0)

    # ----------------------------------------------------
    # 2. Generate synthetic dataset
    # ----------------------------------------------------
    available, choices, item_ids = build_sampled_dataset(
        rows, draws_per_row=1000, seed=0
    )

    print("Training data:")
    print(" available:", available.shape)
    print(" choices:  ", choices.shape)
    print(" item_ids: ", item_ids.shape)

    # ----------------------------------------------------
    # 3. Build model + trainer
    # ----------------------------------------------------
    model = DeepContextChoiceModel(num_items=J)
    trainer = Trainer(model, lr=2e-3)

    # ----------------------------------------------------
    # 4. Train
    # ----------------------------------------------------
    trainer.fit_arrays(
        available=tf.convert_to_tensor(available),
        choices=tf.convert_to_tensor(choices),
        item_ids=tf.convert_to_tensor(item_ids),
        batch_size=1024,
        epochs=100,
        verbose=1,
    )

    # ----------------------------------------------------
    # 5. Evaluate on the 11 choice sets
    # ----------------------------------------------------
    eval_available = []
    for S, _ in rows:
        mask = np.zeros(J, dtype=np.float32)
        for j in [i - 1 for i in S]:
            mask[j] = 1.0
        eval_available.append(mask)

    eval_available = tf.convert_to_tensor(np.stack(eval_available), dtype=tf.float32)
    eval_item_ids = tf.convert_to_tensor(
        np.tile(np.arange(J, dtype=np.int32), (len(rows), 1))
    )

    out = model({"available": eval_available, "item_ids": eval_item_ids},
                training=False)
    probs = tf.exp(out["log_probs"]).numpy()

    # ----------------------------------------------------
    # 6. Save CSV
    # ----------------------------------------------------
    csv_path = os.path.join(CURRENT_DIR, "table1_predictions.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Set", "Target", "Predicted"])
        for (S, target), p in zip(rows, probs):
            writer.writerow([str(S), list(target), list(np.round(p, 5))])

    # ----------------------------------------------------
    # 7. Save heatmaps
    # ----------------------------------------------------
    target_matrix = np.stack([np.array(t) for (_, t) in rows])
    pred_matrix = probs

    target_path = os.path.join(CURRENT_DIR, "heatmap_target.png")
    pred_path = os.path.join(CURRENT_DIR, "heatmap_predicted.png")

    make_heatmap(target_matrix, "Target Probabilities", target_path)
    make_heatmap(pred_matrix, "Predicted Probabilities", pred_path)

    # ----------------------------------------------------
    # 8. Print results
    # ----------------------------------------------------
    print("\n=== Target vs Predicted ===")
    for (S, target), p in zip(rows, probs):
        S_str = str(S)
        print(
            f"S={S_str:12}  "
            f"target={np.round(target, 3)}  "
            f"pred={np.round(p, 3)}"
    )



    print("\nSaved:")
    print(" -", csv_path)
    print(" -", target_path)
    print(" -", pred_path)


if __name__ == "__main__":
    main()
