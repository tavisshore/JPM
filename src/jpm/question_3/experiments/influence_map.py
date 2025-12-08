import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from choice_learn_ext.models.deep_context.deep_halo_core import DeepContextChoiceModel
from choice_learn_ext.models.deep_context.trainer import Trainer

# ---------------------------------------------------------------------
# Make sure choice_learn_ext is importable when running this as a script
# ---------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
QUESTION3_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if QUESTION3_DIR not in sys.path:
    sys.path.insert(0, QUESTION3_DIR)


# ------------------------------------------------------------
# 1. Use the same Table-1 synthetic setup as before
# ------------------------------------------------------------
J = 4  # four items, indexed 0..3

# rows: (choice set S using 1-based labels, target probability vec over 4 items)
rows = [
    ((1, 2), [0.98, 0.02, 0.00, 0.00]),
    ((1, 3), [0.50, 0.00, 0.50, 0.00]),
    ((1, 4), [0.50, 0.00, 0.00, 0.50]),
    ((2, 3), [0.00, 0.50, 0.50, 0.00]),
    ((2, 4), [0.00, 0.50, 0.00, 0.50]),
    ((3, 4), [0.00, 0.00, 0.90, 0.10]),
    ((1, 2, 3), [0.49, 0.01, 0.50, 0.00]),
    ((1, 2, 4), [0.49, 0.01, 0.00, 0.50]),
    ((1, 3, 4), [0.50, 0.00, 0.45, 0.05]),
    ((2, 3, 4), [0.00, 0.50, 0.45, 0.05]),
    ((1, 2, 3, 4), [0.49, 0.01, 0.45, 0.05]),
]


def build_sampled_dataset(rows, draws_per_row=1000, seed=0):
    rng = np.random.default_rng(seed)
    avail_list, choice_list = [], []

    for S_1based, probs in rows:
        S = [i - 1 for i in S_1based]  # to 0-based indices
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


def train_deephalo():
    """Train DeepContextChoiceModel on the Table-1 synthetic dataset."""
    tf.random.set_seed(0)
    np.random.seed(0)

    available, choices, item_ids = build_sampled_dataset(
        rows, draws_per_row=1000, seed=0
    )
    print("Training data shape:")
    print("  available:", available.shape)
    print("  choices:  ", choices.shape)
    print("  item_ids: ", item_ids.shape)

    model = DeepContextChoiceModel(num_items=J)
    trainer = Trainer(model, lr=2e-3)

    trainer.fit_arrays(
        available=tf.convert_to_tensor(available),
        choices=tf.convert_to_tensor(choices),
        item_ids=tf.convert_to_tensor(item_ids),
        batch_size=1024,
        epochs=100,
        verbose=1,
    )
    return model


def compute_probabilities_for_sets(model):
    """
    Evaluate the trained model on all distinct choice sets S in `rows`.
    Returns a dict mapping frozenset(S) -> probability vector p_S (length J).
    """
    eval_sets = []
    for S_1based, _ in rows:
        S0 = tuple(sorted(i - 1 for i in S_1based))
        eval_sets.append(S0)

    eval_sets_unique = sorted(set(eval_sets))

    # build availability matrix
    avail_list = []
    for S in eval_sets_unique:
        mask = np.zeros(J, dtype=np.float32)
        for j in S:
            mask[j] = 1.0
        avail_list.append(mask)

    eval_available = tf.convert_to_tensor(np.stack(avail_list), dtype=tf.float32)
    eval_item_ids = tf.convert_to_tensor(
        np.tile(np.arange(J, dtype=np.int32), (len(eval_sets_unique), 1))
    )

    outputs = model(
        {"available": eval_available, "item_ids": eval_item_ids},
        training=False,
    )
    probs = tf.exp(outputs["log_probs"]).numpy()

    set_to_probs = {}
    for S, p in zip(eval_sets_unique, probs, strict=True):
        set_to_probs[S] = p

    return set_to_probs


def compute_influence_matrix(set_to_probs):
    """
    Compute influence matrix I(i -> j) averaged over all baseline sets S
    where j in S, i not in S, and both S and S ∪ {i} exist in set_to_probs.
    """
    items = list(range(J))
    influence = np.zeros((J, J), dtype=np.float64)

    all_sets = list(set_to_probs.keys())

    for i in items:
        for j in items:
            if i == j:
                continue

            diffs = []
            for S in all_sets:
                S = tuple(S)
                if (j in S) and (i not in S):
                    S_aug = tuple(sorted(S + (i,)))
                    if S_aug in set_to_probs:
                        p_base = set_to_probs[S][j]
                        p_aug = set_to_probs[S_aug][j]
                        diffs.append(p_aug - p_base)

            if diffs:
                influence[i, j] = float(np.mean(diffs))
            else:
                influence[i, j] = 0.0

    return influence


def plot_influence_heatmap(influence, out_path):
    """
    Plot a J x J heatmap of influence[i,j] = effect of i on j.
    Columns: source item i
    Rows:    affected item j
    """
    plt.figure(figsize=(6, 5))
    # We want rows = j, cols = i, so transpose.
    mat = influence.T
    im = plt.imshow(mat, aspect="equal")
    plt.colorbar(im)
    plt.xlabel("Influence source i (items 1–4)")
    plt.ylabel("Affected item j (items 1–4)")
    plt.xticks(range(J), [str(k + 1) for k in range(J)])
    plt.yticks(range(J), [str(k + 1) for k in range(J)])
    plt.title("Average influence I(i → j)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_influence_csv(influence, out_path):
    header = ["item_j \\ item_i"] + [f"i={i + 1}" for i in range(J)]
    rows_csv = []
    for j in range(J):
        row = [f"j={j + 1}"] + [f"{influence[i, j]:.5f}" for i in range(J)]
        rows_csv.append(",".join(row))
    with open(out_path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows_csv:
            f.write(r + "\n")


def main():
    model = train_deephalo()
    set_to_probs = compute_probabilities_for_sets(model)
    influence = compute_influence_matrix(set_to_probs)

    print("\nInfluence matrix I(i -> j):")
    print(influence)

    out_png = os.path.join(CURRENT_DIR, "influence_matrix.png")
    out_csv = os.path.join(CURRENT_DIR, "influence_matrix.csv")
    plot_influence_heatmap(influence, out_png)
    save_influence_csv(influence, out_csv)

    print("\nSaved:")
    print("  - influence heatmap:", out_png)
    print("  - influence csv:    ", out_csv)


if __name__ == "__main__":
    main()
