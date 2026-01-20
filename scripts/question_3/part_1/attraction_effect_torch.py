"""
attraction_effect_torch.py
--------------------------------------------------------------

This script reproduces the *attraction effect* using the authors'
original **PyTorch DeepHalo implementation** (FeatureBased.py).

The attraction effect:
    Adding a dominated option C increases the probability of choosing
    option B, violating IIA.

Choice structure:
    Items: 0 = A, 1 = B, 2 = C
    S1 = {A, B}
    S2 = {A, B, C}

Ground-truth synthetic probabilities:
    P(. | {A,B})   = [0.50, 0.50, 0.00]
    P(. | {A,B,C}) = [0.30, 0.70, 0.00]

How the authors’ Torch model works:
    - It uses prefix lengths instead of availability masks.
    - Item features are one-hot vectors of dimension J.
    - X has shape (B, J, J) but only the "first length" items are active.
    - This matches exactly the implementation used in the original 2025 code.

Steps performed by this script:
    1. Generate synthetic dataset sampled from the "true" probabilities.
    2. Train the authors' DeepHalo model for 80 epochs.
    3. Evaluate P(. | {A,B}) and P(. | {A,B,C}).
    4. Compute attraction effect Δ_B.
    5. Save a bar plot:
            attraction_effect_probs_torch.png

Run:
    python src/jpm/question_3/experiments/attraction_effect_torch.py
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Authors' official DeepHalo (PyTorch version)
from jpm.question_3.authors.FeatureBased import DeepHalo

# ---------------------------------------------------------------------
# Ensure authors' model is importable when running this as a script
# ---------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AUTHORS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "authors"))
SAVE_DIR = Path("results/question_3/part_1")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
if AUTHORS_DIR not in sys.path:
    sys.path.insert(0, AUTHORS_DIR)


# ---------------------------------------------------------------------
# Synthetic attraction-effect setup
# ---------------------------------------------------------------------
J = 3  # A,B,C

rows = [
    ((0, 1), [0.50, 0.50, 0.00]),  # {A,B}
    ((0, 1, 2), [0.30, 0.70, 0.00]),  # {A,B,C} with decoy C
]


def build_sampled_dataset(rows, draws_per_row=5000, seed=0):
    """
    Build synthetic dataset for the authors' PyTorch model.

    The authors’ version uses "lengths" instead of availability masks.
    - If length = 2 → only A,B available
    - If length = 3 → A,B,C available

    Returns:
        lengths: (N,) int64
        choices: (N,) int64
    """
    rng = np.random.default_rng(seed)
    lengths_list, choice_list = [], []

    for S, probs in rows:
        p = np.array(probs, dtype=np.float64)
        length = len(S)  # prefix representation

        # Mask & renormalize
        p_masked = p.copy()
        p_masked[length:] = 0.0
        p_masked = p_masked / p_masked.sum()

        # Draw choices
        choices = rng.choice(np.arange(J), size=draws_per_row, p=p_masked)
        for c in choices:
            lengths_list.append(length)
            choice_list.append(int(c))

    lengths = np.array(lengths_list, dtype=np.int64)
    choices = np.array(choice_list, dtype=np.int64)
    return lengths, choices


def make_item_features(J):
    """
    The authors use fixed one-hot item features.

    Output:
        (1, J, J) tensor that is broadcast across the batch.
    """
    eye = np.eye(J, dtype=np.float32)
    return torch.from_numpy(eye).unsqueeze(0)  # shape (1, J, J)


def plot_attraction_effect(probs_no_decoy, probs_decoy, out_path):
    labels = ["A (no decoy)", "A (with decoy)", "B (no decoy)", "B (with decoy)"]
    values = [
        probs_no_decoy[0],  # A
        probs_decoy[0],
        probs_no_decoy[1],  # B
        probs_decoy[1],
    ]

    x = np.arange(len(labels))
    plt.figure(figsize=(8, 4))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Choice probability")
    plt.title("Attraction effect (PyTorch DeepHalo)")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    # --------------------------------------------------------
    # 1. Generate synthetic dataset
    # --------------------------------------------------------
    lengths, choices = build_sampled_dataset(rows)
    N = lengths.shape[0]
    print("Training data:")
    print("  N samples:", N)

    lengths_t = torch.from_numpy(lengths)
    choices_t = torch.from_numpy(choices)

    item_feats = make_item_features(J)  # (1, J, J)

    # --------------------------------------------------------
    # 2. Instantiate authors’ PyTorch DeepHalo
    # --------------------------------------------------------
    model = DeepHalo(
        n=J,
        input_dim=J,
        H=4,
        L=2,
        embed=128,
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # --------------------------------------------------------
    # 3. Train model
    # --------------------------------------------------------
    batch_size = 1024
    num_epochs = 80
    model.train()

    for epoch in range(1, num_epochs + 1):
        perm = torch.randperm(N)
        lengths_ep = lengths_t[perm]
        choices_ep = choices_t[perm]

        total_loss = 0.0
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            lens_b = lengths_ep[start:end]
            choices_b = choices_ep[start:end]
            B = lens_b.shape[0]

            X = item_feats.expand(B, -1, -1)  # (B, J, J)
            log_probs, logits = model(X, lens_b)

            nll = -log_probs[torch.arange(B), choices_b].mean()

            optimizer.zero_grad()
            nll.backward()
            optimizer.step()
            total_loss += float(nll.detach()) * B

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}  NLL: {total_loss / N:.4f}")

    # --------------------------------------------------------
    # 4. Evaluate attraction effect
    # --------------------------------------------------------
    model.eval()
    with torch.no_grad():
        lens_eval = torch.tensor([2, 3], dtype=torch.long)  # {A,B} and {A,B,C}
        B = lens_eval.shape[0]
        X_eval = item_feats.expand(B, -1, -1)
        log_probs_eval, _ = model(X_eval, lens_eval)
        probs_eval = torch.exp(log_probs_eval).cpu().numpy()

    probs_no_decoy = probs_eval[0]
    probs_decoy = probs_eval[1]

    print("\nPredicted probabilities (Torch):")
    print("  P(. | {A,B})   =", np.round(probs_no_decoy, 3))
    print("  P(. | {A,B,C}) =", np.round(probs_decoy, 3))

    delta_B = probs_decoy[1] - probs_no_decoy[1]
    print(f"\nAttraction effect on B (Torch): Δ = {delta_B:.3f}")

    # --------------------------------------------------------
    # 5. Save bar plot
    # --------------------------------------------------------
    out_plot = SAVE_DIR / "attraction_effect_probs_torch.png"
    plot_attraction_effect(probs_no_decoy, probs_decoy, out_plot)
    print("\nSaved Torch bar plot to:", out_plot)


if __name__ == "__main__":
    main()
