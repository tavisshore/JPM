import os
import sys

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


# Start of the compromised effect tf reporducation of Zhang
# ------------------------------------------------------------
def build_sampled_dataset(rows, J, draws_per_row=2000, seed=0):
    rng = np.random.default_rng(seed)
    avail_list, choice_list = [], []

    for S, probs in rows:
        S0 = [i - 1 for i in S]  # 1-based → 0-based
        p = np.array(probs, dtype=np.float64)

        mask = np.zeros(J, dtype=np.float32)
        for j in S0:
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


def main():
    # Three items: A, B (compromise), C
    J = 3

    rows = [
        ((1, 2), [0.30, 0.70, 0.00]),  # {A,B}
        ((2, 3), [0.00, 0.30, 0.70]),  # {B,C}
        ((1, 2, 3), [0.10, 0.80, 0.10]),  # {A,B,C}
    ]

    available, choices, item_ids = build_sampled_dataset(
        rows, J=J, draws_per_row=4000, seed=0
    )

    print("Training data shape:")
    print(" available:", available.shape)
    print(" choices:  ", choices.shape)
    print(" item_ids: ", item_ids.shape)

    # Build + train TF model
    model = DeepContextChoiceModel(num_items=J)
    trainer = Trainer(model, lr=2e-3)

    trainer.fit_arrays(
        available=tf.convert_to_tensor(available),
        choices=tf.convert_to_tensor(choices),
        item_ids=tf.convert_to_tensor(item_ids),
        batch_size=1024,
        epochs=80,
    )

    # Evaluate compromise effect
    eval_sets = {
        "AB": (1, 2),
        "BC": (2, 3),
        "ABC": (1, 2, 3),
    }

    eval_available = []
    for S in eval_sets.values():
        mask = np.zeros(J, dtype=np.float32)
        for j in [i - 1 for i in S]:
            mask[j] = 1.0
        eval_available.append(mask)

    eval_available = tf.convert_to_tensor(np.stack(eval_available))
    eval_item_ids = tf.convert_to_tensor(np.tile(np.arange(J), (len(eval_sets), 1)))

    probs = trainer.predict_probs(eval_available, eval_item_ids).numpy()

    p_AB, p_BC, p_ABC = probs

    print("\nPredicted probabilities:")
    print(f"  P(. | {{A,B}})   = {np.round(p_AB, 3)}")
    print(f"  P(. | {{B,C}})   = {np.round(p_BC, 3)}")
    print(f"  P(. | {{A,B,C}}) = {np.round(p_ABC, 3)}")

    pB_AB = p_AB[1]
    pB_BC = p_BC[1]
    pB_ABC = p_ABC[1]

    delta = pB_ABC - max(pB_AB, pB_BC)

    print("\nCompromise effect on B:")
    print(f"  P(B | ABC) = {pB_ABC:.3f}")
    print(f"  max(P(B | AB), P(B | BC)) = {max(pB_AB, pB_BC):.3f}")
    print(f"  Δ_comp = {delta:.3f}")

    if delta > 0:
        print("  -> Compromise effect present.")
    else:
        print("  -> No compromise effect detected.")


if __name__ == "__main__":
    main()
