import numpy as np
import tensorflow as tf
from choice_learn_ext.models.deep_context.deep_halo_core import DeepContextChoiceModel
from choice_learn_ext.models.deep_context.trainer import Trainer


def _simulate_decoy_data(N_per_type=300):
    """
    Two types of choice sets over 3 items {0,1,2}:

    A = {0,1}:
        P(0|A)=0.5, P(1|A)=0.5, P(2|A)=0

    B = {0,1,2} with decoy 2 making 0 more attractive:
        P(0|B)=0.7, P(1|B)=0.2, P(2|B)=0.1
    """
    J = 3
    mask_A = np.array([1, 1, 0], dtype=np.float32)
    mask_B = np.array([1, 1, 1], dtype=np.float32)

    p_A = np.array([0.5, 0.5, 0.0])
    p_B = np.array([0.7, 0.2, 0.1])

    avail_list = []
    choices_list = []

    # Generate samples for A
    for _ in range(N_per_type):
        avail_list.append(mask_A)
        choices_list.append(np.random.choice(J, p=p_A))

    # Generate samples for B
    for _ in range(N_per_type):
        avail_list.append(mask_B)
        choices_list.append(np.random.choice(J, p=p_B))

    available = tf.convert_to_tensor(np.stack(avail_list), dtype=tf.float32)
    choices = tf.convert_to_tensor(np.array(choices_list), dtype=tf.int32)
    item_ids = tf.tile(tf.range(J)[tf.newaxis, :], [available.shape[0], 1])

    return available, item_ids, choices, mask_A, mask_B


def test_decoy_effect_learned():
    tf.random.set_seed(0)
    np.random.seed(0)

    available, item_ids, choices, mask_A, mask_B = _simulate_decoy_data()

    num_items = 3
    model = DeepContextChoiceModel(num_items=num_items)
    trainer = Trainer(model, lr=5e-3)

    # Train DeepHalo on the synthetic decoy data
    trainer.fit_arrays(
        available=available,
        choices=choices,
        item_ids=item_ids,
        batch_size=64,
        epochs=40,
        verbose=0,
    )

    # Evaluate probabilities on A and B
    avail_eval = tf.convert_to_tensor(np.stack([mask_A, mask_B]), dtype=tf.float32)
    item_ids_eval = tf.tile(tf.range(num_items)[tf.newaxis, :], [2, 1])

    out = model({"available": avail_eval, "item_ids": item_ids_eval}, training=False)
    probs = tf.exp(out["log_probs"]).numpy()

    p_A = probs[0]  # for set A={0,1}
    p_B = probs[1]  # for set B={0,1,2} with decoy

    # DeepHalo should pick up the decoy effect:
    # P(0|B) > P(0|A) by a noticeable margin.
    assert p_B[0] > p_A[0] + 0.05
