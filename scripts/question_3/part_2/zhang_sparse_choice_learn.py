from __future__ import annotations

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")


import numpy as np
import tensorflow as tf

from jpm.question_3.deephalo_extension.zhang_sparse_choice_learn import (
    ZhangSparseConfig,
    build_and_init_model,
    choice_dataset_to_tensors,
    run_one_ablation,
    simulate_context_plus_sparse,
)


def main():
    cfg = ZhangSparseConfig()
    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)

    # data
    T = 25
    J_inside = 15
    N_t = 1000
    n_items = J_inside + 1

    print("Simulating data...")
    dataset, meta = simulate_context_plus_sparse(
        T=T,
        J_inside=J_inside,
        N_t=N_t,
        seed=cfg.seed,
        d_embed=cfg.d_embed,
        n_blocks=cfg.n_blocks,
        n_heads=cfg.n_heads,
        sparse_frac_nonzero=0.4,
        mu_sd=2.0,
        d_sd=1.0,
    )
    data = choice_dataset_to_tensors(dataset, n_items=n_items)

    # Build a base model once to get a common weight init for fair ablations
    base_model = build_and_init_model(cfg, n_items, T, J_inside, init_from_weights=None)
    init_weights = base_model.get_weights()

    # Run the three ablations
    results = {}

    _, results["DeepHalo-only"] = run_one_ablation(
        name="DeepHalo-only (mu=d frozen at 0)",
        cfg=cfg,
        data=data,
        n_items=n_items,
        T=T,
        J_inside=J_inside,
        learn_mu=False,
        learn_d=False,
        init_weights=init_weights,
    )

    _, results["Halo+mu"] = run_one_ablation(
        name="DeepHalo + mu (d frozen at 0)",
        cfg=cfg,
        data=data,
        n_items=n_items,
        T=T,
        J_inside=J_inside,
        learn_mu=True,
        learn_d=False,
        init_weights=init_weights,
    )

    full_model, results["Full"] = run_one_ablation(
        name="Full (DeepHalo + mu + sparse d)",
        cfg=cfg,
        data=data,
        n_items=n_items,
        T=T,
        J_inside=J_inside,
        learn_mu=True,
        learn_d=True,
        init_weights=init_weights,
    )

    # Summary table
    print("\n=== Summary (in-sample) ===")
    print(f"{'Model':<32} {'NLL':>10} {'mean|d|':>10} {'std(mu)':>10}")
    print("-" * 68)
    for k in ["DeepHalo-only", "Halo+mu", "Full"]:
        r = results[k]
        print(
            f"{k:<32} {r['nll']:>10.4f} {r['mean_abs_d']:>10.4f} {r['std_mu']:>10.4f}"
        )

    # Optional: support recovery for Full only
    d_hat = full_model.d.numpy()
    gamma_hat = (np.abs(d_hat) > cfg.tau_detect).astype(np.int32)
    gamma_true = meta["gamma_true"]

    true_nz = gamma_true == 1
    true_z = gamma_true == 0
    sens = (gamma_hat[true_nz] == 1).mean() if true_nz.sum() > 0 else np.nan
    spec = (gamma_hat[true_z] == 0).mean() if true_z.sum() > 0 else np.nan

    print("\nFull model support recovery (|d| > tau)")
    print("  tau:", cfg.tau_detect)
    print("  sensitivity:", float(sens))
    print("  specificity:", float(spec))

    print("\nDone.")


if __name__ == "__main__":
    main()
