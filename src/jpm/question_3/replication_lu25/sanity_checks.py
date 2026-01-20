# diagnostic_sparsity.py
# Deep dive into sparsity pattern recovery

import numpy as np

from jpm.config import SimConfig
from jpm.question_3.replication_lu25.estimators.blp import (
    build_matrices,
    invert_delta_contraction,
)
from jpm.question_3.replication_lu25.estimators.shrinkage import (
    estimate_shrinkage_sigma,
)
from jpm.question_3.replication_lu25.simulation.simulate import simulate_dataset


def main():
    np.random.seed(123)
    cfg = SimConfig()

    DGP = "DGP1"
    T = 25
    J = 15

    print("=" * 70)
    print("DETAILED SPARSITY PATTERN DIAGNOSTICS")
    print("=" * 70)

    # Simulate
    markets = simulate_dataset(DGP, T=T, J=J, cfg=cfg)

    # Run shrinkage
    sigma_shrink, beta_shrink, score_shrink, gamma_prob = estimate_shrinkage_sigma(
        markets, R=cfg.R0, n_iter=200, burn=100, v0=1e-4, v1=1.0
    )

    print(f"\nEstimated σ = {sigma_shrink:.4f}")
    print(f"Estimated β = {beta_shrink}")

    # Compute residuals
    delta_list = []
    for m in markets:
        delta_t = invert_delta_contraction(m["s"], m["p"], sigma_shrink, R=cfg.R0)
        delta_list.append(delta_t.numpy())
    delta_vec = np.concatenate(delta_list, axis=0)
    X, _ = build_matrices(markets, iv_type="nocost")
    residuals = delta_vec - X @ beta_shrink

    # Reshape to (T, J)
    gamma_matrix = gamma_prob.reshape(T, J)
    residuals_matrix = residuals.reshape(T, J)

    cutoff = int(cfg.sparse_frac * J)

    # ==================================================================
    # DIAGNOSTIC 1: Detection Rate by Product
    # ==================================================================
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 1: Detection Rate Across Markets")
    print("=" * 70)
    print("\nHow often is each product detected as non-zero (γ > 0.5)?")
    print(f"\n{'Product':>8} {'True η':>8} {'Detection Rate':>16} {'Expected':>12}")
    print("-" * 60)

    detection_rate = (gamma_matrix > 0.5).mean(axis=0)

    for j in range(J):
        true_nonzero = j < cutoff
        expected = "High (>0.9)" if true_nonzero else "Low (<0.2)"
        marker = " ***" if true_nonzero else ""
        print(
            f"{j:8d} {markets[0]['eta'][j]:8.1f} {detection_rate[j]:16.3f} {expected:>12}{marker}"
        )

    signal_detect = detection_rate[:cutoff].mean()
    noise_detect = detection_rate[cutoff:].mean()
    print("\nAverage detection rate:")
    print(f"  Products with η≠0 (0-{cutoff - 1}):  {signal_detect:.3f}  (want: >0.9)")
    print(f"  Products with η=0 ({cutoff}-{J - 1}):   {noise_detect:.3f}  (want: <0.2)")

    # ==================================================================
    # DIAGNOSTIC 2: Residual Magnitude
    # ==================================================================
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 2: Average Residual Magnitude")
    print("=" * 70)
    print("\nProducts with η≠0 should have large |residuals|")
    print(f"\n{'Product':>8} {'True η':>8} {'Avg |resid|':>14} {'Std(resid)':>14}")
    print("-" * 60)

    avg_abs_resid = np.abs(residuals_matrix).mean(axis=0)
    std_resid = residuals_matrix.std(axis=0)

    for j in range(J):
        marker = " ***" if j < cutoff else ""
        print(
            f"{j:8d} {markets[0]['eta'][j]:8.1f} {avg_abs_resid[j]:14.4f} {std_resid[j]:14.4f}{marker}"
        )

    signal_resid = avg_abs_resid[:cutoff].mean()
    noise_resid = avg_abs_resid[cutoff:].mean()
    print("\nAverage |residual|:")
    print(f"  Products with η≠0:  {signal_resid:.4f}")
    print(f"  Products with η=0:   {noise_resid:.4f}")
    print(f"  Ratio (signal/noise): {signal_resid / noise_resid:.2f}  (want: >2)")

    # ==================================================================
    # DIAGNOSTIC 3: Within-Market Specificity
    # ==================================================================
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 3: Within-Market Classification Metrics")
    print("=" * 70)
    print("\nFor each market, compute sensitivity & specificity")

    sensitivities = []
    specificities = []

    for t in range(T):
        gamma_t = gamma_matrix[t, :]

        # True positives: detected AND actually non-zero
        tp = (gamma_t[:cutoff] > 0.5).sum()
        # False negatives: not detected BUT actually non-zero
        fn = (gamma_t[:cutoff] <= 0.5).sum()
        # True negatives: not detected AND actually zero
        tn = (gamma_t[cutoff:] <= 0.5).sum()
        # False positives: detected BUT actually zero
        fp = (gamma_t[cutoff:] > 0.5).sum()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        sensitivities.append(sens)
        specificities.append(spec)

    print(f"\nAcross {T} markets:")
    print(f"  Average Sensitivity: {np.mean(sensitivities):.3f}  (want: >0.85)")
    print(f"  Average Specificity: {np.mean(specificities):.3f}  (want: >0.80)")
    print(f"  Median Sensitivity:  {np.median(sensitivities):.3f}")
    print(f"  Median Specificity:  {np.median(specificities):.3f}")

    # Show distribution
    high_spec_markets = (np.array(specificities) > 0.8).sum()
    print(f"  Markets with specificity > 0.8: {high_spec_markets}/{T}")

    # ==================================================================
    # DIAGNOSTIC 4: Gamma Distribution
    # ==================================================================
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 4: Distribution of γ Values")
    print("=" * 70)

    gamma_signal = gamma_matrix[:, :cutoff].flatten()
    gamma_noise = gamma_matrix[:, cutoff:].flatten()

    print("\nFor products with η≠0:")
    print(f"  Mean γ:   {gamma_signal.mean():.4f}  (want: ~1.0)")
    print(f"  Median γ: {np.median(gamma_signal):.4f}")

    print(f"  Min γ:    {gamma_signal.min():.4f}")
    print(f"  % with γ>0.9: {(gamma_signal > 0.9).mean() * 100:.1f}%")

    print("\nFor products with η=0:")
    print(f"  Mean γ:   {gamma_noise.mean():.4f}  (want: ~0.0)")
    print(f"  Median γ: {np.median(gamma_signal):.4f}")
    print(f"  Max γ:    {gamma_noise.max():.4f}")
    print(f"  % with γ<0.1: {(gamma_noise < 0.1).mean() * 100:.1f}%")

    # ==================================================================
    # DIAGNOSTIC 5: Correlation Structure
    # ==================================================================
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 5: Do Residuals Reflect True η Pattern?")
    print("=" * 70)

    # For each market, compute correlation between |residual| and |true η|
    correlations = []
    for t in range(T):
        true_eta = np.array([markets[t]["eta"][j] for j in range(J)])
        resid_t = residuals_matrix[t, :]

        corr = np.corrcoef(np.abs(true_eta), np.abs(resid_t))[0, 1]
        correlations.append(corr)

    print("\nCorrelation between |η| and |residual|:")
    print(f"  Mean:   {np.mean(correlations):.3f}  (want: >0.5)")
    print(f"  Median: {np.median(correlations):.3f}")
    print(f"  Min:    {np.min(correlations):.3f}")
    print(f"  Max:    {np.max(correlations):.3f}")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print("\n" + "=" * 70)
    print("SUMMARY ASSESSMENT")
    print("=" * 70)

    # Criteria
    criteria = {
        "Detection rate (signal)": (signal_detect, 0.9, ">"),
        "Detection rate (noise)": (noise_detect, 0.3, "<"),
        "Residual ratio (signal/noise)": (signal_resid / noise_resid, 2.0, ">"),
        "Avg sensitivity": (np.mean(sensitivities), 0.85, ">"),
        "Avg specificity": (np.mean(specificities), 0.80, ">"),
        "Avg |η|-|resid| correlation": (np.mean(correlations), 0.5, ">"),
    }

    print("\nCriteria Checklist:")
    for name, (value, threshold, direction) in criteria.items():
        if direction == ">":
            status = "✓" if value > threshold else "✗"
        else:
            status = "✓" if value < threshold else "✗"
        print(
            f"  {status} {name:<35}: {value:6.3f} (threshold: {direction}{threshold})"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
