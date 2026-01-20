"""
Single Replication Script
- One Monte Carlo draw
- Results saved to results/ folder in multiple formats
- Generates summary report and comparison table
"""

from datetime import datetime
from pathlib import Path

import numpy as np
from simulation.simulate import simulate_dataset

from jpm.config import SimConfig
from jpm.question_3.replication_lu25.estimators.blp import estimate_blp_sigma
from jpm.question_3.replication_lu25.estimators.lu25_map import (
    Lu25MapConfig,
    estimate_lu25_map,
)
from jpm.question_3.replication_lu25.estimators.shrinkage import (
    estimate_shrinkage_sigma,
)


def ensure_results_dir():
    """Create results directory if it doesn't exist, but should already"""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    return results_dir


def print_header(title, width=80):
    """Print a nice header"""
    print("\n" + "=" * width)
    print(f"{title:^{width}}")
    print("=" * width)


def print_section(title, width=80):
    """Print a section header"""
    print("\n" + "-" * width)
    print(f"  {title}")
    print("-" * width)


def print_status(message, status="info"):
    """Print status message with icon"""
    icons = {"info": "ℹ", "success": "✓", "error": "✗", "running": "▶", "waiting": "⏳"}
    icon = icons.get(status, "•")
    print(f"  {icon} {message}")


def format_bias(bias, threshold_good=0.05, threshold_ok=0.10):
    """Format bias with visual indicator"""
    abs_bias = abs(bias)
    if abs_bias < threshold_good:
        grade = "A"
        symbol = "✓"
    elif abs_bias < threshold_ok:
        grade = "B"
        symbol = "~"
    else:
        grade = "F"
        symbol = "✗"
    return f"{bias:+.4f} ({grade}) {symbol}"


def save_results_report(results, filename):  # noqa: C901
    """Save  report"""
    with open(filename, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("SINGLE REPLICATION TEST REPORT\n")
        f.write(f"Generated: {results['timestamp']}\n")
        f.write("=" * 80 + "\n\n")

        # Configuration
        f.write("CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        cfg = results["configuration"]
        f.write(f"  DGP:      {cfg['DGP']}\n")
        f.write(f"  Markets:  {cfg['T']}\n")
        f.write(f"  Products: {cfg['J']}\n")
        f.write(f"  Seed:     {cfg['seed']}\n\n")

        # True Parameters
        f.write("TRUE PARAMETERS\n")
        f.write("-" * 80 + "\n")
        true = results["true_parameters"]
        f.write(f"  σ*   = {true['sigma']:.4f}\n")
        f.write(f"  β_p* = {true['beta_p']:.4f}\n")
        f.write(f"  β_w* = {true['beta_w']:.4f}\n\n")

        # Data Diagnostics
        f.write("DATA DIAGNOSTICS\n")
        f.write("-" * 80 + "\n")
        diag = results["data_diagnostics"]
        f.write(f"  Outside share (market 0):  {diag['outside_share']:.4f}\n")
        f.write(f"  Corr(p, ξ) (market 0):     {diag['price_xi_corr']:.4f}\n\n")

        # Results by method
        for method_name in [
            "BLP + Cost IV",
            "BLP - Cost IV",
            "Shrinkage",
            "Lu25 MAP (No IV, No Inversion)",
        ]:
            if method_name not in results["estimates"]:
                continue

            f.write(f"\n{'=' * 80}\n")
            f.write(f"{method_name.upper()}\n")
            f.write(f"{'=' * 80}\n\n")

            est = results["estimates"][method_name]

            if est["status"] == "success":
                f.write(
                    f"{'Parameter':<12} {'Estimate':>12} {'True Value':>12} {'Bias':>12} {'Assessment':>15}\n"
                )
                f.write("-" * 80 + "\n")

                # sigma
                bias_sigma = est["sigma"] - true["sigma"]
                assess = (
                    "Excellent"
                    if abs(bias_sigma) < 0.05
                    else "Good" if abs(bias_sigma) < 0.10 else "Poor"
                )
                f.write(
                    f"{'σ':<12} {est['sigma']:>12.4f} {true['sigma']:>12.4f} {bias_sigma:>+12.4f} {assess:>15}\n"
                )

                # beta_p
                bias_bp = est["beta_p"] - true["beta_p"]
                assess = (
                    "Excellent"
                    if abs(bias_bp) < 0.05
                    else "Good" if abs(bias_bp) < 0.10 else "Poor"
                )
                f.write(
                    f"{'β_p':<12} {est['beta_p']:>12.4f} {true['beta_p']:>12.4f} {bias_bp:>+12.4f} {assess:>15}\n"
                )

                # beta_w
                bias_bw = est["beta_w"] - true["beta_w"]
                assess = (
                    "Excellent"
                    if abs(bias_bw) < 0.05
                    else "Good" if abs(bias_bw) < 0.10 else "Poor"
                )
                f.write(
                    f"{'β_w':<12} {est['beta_w']:>12.4f} {true['beta_w']:>12.4f} {bias_bw:>+12.4f} {assess:>15}\n"
                )

                if "neg_log_post" in est:
                    f.write(
                        f"\nNeg log posterior (MAP objective): {est['neg_log_post']:.6f}\n"
                    )
                else:
                    f.write(f"\nObjective: {est['objective']:.6f}\n")

                # Additional shrinkage info
                if method_name == "Shrinkage" and "avg_gamma" in est:
                    f.write("\nSparsity Metrics:\n")
                    f.write(f"  Average γ: {est['avg_gamma']:.4f}\n")

            else:
                f.write(f"STATUS: {est['status']}\n")
                f.write(f"ERROR: {est.get('error', 'Unknown error')}\n")

        # Sparsity Analysis
        if results["sparsity_analysis"]:
            f.write(f"\n{'=' * 80}\n")
            f.write("SPARSITY PATTERN RECOVERY\n")
            f.write(f"{'=' * 80}\n\n")

            sp = results["sparsity_analysis"]
            f.write("Signal Products (η ≠ 0):\n")
            f.write(f"  Average γ:  {sp['signal_gamma']:.4f}  (target: > 0.90)\n")
            f.write(
                f"  Assessment: {'✓ Excellent' if sp['signal_gamma'] > 0.90 else '~ Good' if sp['signal_gamma'] > 0.80 else '✗ Poor'}\n\n"
            )

            f.write("Noise Products (η = 0):\n")
            f.write(f"  Average γ:  {sp['noise_gamma']:.4f}  (target: < 0.20)\n")
            f.write(
                f"  Assessment: {'✓ Excellent' if sp['noise_gamma'] < 0.20 else '~ Good' if sp['noise_gamma'] < 0.30 else '✗ Poor'}\n\n"
            )

            f.write("Classification:\n")
            f.write(f"  Sensitivity: {sp['sensitivity']:.4f}\n")
            f.write(f"  Specificity: {sp['specificity']:.4f}\n")
            f.write(f"  Detected:    {sp['detected']}/{sp['total']} products\n")

        # Lu25 MAP sparsity analysis
        if results.get("lu25_map_sparsity"):
            f.write(f"\n{'=' * 80}\n")
            f.write("LU25 MAP SPARSITY RECOVERY (|d| > tau)\n")
            f.write(f"{'=' * 80}\n\n")

            spm = results["lu25_map_sparsity"]
            f.write(f"  tau:         {spm['tau']:.4f}\n")
            f.write(f"  Sensitivity: {spm['sensitivity']:.4f}\n")
            f.write(f"  Specificity: {spm['specificity']:.4f}\n")
            f.write(f"  Signal detect rate: {spm['signal_detect_rate']:.4f}\n")
            f.write(f"  Noise detect rate:  {spm['noise_detect_rate']:.4f}\n")
            f.write(
                f"  Detected:    {spm['detected']}/{spm['total']} product-markets\n\n"
            )

        # Comparison Summary
        f.write(f"\n{'=' * 80}\n")
        f.write("COMPARISON SUMMARY\n")
        f.write(f"{'=' * 80}\n\n")

        f.write(f"{'Method':<20} {'σ Bias':>12} {'β_p Bias':>12} {'Overall':>15}\n")
        f.write("-" * 80 + "\n")

        for method_name in [
            "BLP + Cost IV",
            "BLP - Cost IV",
            "Shrinkage",
            "Lu25 MAP (No IV, No Inversion)",
        ]:
            if method_name not in results["estimates"]:
                continue
            est = results["estimates"][method_name]
            if est["status"] == "success":
                bias_s = est["sigma"] - true["sigma"]
                bias_p = est["beta_p"] - true["beta_p"]

                # Overall grade
                max_bias = max(abs(bias_s), abs(bias_p))
                if max_bias < 0.05:
                    overall = "A (Excellent)"
                elif max_bias < 0.10:
                    overall = "B (Good)"
                elif max_bias < 0.30:
                    overall = "C (Fair)"
                else:
                    overall = "F (Poor)"

                f.write(
                    f"{method_name:<20} {bias_s:>+12.4f} {bias_p:>+12.4f} {overall:>15}\n"
                )

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print_status(f"Report saved: {filename}", "success")


def save_comparison_table(results, filename):
    """Save a compact comparison table (CSV format)"""
    with open(filename, "w") as f:
        f.write("method,parameter,estimate,true_value,bias,abs_bias,status\n")

        true = results["true_parameters"]

        for method_name in [
            "BLP + Cost IV",
            "BLP - Cost IV",
            "Shrinkage",
            "Lu25 MAP (No IV, No Inversion)",
        ]:
            if method_name not in results["estimates"]:
                continue

            est = results["estimates"][method_name]

            if est["status"] == "success":
                # sigma
                bias_s = est["sigma"] - true["sigma"]
                f.write(
                    f"{method_name},sigma,{est['sigma']:.6f},{true['sigma']:.6f},{bias_s:+.6f},{abs(bias_s):.6f},success\n"
                )

                # beta_p
                bias_p = est["beta_p"] - true["beta_p"]
                f.write(
                    f"{method_name},beta_p,{est['beta_p']:.6f},{true['beta_p']:.6f},{bias_p:+.6f},{abs(bias_p):.6f},success\n"
                )

                # beta_w
                bias_w = est["beta_w"] - true["beta_w"]
                f.write(
                    f"{method_name},beta_w,{est['beta_w']:.6f},{true['beta_w']:.6f},{bias_w:+.6f},{abs(bias_w):.6f},success\n"
                )
            else:
                f.write(f"{method_name},all,NA,NA,NA,NA,{est['status']}\n")

    print_status(f"Comparison table saved: {filename}", "success")


def main():  # noqa: C901
    # Setup
    np.random.seed(123)
    cfg = SimConfig()
    results_dir = ensure_results_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Configuration
    DGP = "DGP1"
    T = 25
    J = 15

    # Print welcome
    print_header("SINGLE REPLICATION TEST - Lu & Shimizu (2025)", 80)
    print(f"\n  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Show configuration
    print_section("Configuration")
    print(f"  DGP:          {DGP}")
    print(f"  Markets (T):  {T}")
    print(f"  Products (J): {J}")
    print("  Seed:         123")

    print_section("True Parameters")
    print(f"  σ*   = {cfg.sigma_star:.4f}")
    print(f"  β_p* = {cfg.beta_p_star:.4f}")
    print(f"  β_w* = {cfg.beta_w_star:.4f}")

    # Initialize results structure
    results = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {"DGP": DGP, "T": T, "J": J, "seed": 123},
        "true_parameters": {
            "sigma": cfg.sigma_star,
            "beta_p": cfg.beta_p_star,
            "beta_w": cfg.beta_w_star,
        },
        "data_diagnostics": {},
        "estimates": {},
        "sparsity_analysis": None,
    }

    # ===================================
    # Step 1: Simulate Data
    # ===================================
    print_section("Step 1: Data Simulation")
    print_status("Simulating dataset...", "running")

    markets = simulate_dataset(DGP, T=T, J=J, cfg=cfg)

    # Data diagnostics
    s0 = 1 - markets[0]["s"].sum()
    corr_px = np.corrcoef(markets[0]["p"], markets[0]["xi"])[0, 1]

    results["data_diagnostics"] = {
        "outside_share": float(s0),
        "price_xi_corr": float(corr_px),
    }

    print_status(f"Generated {len(markets)} markets", "success")
    print(f"    Outside share (market 0): {s0:.4f}")
    print(f"    Corr(p, ξ) (market 0):    {corr_px:.4f}")

    # ===================================
    # Step 2: BLP with Cost IV
    # ===================================
    print_section("Step 2: BLP with Cost IV (Valid Instruments)")
    print_status("Running estimation...", "running")

    try:
        sigma_cost, beta_cost, obj_cost = estimate_blp_sigma(
            markets, iv_type="cost", R=cfg.R0
        )

        results["estimates"]["BLP + Cost IV"] = {
            "status": "success",
            "sigma": float(sigma_cost),
            "beta_p": float(beta_cost[1]),
            "beta_w": float(beta_cost[2]),
            "beta_intercept": float(beta_cost[0]),
            "objective": float(obj_cost),
        }

        print_status("Estimation completed", "success")
        print(f"\n  {'Parameter':<12} {'Estimate':>12} {'True':>12} {'Bias':>15}")
        print(f"  {'-' * 55}")
        print(
            f"  {'σ':<12} {sigma_cost:>12.4f} {cfg.sigma_star:>12.4f} {format_bias(sigma_cost - cfg.sigma_star):>15}"
        )
        print(
            f"  {'β_p':<12} {beta_cost[1]:>12.4f} {cfg.beta_p_star:>12.4f} {format_bias(beta_cost[1] - cfg.beta_p_star):>15}"
        )
        print(
            f"  {'β_w':<12} {beta_cost[2]:>12.4f} {cfg.beta_w_star:>12.4f} {format_bias(beta_cost[2] - cfg.beta_w_star):>15}"
        )

    except Exception as e:
        print_status(f"Estimation failed: {str(e)}", "error")
        results["estimates"]["BLP + Cost IV"] = {"status": "error", "error": str(e)}

    # ===================================
    # Step 3: BLP without Cost IV
    # ===================================
    print_section("Step 3: BLP without Cost IV (Weak Instruments)")
    print_status("Running estimation...", "running")

    try:
        sigma_nocost, beta_nocost, obj_nocost = estimate_blp_sigma(
            markets, iv_type="nocost", R=cfg.R0
        )

        results["estimates"]["BLP - Cost IV"] = {
            "status": "success",
            "sigma": float(sigma_nocost),
            "beta_p": float(beta_nocost[1]),
            "beta_w": float(beta_nocost[2]),
            "beta_intercept": float(beta_nocost[0]),
            "objective": float(obj_nocost),
        }

        print_status("Estimation completed", "success")
        print(f"\n  {'Parameter':<12} {'Estimate':>12} {'True':>12} {'Bias':>15}")
        print(f"  {'-' * 55}")
        print(
            f"  {'σ':<12} {sigma_nocost:>12.4f} {cfg.sigma_star:>12.4f} {format_bias(sigma_nocost - cfg.sigma_star):>15}"
        )
        print(
            f"  {'β_p':<12} {beta_nocost[1]:>12.4f} {cfg.beta_p_star:>12.4f} {format_bias(beta_nocost[1] - cfg.beta_p_star):>15}"
        )
        print(
            f"  {'β_w':<12} {beta_nocost[2]:>12.4f} {cfg.beta_w_star:>12.4f} {format_bias(beta_nocost[2] - cfg.beta_w_star):>15}"
        )

        if abs(sigma_nocost - cfg.sigma_star) > 0.3:
            print_status("Large bias detected (expected with weak IVs)", "info")

    except Exception as e:
        print_status(f"Estimation failed: {str(e)}", "error")
        results["estimates"]["BLP - Cost IV"] = {"status": "error", "error": str(e)}

    # ===================================
    # Step 4: Shrinkage Estimator
    # ===================================
    print_section("Step 4: Shrinkage Estimator (No IV, uses sparsity)")
    print_status("Running MCMC estimation (this may take 1-2 minutes)...", "waiting")

    try:
        sigma_shrink, beta_shrink, score_shrink, gamma_prob = estimate_shrinkage_sigma(
            markets,
            R=cfg.R0,
            n_iter=200,
            burn=100,
            # v0=1e-4, could use as baseline
            v0=0.05,
            v1=1.0,
        )

        # Sparsity analysis
        gamma_matrix = gamma_prob.reshape(T, J)
        cutoff = int(cfg.sparse_frac * J)

        signal_gamma = gamma_matrix[:, :cutoff].mean()
        noise_gamma = gamma_matrix[:, cutoff:].mean()

        detected = (gamma_matrix > 0.5).sum()
        true_positives = (gamma_matrix[:, :cutoff] > 0.5).sum()
        false_positives = (gamma_matrix[:, cutoff:] > 0.5).sum()

        sensitivity = true_positives / (T * cutoff) if cutoff > 0 else 0
        specificity = 1 - false_positives / (T * (J - cutoff)) if J > cutoff else 0

        results["estimates"]["Shrinkage"] = {
            "status": "success",
            "sigma": float(sigma_shrink),
            "beta_p": float(beta_shrink[1]),
            "beta_w": float(beta_shrink[2]),
            "beta_intercept": float(beta_shrink[0]),
            "objective": float(score_shrink),
            "avg_gamma": float(gamma_prob.mean()),
        }

        results["sparsity_analysis"] = {
            "signal_gamma": float(signal_gamma),
            "noise_gamma": float(noise_gamma),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "detected": int(detected),
            "total": T * J,
            "cutoff": cutoff,
        }

        print_status("Estimation completed", "success")
        print(f"\n  {'Parameter':<12} {'Estimate':>12} {'True':>12} {'Bias':>15}")
        print(f"  {'-' * 55}")
        print(
            f"  {'σ':<12} {sigma_shrink:>12.4f} {cfg.sigma_star:>12.4f} {format_bias(sigma_shrink - cfg.sigma_star):>15}"
        )
        print(
            f"  {'β_p':<12} {beta_shrink[1]:>12.4f} {cfg.beta_p_star:>12.4f} {format_bias(beta_shrink[1] - cfg.beta_p_star):>15}"
        )
        print(
            f"  {'β_w':<12} {beta_shrink[2]:>12.4f} {cfg.beta_w_star:>12.4f} {format_bias(beta_shrink[2] - cfg.beta_w_star):>15}"
        )

        print("\n  Sparsity Recovery:")
        print(
            f"    Signal (η≠0): γ̄ = {signal_gamma:.4f}  {'✓' if signal_gamma > 0.85 else '✗'}"
        )
        print(
            f"    Noise (η=0):  γ̄ = {noise_gamma:.4f}  {'✓' if noise_gamma < 0.30 else '✗'}"
        )
        print(f"    Sensitivity:     {sensitivity:.4f}")
        print(f"    Specificity:     {specificity:.4f}")

    except Exception as e:
        print_status(f"Estimation failed: {str(e)}", "error")
        results["estimates"]["Shrinkage"] = {"status": "error", "error": str(e)}

    # ===================================
    # Step 5: Lu & Shimizu (2025)–style MAP estimator (Likelihood-based, no inversion, no IV)
    # ===================================
    print_section(
        "Step 5: Lu & Shimizu (2025)–style MAP estimator (Likelihood-based, sparse shocks, no inversion)"
    )
    print_status("Running MAP optimization (may take ~1-3 minutes)...", "waiting")

    try:
        lu_cfg = Lu25MapConfig(
            R=cfg.R0,
            steps=1200,
            lr=0.05,
            l1_strength=8.0,
            tau_detect=0.25,
            mu_sd=2.0,
            default_market_size=1000,
            seed=123,
        )
        lu_res = estimate_lu25_map(markets, cfg=lu_cfg)

        gamma_hat = lu_res["gamma_hat"].reshape(T, J)
        cutoff = int(cfg.sparse_frac * J)

        signal_detect = gamma_hat[:, :cutoff]
        noise_detect = gamma_hat[:, cutoff:]

        sens_map = signal_detect.mean()
        spec_map = 1.0 - noise_detect.mean()

        sigma_lu = lu_res["sigma_hat"]
        beta_lu = lu_res["beta_hat"]

        results["estimates"]["Lu25 MAP (No IV, No Inversion)"] = {
            "status": "success",
            "sigma": float(sigma_lu),
            "beta_p": float(beta_lu[1]),
            "beta_w": float(beta_lu[2]),
            "beta_intercept": float(beta_lu[0]),
            "neg_log_post": float(lu_res["final_neg_log_post"]),
            "objective": float(lu_res["final_neg_log_post"]),
        }

        results["lu25_map_sparsity"] = {
            "tau": float(lu_res["sparsity"]["tau_for_detection"]),
            "signal_detect_rate": float(signal_detect.mean()),
            "noise_detect_rate": float(noise_detect.mean()),
            "sensitivity": float(sens_map),
            "specificity": float(spec_map),
            "detected": int(gamma_hat.sum()),
            "total": int(T * J),
            "cutoff": int(cutoff),
        }

        results["lu25_map_diagnostics"] = {
            "sparsity": lu_res["sparsity"],
            "config": lu_res["config"],
            "history_tail": lu_res["history"][-5:] if lu_res.get("history") else None,
        }

        print_status("MAP estimation completed", "success")
        print(f"\n  {'Parameter':<12} {'Estimate':>12} {'True':>12} {'Bias':>15}")
        print(f"  {'-' * 55}")
        print(
            f"  {'σ':<12} {sigma_lu:>12.4f} {cfg.sigma_star:>12.4f} {format_bias(sigma_lu - cfg.sigma_star):>15}"
        )
        print(
            f"  {'β_p':<12} {beta_lu[1]:>12.4f} {cfg.beta_p_star:>12.4f} {format_bias(beta_lu[1] - cfg.beta_p_star):>15}"
        )
        print(
            f"  {'β_w':<12} {beta_lu[2]:>12.4f} {cfg.beta_w_star:>12.4f} {format_bias(beta_lu[2] - cfg.beta_w_star):>15}"
        )

        print("\n  MAP Sparsity Recovery (|d| > tau):")
        print(f"    tau:          {lu_res['sparsity']['tau_for_detection']:.4f}")
        print(f"    Signal (η≠0): detect rate = {sens_map:.4f}")
        print(f"    Noise (η=0):  detect rate = {noise_detect.mean():.4f}")
        print(f"    Sensitivity:     {sens_map:.4f}")
        print(f"    Specificity:     {spec_map:.4f}")
        print(f"    Detected:        {int(gamma_hat.sum())}/{T * J}")

    except Exception as e:
        print_status(f"Lu25 MAP estimation failed: {str(e)}", "error")
        results["estimates"]["Lu25 MAP (No IV, No Inversion)"] = {
            "status": "error",
            "error": str(e),
        }

    # ===================================
    # Save Results
    # ===================================
    print_section("Saving Results")

    # Generate filenames
    base_name = f"single_replication_{timestamp}"
    report_file = results_dir / f"{base_name}_report.txt"
    table_file = results_dir / f"{base_name}_comparison.csv"

    # Save in multiple formats
    save_results_report(results, report_file)
    save_comparison_table(results, table_file)

    # ===================================
    # Final Summary
    # ===================================
    print_header("SUMMARY", 80)

    successful = sum(
        1 for est in results["estimates"].values() if est["status"] == "success"
    )
    total = len(results["estimates"])

    print(f"\n  Estimations completed: {successful}/{total}")
    print("\n  Results saved to: results/")
    print(f"    • {report_file.name}")
    print(f"    • {table_file.name}")

    # Quick assessment
    if successful == total:
        print_status("All estimations successful!", "success")

        # Check if results are reasonable
        all_good = True
        for method, est in results["estimates"].items():
            if est["status"] == "success":
                bias_sigma = abs(est["sigma"] - cfg.sigma_star)
                if method == "BLP + Cost IV" and bias_sigma > 0.10:
                    all_good = False
                elif (
                    method in ["Shrinkage", "Lu25 MAP (No IV, No Inversion)"]
                    and bias_sigma > 0.15
                ):
                    all_good = False

        if all_good:
            print_status("Results look good! Ready for Monte Carlo.", "success")
        else:
            print_status("Some unusual biases detected - review results", "info")
    else:
        print_status(f"{total - successful} estimation(s) failed", "error")

    print("\n" + "=" * 80 + "\n")

    return results


if __name__ == "__main__":
    main()
