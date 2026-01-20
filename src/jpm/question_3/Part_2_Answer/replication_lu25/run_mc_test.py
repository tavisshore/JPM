"""run_mc_final.py

Monte Carlo replication driver for Lu & Shimizu (2025):
  • BLP with cost IV (valid instruments)
  • BLP without cost IV (weak instruments)
  • Shrinkage estimator (no IV)

This version is aligned to Table 1 in the paper:
  • grid of (T, J) ∈ {(5,25), (5,100), (15,25), (15,100)}
  • default market size N_t taken from SimConfig.default_market_size
  • shrinkage spike-and-slab variances set to (tau0^2, tau1^2) via (v0, v1)

Outputs (written to ./results):
  • a log file (*.txt)
  • a compact CSV summary (*.csv)
  • OPTIONAL: a LaTeX side-by-side comparison table for Table 1 (DGP1–DGP2)

Usage examples:
  python run_mc_final.py --R_mc 50
  python run_mc_final.py --full_grid --R_mc 50
  python run_mc_final.py --R_mc 10 --shrink_n_iter 400 --shrink_burn 200

Notes:
  • This script intentionally avoids JSON outputs.
  • If you need reproducibility across replications, keep the base seed fixed.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Handle both flat and organized project structures
try:
    from simulation.config import SimConfig
    from simulation.simulate import simulate_dataset
    from estimators.blp import estimate_blp_sigma
    from estimators.shrinkage import estimate_shrinkage_sigma
except ImportError:  # pragma: no cover
    from config import SimConfig
    from simulate import simulate_dataset
    from blp import estimate_blp_sigma
    from shrinkage import estimate_shrinkage_sigma


# -----------------------------
# Logging utilities
# -----------------------------

class OutputLogger:
    """Write to both console and a log file."""

    def __init__(self, filename: Path):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def close(self) -> None:
        try:
            self.log.close()
        except Exception:
            pass


def ensure_results_dir() -> Path:
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    return results_dir


def print_progress_bar(iteration: int, total: int, prefix: str = "", suffix: str = "", length: int = 40) -> None:
    """Simple progress bar."""
    percent = 100 * (iteration / float(total))
    filled_length = int(length * iteration // total)
    bar = "█" * filled_length + "░" * (length - filled_length)
    print(f"\r{prefix} |{bar}| {percent:5.1f}% {suffix}", end="", flush=True)
    if iteration == total:
        print()


# -----------------------------
# Paper Table 1 (for LaTeX comparison)
# -----------------------------

# Values are (bias, sd) for each parameter and method.
# Methods keys: "blp_iv" (BLP+cost IV), "blp_noiv" (BLP−cost IV), "shrink" (Shrinkage)
PAPER_TABLE1: Dict[str, Dict[Tuple[int, int], Dict[str, Dict[str, Tuple[float, float]]]]] = {
    "DGP1": {
        (5, 25): {
            "sigma": {"blp_iv": (0.134, 0.127), "blp_noiv": (0.132, 0.130), "shrink": (0.169, 0.134)},
            "beta_p": {"blp_iv": (0.006, 0.146), "blp_noiv": (0.014, 0.143), "shrink": (0.020, 0.151)},
        },
        (5, 100): {
            "sigma": {"blp_iv": (0.072, 0.120), "blp_noiv": (0.075, 0.122), "shrink": (0.095, 0.126)},
            "beta_p": {"blp_iv": (0.018, 0.103), "blp_noiv": (0.018, 0.103), "shrink": (0.034, 0.106)},
        },
        (15, 25): {
            "sigma": {"blp_iv": (0.067, 0.070), "blp_noiv": (0.067, 0.070), "shrink": (0.084, 0.073)},
            "beta_p": {"blp_iv": (0.009, 0.085), "blp_noiv": (0.012, 0.084), "shrink": (0.016, 0.088)},
        },
        (15, 100): {
            "sigma": {"blp_iv": (0.060, 0.066), "blp_noiv": (0.060, 0.066), "shrink": (0.076, 0.068)},
            "beta_p": {"blp_iv": (0.012, 0.070), "blp_noiv": (0.012, 0.070), "shrink": (0.020, 0.071)},
        },
    },
    "DGP2": {
        (5, 25): {
            "sigma": {"blp_iv": (0.217, 0.173), "blp_noiv": (1.008, 0.094), "shrink": (0.262, 0.185)},
            "beta_p": {"blp_iv": (-0.007, 0.144), "blp_noiv": (0.525, 0.113), "shrink": (-0.010, 0.153)},
        },
        (5, 100): {
            "sigma": {"blp_iv": (0.117, 0.136), "blp_noiv": (0.433, 0.072), "shrink": (0.158, 0.146)},
            "beta_p": {"blp_iv": (0.007, 0.104), "blp_noiv": (0.185, 0.066), "shrink": (0.008, 0.111)},
        },
        (15, 25): {
            "sigma": {"blp_iv": (0.099, 0.080), "blp_noiv": (0.650, 0.071), "shrink": (0.141, 0.084)},
            "beta_p": {"blp_iv": (-0.003, 0.085), "blp_noiv": (0.244, 0.067), "shrink": (-0.001, 0.089)},
        },
        (15, 100): {
            "sigma": {"blp_iv": (0.082, 0.071), "blp_noiv": (0.287, 0.042), "shrink": (0.118, 0.074)},
            "beta_p": {"blp_iv": (0.002, 0.070), "blp_noiv": (0.096, 0.042), "shrink": (0.001, 0.073)},
        },
    },
}


# -----------------------------
# Monte Carlo core
# -----------------------------

@dataclass
class ShrinkageSettings:
    n_iter: int = 200
    burn: int = 100
    v0: float = 1e-4   # aligns with tau0^2 used in this repo's single-replication script
    v1: float = 1.0    # aligns with tau1^2


@dataclass
class MCCellResult:
    # arrays of length R_mc
    sigma: np.ndarray
    beta_p: np.ndarray
    beta_w: np.ndarray


def run_mc_cell(
    DGP: str,
    T: int,
    J: int,
    R_mc: int,
    cfg: SimConfig,
    base_seed: int,
    iv_type: str,
) -> MCCellResult:
    """Run BLP Monte Carlo cell."""

    sigma = np.zeros(R_mc)
    beta_p = np.zeros(R_mc)
    beta_w = np.zeros(R_mc)

    for r in range(R_mc):
        seed_r = base_seed + 10_000 * r
        markets = simulate_dataset(DGP, T=T, J=J, cfg=cfg, seed=seed_r)
        sigma_hat, beta_hat, _ = estimate_blp_sigma(markets, iv_type=iv_type, R=cfg.R0)

        sigma[r] = float(sigma_hat)
        beta_p[r] = float(beta_hat[1])
        beta_w[r] = float(beta_hat[2])

        print_progress_bar(r + 1, R_mc, prefix=f"  BLP({iv_type:6}) T={T:<2} J={J:<3}", suffix=f"({r+1}/{R_mc})")

    return MCCellResult(sigma=sigma, beta_p=beta_p, beta_w=beta_w)


def run_mc_cell_shrinkage(
    DGP: str,
    T: int,
    J: int,
    R_mc: int,
    cfg: SimConfig,
    base_seed: int,
    settings: ShrinkageSettings,
) -> MCCellResult:
    """Run Shrinkage Monte Carlo cell."""

    sigma = np.zeros(R_mc)
    beta_p = np.zeros(R_mc)
    beta_w = np.zeros(R_mc)

    for r in range(R_mc):
        seed_r = base_seed + 10_000 * r
        markets = simulate_dataset(DGP, T=T, J=J, cfg=cfg, seed=seed_r)
        sigma_hat, beta_hat, _, _ = estimate_shrinkage_sigma(
            markets,
            R=cfg.R0,
            n_iter=settings.n_iter,
            burn=settings.burn,
            v0=settings.v0,
            v1=settings.v1,
        )

        sigma[r] = float(sigma_hat)
        beta_p[r] = float(beta_hat[1])
        beta_w[r] = float(beta_hat[2])

        print_progress_bar(r + 1, R_mc, prefix=f"  Shrinkage  T={T:<2} J={J:<3}", suffix=f"({r+1}/{R_mc})")

    return MCCellResult(sigma=sigma, beta_p=beta_p, beta_w=beta_w)


def summarize(arr: np.ndarray, true_val: float) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=float)
    mean = float(arr.mean())
    bias = float(mean - true_val)
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")
    rmse = float(np.sqrt(bias * bias + sd * sd)) if not np.isnan(sd) else float("nan")
    return {"mean": mean, "bias": bias, "sd": sd, "rmse": rmse}


def print_table_cell_header(DGP: str, T: int, J: int, cfg: SimConfig) -> None:
    print("\n" + "=" * 90)
    print(f"Cell: {DGP}, T={T}, J={J}, N_t={cfg.default_market_size}, R0={cfg.R0}")
    print("=" * 90)


def print_summary_block(title: str, stats: Dict[str, Dict[str, float]], cfg: SimConfig) -> None:
    print("\n" + "-" * 90)
    print(f"{title}")
    print("-" * 90)
    print(f"{'Param':<10} {'True':>10} {'Mean':>12} {'Bias':>12} {'SD':>12} {'RMSE':>12}")
    print("-" * 90)

    mapping = {
        "sigma": ("σ", cfg.sigma_star),
        "beta_p": ("β_p", cfg.beta_p_star),
        "beta_w": ("β_w", cfg.beta_w_star),
    }
    for key, (sym, true_v) in mapping.items():
        s = stats[key]
        print(f"{sym:<10} {true_v:>10.4f} {s['mean']:>12.4f} {s['bias']:>12.4f} {s['sd']:>12.4f} {s['rmse']:>12.4f}")


def save_summary_csv(rows: List[Dict[str, object]], out_csv: Path) -> None:
    import csv

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "dgp",
                "T",
                "J",
                "method",
                "param",
                "true",
                "mean",
                "bias",
                "sd",
                "rmse",
                "R_mc",
                "N_t",
                "R0",
                "seed",
            ],
        )
        w.writeheader()
        w.writerows(rows)


def write_table1_latex(paper: dict, ours: dict, out_tex: Path) -> None:
    """Write a side-by-side LaTeX table (Paper vs Our results) for DGP1–DGP2.

    Expects ours[dgp][(T,J)][param][method] = (bias, sd)
    """

    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Replication of Table 1: Paper vs. Our Monte Carlo results (bias, standard deviation).}")
    lines.append(r"\label{tab:table1_replication}")
    lines.append(r"\begin{tabular}{ll l cc cc}")
    lines.append(r"\toprule")
    lines.append(r"DGP & (T,J) & Method & \multicolumn{2}{c}{Paper} & \multicolumn{2}{c}{Ours}\\")
    lines.append(r"\cmidrule(lr){4-5} \cmidrule(lr){6-7}")
    lines.append(r" & & & Bias & SD & Bias & SD\\")
    lines.append(r"\midrule")

    method_order = [
        ("blp_iv", "BLP + cost IV"),
        ("blp_noiv", "BLP − cost IV"),
        ("shrink", "Shrinkage"),
    ]
    param_order = [("sigma", r"$\sigma$"), ("beta_p", r"$\beta_p$")]

    for dgp in ["DGP1", "DGP2"]:
        for (T, J) in [(5, 25), (5, 100), (15, 25), (15, 100)]:
            for param_key, _param_tex in param_order:
                first_row = True
                for mkey, mname in method_order:
                    pbias, psd = paper[dgp][(T, J)][param_key][mkey]
                    obias, osd = ours.get(dgp, {}).get((T, J), {}).get(param_key, {}).get(mkey, (float("nan"), float("nan")))

                    dgp_cell = dgp if first_row and param_key == "sigma" and mkey == "blp_iv" else ""
                    tj_cell = f"({T},{J})" if first_row and mkey == "blp_iv" else ""

                    lines.append(
                        f"{dgp_cell} & {tj_cell} & {mname} & {pbias:+.3f} & {psd:.3f} & {obias:+.3f} & {osd:.3f} \\")
                    first_row = False

                lines.append(r"\addlinespace[2pt]")

        lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------
# Main driver
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo replication aligned to Lu & Shimizu (2025) Table 1")
    parser.add_argument("--R_mc", type=int, default=50, help="number of Monte Carlo replications per cell")
    parser.add_argument("--seed", type=int, default=123, help="base RNG seed")
    parser.add_argument("--full_grid", action="store_true", help="run DGP1–DGP4 over Table-1 (T,J) grid")
    parser.add_argument("--write_latex", action="store_true", help="write side-by-side LaTeX table for DGP1–DGP2")

    # shrinkage hyperparameters
    parser.add_argument("--shrink_n_iter", type=int, default=200)
    parser.add_argument("--shrink_burn", type=int, default=100)
    parser.add_argument("--v0", type=float, default=1e-4, help="spike variance (tau0^2)")
    parser.add_argument("--v1", type=float, default=1.0, help="slab variance (tau1^2)")

    args = parser.parse_args()

    cfg = SimConfig()
    results_dir = ensure_results_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Table 1 grid in the paper
    grid_TJ: List[Tuple[int, int]] = [(5, 25), (5, 100), (15, 25), (15, 100)]
    dgps = ["DGP1", "DGP2", "DGP3", "DGP4"] if args.full_grid else ["DGP1"]

    # Setup output logging
    out_log = results_dir / f"mc_table1_{'full' if args.full_grid else 'single'}_R{args.R_mc}_{timestamp}.txt"
    logger = OutputLogger(out_log)
    sys.stdout = logger

    print("\n" + "#" * 90)
    print("MONTE CARLO SIMULATION (Table 1 alignment)".center(90))
    print("#" * 90)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"R_mc: {args.R_mc}")
    print(f"Base seed: {args.seed}")
    print(f"Default market size N_t: {cfg.default_market_size}")
    print(f"Consumer draws R0: {cfg.R0}")
    print(f"Shrinkage: n_iter={args.shrink_n_iter}, burn={args.shrink_burn}, v0={args.v0:g}, v1={args.v1:g}")

    shrink_settings = ShrinkageSettings(n_iter=args.shrink_n_iter, burn=args.shrink_burn, v0=args.v0, v1=args.v1)

    csv_rows: List[Dict[str, object]] = []

    # For LaTeX comparison (DGP1–DGP2 only)
    ours_for_latex: Dict[str, Dict[Tuple[int, int], Dict[str, Dict[str, Tuple[float, float]]]]] = {}

    for dgp in dgps:
        for (T, J) in grid_TJ:
            print_table_cell_header(dgp, T, J, cfg)

            # --- BLP + cost IV
            res_iv = run_mc_cell(dgp, T, J, args.R_mc, cfg, args.seed, iv_type="cost")
            stats_iv = {
                "sigma": summarize(res_iv.sigma, cfg.sigma_star),
                "beta_p": summarize(res_iv.beta_p, cfg.beta_p_star),
                "beta_w": summarize(res_iv.beta_w, cfg.beta_w_star),
            }
            print_summary_block("BLP + cost IV", stats_iv, cfg)

            # --- BLP − cost IV
            res_noiv = run_mc_cell(dgp, T, J, args.R_mc, cfg, args.seed, iv_type="nocost")
            stats_noiv = {
                "sigma": summarize(res_noiv.sigma, cfg.sigma_star),
                "beta_p": summarize(res_noiv.beta_p, cfg.beta_p_star),
                "beta_w": summarize(res_noiv.beta_w, cfg.beta_w_star),
            }
            print_summary_block("BLP − cost IV", stats_noiv, cfg)

            # --- Shrinkage
            res_sh = run_mc_cell_shrinkage(dgp, T, J, args.R_mc, cfg, args.seed, shrink_settings)
            stats_sh = {
                "sigma": summarize(res_sh.sigma, cfg.sigma_star),
                "beta_p": summarize(res_sh.beta_p, cfg.beta_p_star),
                "beta_w": summarize(res_sh.beta_w, cfg.beta_w_star),
            }
            print_summary_block("Shrinkage", stats_sh, cfg)

            # Append CSV rows
            for method, stats in [
                ("BLP+costIV", stats_iv),
                ("BLP-noCostIV", stats_noiv),
                ("Shrinkage", stats_sh),
            ]:
                for param_key, true_val in [
                    ("sigma", cfg.sigma_star),
                    ("beta_p", cfg.beta_p_star),
                    ("beta_w", cfg.beta_w_star),
                ]:
                    s = stats[param_key]
                    csv_rows.append(
                        {
                            "dgp": dgp,
                            "T": T,
                            "J": J,
                            "method": method,
                            "param": param_key,
                            "true": true_val,
                            "mean": s["mean"],
                            "bias": s["bias"],
                            "sd": s["sd"],
                            "rmse": s["rmse"],
                            "R_mc": args.R_mc,
                            "N_t": cfg.default_market_size,
                            "R0": cfg.R0,
                            "seed": args.seed,
                        }
                    )

            # Collect for LaTeX table (only sigma and beta_p, only DGP1–DGP2)
            if dgp in ("DGP1", "DGP2"):
                ours_for_latex.setdefault(dgp, {}).setdefault((T, J), {})
                ours_for_latex[dgp][(T, J)].setdefault("sigma", {})
                ours_for_latex[dgp][(T, J)].setdefault("beta_p", {})

                ours_for_latex[dgp][(T, J)]["sigma"]["blp_iv"] = (stats_iv["sigma"]["bias"], stats_iv["sigma"]["sd"])
                ours_for_latex[dgp][(T, J)]["sigma"]["blp_noiv"] = (stats_noiv["sigma"]["bias"], stats_noiv["sigma"]["sd"])
                ours_for_latex[dgp][(T, J)]["sigma"]["shrink"] = (stats_sh["sigma"]["bias"], stats_sh["sigma"]["sd"])

                ours_for_latex[dgp][(T, J)]["beta_p"]["blp_iv"] = (stats_iv["beta_p"]["bias"], stats_iv["beta_p"]["sd"])
                ours_for_latex[dgp][(T, J)]["beta_p"]["blp_noiv"] = (stats_noiv["beta_p"]["bias"], stats_noiv["beta_p"]["sd"])
                ours_for_latex[dgp][(T, J)]["beta_p"]["shrink"] = (stats_sh["beta_p"]["bias"], stats_sh["beta_p"]["sd"])

    # Save CSV
    out_csv = results_dir / f"mc_table1_{'full' if args.full_grid else 'single'}_R{args.R_mc}_{timestamp}.csv"
    save_summary_csv(csv_rows, out_csv)

    # Optional LaTeX comparison
    if args.write_latex:
        out_tex = results_dir / f"table1_paper_vs_ours_R{args.R_mc}_{timestamp}.tex"
        write_table1_latex(PAPER_TABLE1, ours_for_latex, out_tex)
        print(f"\nLaTeX table written: {out_tex.name}")

    print("\n" + "=" * 90)
    print("Saved outputs:".center(90))
    print(f"  {out_log.name}")
    print(f"  {out_csv.name}")
    if args.write_latex:
        print(f"  {out_tex.name}")
    print("=" * 90 + "\n")

    # Restore stdout
    sys.stdout = logger.terminal
    logger.close()


if __name__ == "__main__":
    main()
