"""run_replication_study.py

Paper-aligned + repo-consistent replication driver for Lu & Shimizu (2025) Section 4.

What this script does
---------------------
Uses THIS REPO'S implementation (simulation + estimators) to run the Monte Carlo study
and produce Table-1-style summaries.

It runs the benchmark estimators used in the simulation section:
  1) BLP + Cost IV (valid instruments)
  2) BLP − Cost IV (weak instruments)
  3) Shrinkage (Bayesian-style sparsity estimator implemented in this repo)
Optionally (if present in the repo):
  4) Lu25 MAP (likelihood-based, sparse shocks, no inversion)

Paper alignment principles
--------------------------
- Uses SimConfig from replication_lu25/simulation/config.py as the single source of truth
  for: true parameters, market size N_t (cfg.Nt), integration draws (cfg.R0), sparsity fraction, etc.
- Uses simulate_dataset(...) from replication_lu25/simulation/simulate.py.
- Uses estimate_blp_sigma / estimate_shrinkage_sigma / estimate_lu25_map from replication_lu25/estimators.

Outputs
-------
- Console output is also logged to results/.
- A CSV summary table is saved for downstream LaTeX table construction.

How to run
----------
From the repo root:
    python replication_lu25/run_replication_study.py

You can edit DEFAULT_GRID below to match Table 1 exactly.

Notes
-----
This file intentionally avoids re-implementing BLP or shrinkage logic.
If results differ from the paper, the right place to investigate is:
  - simulation/config.py (DGP and parameter settings)
  - simulation/simulate.py (data generation)
  - estimators/blp.py, estimators/shrinkage.py (estimation routines)
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from jpm.config import SimConfig, StudyConfig
from jpm.question_3.replication_lu25.estimators.blp import estimate_blp_sigma
from jpm.question_3.replication_lu25.estimators.shrinkage import (
    estimate_shrinkage_sigma,
)
from jpm.question_3.replication_lu25.simulation.simulate import simulate_dataset
from jpm.question_3.replication_lu25.utils import (
    OutputLogger,
    init_storage,
    print_method_table,
    save_summary_csv,
    summarize_vec,
)

HAS_LU25_MAP = False
estimate_lu25_map = None
Lu25MapConfig = None

try:
    from jpm.question_3.replication_lu25.estimators.lu25_map import (
        Lu25MapConfig,
        estimate_lu25_map,
    )

    print("\nLu25 MAP estimator found and will be included in the study.\n")
    HAS_LU25_MAP = True
except Exception:
    pass

# -----------------------------------------------------------------------------
# Repo import handling (FIXED)
# -----------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
RESULTS_DIR = Path("results/question_3/part_2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# IMPORTANT: edit this grid to match Table 1 exactly.
# If the paper uses multiple (T,J) cells, list them here.
DEFAULT_GRID: List[Tuple[str, int, int]] = [
    ("DGP1", 25, 15),
    ("DGP2", 25, 15),
    ("DGP3", 25, 15),
    ("DGP4", 25, 15),
]

# -----------------------------------------------------------------------------
# Estimator runners (repo-consistent)
# -----------------------------------------------------------------------------


def run_blp_mc(markets, cfg: SimConfig, iv_type: str):
    return estimate_blp_sigma(markets, iv_type=iv_type, R=cfg.R0)


def run_lu25_map_mc(markets, study: StudyConfig, cfg: SimConfig, rep_seed: int):
    if not HAS_LU25_MAP:
        raise RuntimeError("Lu25 MAP estimator not available in this repo layout.")

    lu_cfg = Lu25MapConfig(
        R=cfg.R0,
        steps=study.lu_steps,
        lr=study.lu_lr,
        l1_strength=study.lu_l1_strength,
        mu_sd=study.lu_mu_sd,
        tau_detect=study.lu_tau_detect,
        default_market_size=int(getattr(cfg, "Nt", 1000)),
        seed=rep_seed,
    )
    return estimate_lu25_map(markets, cfg=lu_cfg)


# -----------------------------------------------------------------------------
# One cell (DGP,T,J)
# -----------------------------------------------------------------------------


def run_cell(  # noqa: C901
    dgp: str, T: int, J: int, study: StudyConfig, cfg: SimConfig, include_lu25_map: bool
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Run one Table-1-style cell and return summaries."""

    true_params = {
        "sigma": float(cfg.sigma_star),
        "beta_p": float(cfg.beta_p_star),
        "beta_w": float(cfg.beta_w_star),
    }

    # storage
    blp_iv = init_storage(study.R_mc)
    blp_noiv = init_storage(study.R_mc)
    shrink = init_storage(study.R_mc)
    lu25 = init_storage(study.R_mc) if include_lu25_map and HAS_LU25_MAP else None

    # optional sparsity tracking
    # shrinkage returns gamma probabilities (T*J,) per rep in your repo
    shrink_gamma_list: List[np.ndarray] = []
    lu25_gamma_list: List[np.ndarray] = []

    print(f"\n{'=' * 90}")
    print(f"Cell: {dgp}, T={T}, J={J}, N_t={getattr(cfg, 'Nt', 'NA')}, R0={cfg.R0}")
    print(f"{'=' * 90}")

    pbar = tqdm(
        total=study.R_mc,
        desc=f"  {dgp} T={T:<3d} J={J:<3d}",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}]",
    )

    for r in range(study.R_mc):
        pbar.set_postfix_str(f"({r + 1}/{study.R_mc})")

        rep_seed = study.seed + r
        np.random.seed(rep_seed)

        markets = simulate_dataset(dgp, T=T, J=J, cfg=cfg)

        # 1) BLP + cost IV
        try:
            sigma_hat, beta_hat, _ = estimate_blp_sigma(
                markets, iv_type="cost", R=cfg.R0
            )
            blp_iv["sigma"][r] = float(sigma_hat)
            blp_iv["beta_p"][r] = float(beta_hat[1])
            blp_iv["beta_w"][r] = float(beta_hat[2])
        except Exception:
            blp_iv["fail"][r] = 1

        # 2) BLP - cost IV
        try:
            sigma_hat, beta_hat, _ = estimate_blp_sigma(
                markets, iv_type="nocost", R=cfg.R0
            )
            blp_noiv["sigma"][r] = float(sigma_hat)
            blp_noiv["beta_p"][r] = float(beta_hat[1])
            blp_noiv["beta_w"][r] = float(beta_hat[2])
        except Exception:
            blp_noiv["fail"][r] = 1

        # 3) Shrinkage
        try:
            sigma_s, beta_s, _score, gamma_prob = estimate_shrinkage_sigma(
                markets,
                R=cfg.R0,
                n_iter=study.shrink_n_iter,
                burn=study.shrink_burn,
                v0=study.shrink_v0,
                v1=study.shrink_v1,
            )
            shrink["sigma"][r] = float(sigma_s)
            shrink["beta_p"][r] = float(beta_s[1])
            shrink["beta_w"][r] = float(beta_s[2])
            if gamma_prob is not None:
                shrink_gamma_list.append(np.asarray(gamma_prob))
        except Exception:
            shrink["fail"][r] = 1

        # 4) Lu25 MAP (optional)
        if include_lu25_map and HAS_LU25_MAP:
            try:
                lu_res = run_lu25_map_mc(markets, study, cfg, rep_seed=rep_seed)
                sigma_lu = float(lu_res["sigma_hat"])
                beta_lu = np.asarray(lu_res["beta_hat"], dtype=float)

                lu25["sigma"][r] = sigma_lu
                lu25["beta_p"][r] = float(beta_lu[1])
                lu25["beta_w"][r] = float(beta_lu[2])

                if lu_res.get("gamma_hat") is not None:
                    lu25_gamma_list.append(np.asarray(lu_res["gamma_hat"]))
            except Exception:
                lu25["fail"][r] = 1

        pbar.update(1)

    pbar.close()

    # summarize
    summaries: Dict[str, Dict[str, Dict[str, float]]] = {}

    summaries["BLP+CostIV"] = {
        "sigma": summarize_vec(blp_iv["sigma"], true_params["sigma"]),
        "beta_p": summarize_vec(blp_iv["beta_p"], true_params["beta_p"]),
        "beta_w": summarize_vec(blp_iv["beta_w"], true_params["beta_w"]),
    }

    summaries["BLP-NoCostIV"] = {
        "sigma": summarize_vec(blp_noiv["sigma"], true_params["sigma"]),
        "beta_p": summarize_vec(blp_noiv["beta_p"], true_params["beta_p"]),
        "beta_w": summarize_vec(blp_noiv["beta_w"], true_params["beta_w"]),
    }

    summaries["Shrinkage"] = {
        "sigma": summarize_vec(shrink["sigma"], true_params["sigma"]),
        "beta_p": summarize_vec(shrink["beta_p"], true_params["beta_p"]),
        "beta_w": summarize_vec(shrink["beta_w"], true_params["beta_w"]),
    }

    if include_lu25_map and HAS_LU25_MAP and lu25 is not None:
        summaries["Lu25MAP"] = {
            "sigma": summarize_vec(lu25["sigma"], true_params["sigma"]),
            "beta_p": summarize_vec(lu25["beta_p"], true_params["beta_p"]),
            "beta_w": summarize_vec(lu25["beta_w"], true_params["beta_w"]),
        }

    # print tables
    print_method_table("BLP + Cost IV", summaries["BLP+CostIV"], true_params)
    print_method_table("BLP − Cost IV", summaries["BLP-NoCostIV"], true_params)
    print_method_table("Shrinkage", summaries["Shrinkage"], true_params)
    if include_lu25_map and HAS_LU25_MAP and "Lu25MAP" in summaries:
        print_method_table("Lu25 MAP", summaries["Lu25MAP"], true_params)

    # optional sparsity summary (only if shrinkage gamma is available)
    if len(shrink_gamma_list) > 0 and dgp in ["DGP1", "DGP2"]:
        # In your repo: first sparse_frac*J are true signals (non-zero)
        cutoff = int(cfg.sparse_frac * J)
        G = np.stack(
            [g.reshape(T, J) for g in shrink_gamma_list], axis=0
        )  # [R_eff, T, J]
        gamma_avg = G.mean(axis=(0, 1))  # [J]
        signal = float(gamma_avg[:cutoff].mean()) if cutoff > 0 else np.nan
        noise = float(gamma_avg[cutoff:].mean()) if cutoff < J else np.nan
        print("\nSparsity recovery (Shrinkage; avg gamma over markets+reps):")
        print(f"  cutoff (signal products): {cutoff}/{J}")
        print(f"  avg gamma signal: {signal:.4f}")
        print(f"  avg gamma noise:  {noise:.4f}")

    if (
        include_lu25_map
        and HAS_LU25_MAP
        and len(lu25_gamma_list) > 0
        and dgp in ["DGP1", "DGP2"]
    ):
        cutoff = int(cfg.sparse_frac * J)
        H = np.stack(
            [g.reshape(T, J) for g in lu25_gamma_list], axis=0
        )  # [R_eff, T, J]
        gamma_rate = H.mean(axis=0)  # [T,J]
        signal_rate = float(gamma_rate[:, :cutoff].mean()) if cutoff > 0 else np.nan
        noise_rate = float(gamma_rate[:, cutoff:].mean()) if cutoff < J else np.nan
        print("\nSparsity recovery (Lu25MAP; mean detect rate |d|>tau):")
        print(f"  tau_detect: {study.lu_tau_detect:.3f}")
        print(f"  detect rate signal: {signal_rate:.4f}")
        print(f"  detect rate noise:  {noise_rate:.4f}")
        print(
            f"  specificity (1-noise): {1.0 - noise_rate:.4f}"
            if not np.isnan(noise_rate)
            else ""
        )

    # failure counts
    print("\nFailures (count / R):")
    print(f"  BLP+CostIV   : {int(np.sum(blp_iv['fail']))}/{study.R_mc}")
    print(f"  BLP-NoCostIV : {int(np.sum(blp_noiv['fail']))}/{study.R_mc}")
    print(f"  Shrinkage    : {int(np.sum(shrink['fail']))}/{study.R_mc}")
    if include_lu25_map and HAS_LU25_MAP and lu25 is not None:
        print(f"  Lu25MAP      : {int(np.sum(lu25['fail']))}/{study.R_mc}")

    return summaries


# -----------------------------------------------------------------------------
# Main study runner
# -----------------------------------------------------------------------------


def main():
    results_dir = RESULTS_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Configure study
    study = StudyConfig(
        R_mc=50,
        seed=123,
        shrink_n_iter=200,
        shrink_burn=100,
        shrink_v0=1e-4,
        shrink_v1=1.0,
        lu_steps=1200,
        lu_lr=0.05,
        lu_l1_strength=8.0,
        lu_mu_sd=2.0,
        lu_tau_detect=0.25,
    )

    include_lu25_map = HAS_LU25_MAP  # set False if you only want the 3 benchmarks

    # Initialize SimConfig (paper parameters live here)
    cfg = SimConfig()

    # Log file
    log_path = results_dir / f"replication_study_{timestamp}.txt"
    csv_path = results_dir / f"replication_study_{timestamp}.csv"

    logger = OutputLogger(log_path)
    old_stdout = sys.stdout
    sys.stdout = logger

    try:
        print("#" * 90)
        print("LU & SHIMIZU (2025) — SECTION 4 REPLICATION STUDY (repo-consistent)")
        print("#" * 90)
        print(f"Timestamp: {timestamp}")
        print(f"R_mc: {study.R_mc}")
        print(f"Base seed: {study.seed}")
        print(f"Market size N_t (cfg.Nt): {getattr(cfg, 'Nt', 'NA')}")
        print(f"Consumer draws R0 (cfg.R0): {cfg.R0}")
        print(
            f"Shrinkage: n_iter={study.shrink_n_iter}, burn={study.shrink_burn}, v0={study.shrink_v0}, v1={study.shrink_v1}"
        )
        if include_lu25_map:
            print(
                f"Lu25 MAP: steps={study.lu_steps}, lr={study.lu_lr}, l1_strength={study.lu_l1_strength}, tau={study.lu_tau_detect}"
            )
        else:
            print("Lu25 MAP: disabled")

        # True parameters
        print("\nTrue parameters (from SimConfig):")
        print(f"  sigma*:  {cfg.sigma_star:.4f}")
        print(f"  beta_p*: {cfg.beta_p_star:.4f}")
        print(f"  beta_w*: {cfg.beta_w_star:.4f}")

        # Run grid
        for dgp, T, J in DEFAULT_GRID:
            cell_key = f"{dgp}_T{T}_J{J}"
            summaries = run_cell(
                dgp, T, J, study, cfg, include_lu25_map=include_lu25_map
            )
            save_summary_csv(csv_path, cell_key, summaries)

        print("\n" + "=" * 90)
        print("Study complete.")
        print("Saved outputs:")
        print(f"  Log: {log_path.name}")
        print(f"  CSV: {csv_path.name}")
        print("=" * 90)

    finally:
        sys.stdout = old_stdout
        logger.close()


if __name__ == "__main__":
    main()
