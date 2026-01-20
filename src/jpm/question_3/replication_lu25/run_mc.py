"""run_mc_final.py

Monte Carlo driver.

Goal: align the simulation/estimators to the paper's Table 1 settings, while keeping
outputs clean and reproducible.

Default run:
  • one Table-1-style cell (DGP, T, J)
  • estimators:
      - BLP + cost IV
      - BLP − cost IV
      - Shrinkage (no IV)
      - Lu & Shimizu (2025)-style MAP (no IV, likelihood-based, sparse shocks)

Outputs:
  - results/mc_<DGP>_T<T>_J<J>_R<R>_<timestamp>.txt   (console log)
  - results/mc_<DGP>_T<T>_J<J>_R<R>_<timestamp>.csv   (summary table)

Table-1 alignment knobs:
  • N_t (market size used by Lu25 MAP pseudo-likelihood): MCConfig.N_t
    - we also inject N_t into each market dict (market["N"]) so all code paths see it.

  • tau0^2 (shrinkage slab/spike baseline variance): MCConfig.tau0_sq
    - passed as v0 into estimate_shrinkage_sigma.

Reproducibility:
  • replication r uses seed = base_seed + r

NOTE on SimConfig.default_market_size:
  • we do NOT rely on SimConfig having this attribute (some versions are dataclasses/frozen).
  • instead, we set market["N"] = N_t after simulation.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from jpm.config import SimConfig
from jpm.question_3.replication_lu25.estimators.blp import estimate_blp_sigma
from jpm.question_3.replication_lu25.estimators.lu25_map import (
    Lu25MapConfig,
    estimate_lu25_map,
)
from jpm.question_3.replication_lu25.estimators.shrinkage import (
    estimate_shrinkage_sigma,
)
from jpm.question_3.replication_lu25.simulation.simulate import simulate_dataset


class OutputLogger:
    """Print to console AND write to a file."""

    def __init__(self, filename: Path):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        try:
            self.log.close()
        except Exception:
            pass


def ensure_results_dir() -> Path:
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    return results_dir


def print_progress_bar(
    iteration: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    length: int = 40,
    fill: str = "█",
):
    if total <= 0:
        return
    percent = 100 * (iteration / float(total))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + "░" * (length - filled_length)
    print(f"\r{prefix} |{bar}| {percent:5.1f}% {suffix}", end="", flush=True)
    if iteration == total:
        print()


@dataclass
class MCConfig:
    # cell
    DGP: str = "DGP1"
    T: int = 25
    J: int = 15
    R_mc: int = 50
    seed: int = 123

    # Table-1 alignment
    N_t: int = 1000
    tau0_sq: float = 1e-4  # paper's tau0^2 (used as v0 in shrinkage)

    # consumer draws (for estimators)
    R0: Optional[int] = None  # if None, uses cfg.R0

    # Shrinkage MCMC settings
    shrink_n_iter: int = 200
    shrink_burn: int = 100
    shrink_v1: float = 1.0

    # Lu25 MAP settings
    lu_steps: int = 1200
    lu_lr: float = 0.05
    lu_l1_strength: float = 8.0
    lu_mu_sd: float = 2.0
    lu_tau_detect: float = 0.25


def init_storage(R: int) -> Dict[str, np.ndarray]:
    return {
        "sigma": np.full(R, np.nan),
        "beta_p": np.full(R, np.nan),
        "beta_w": np.full(R, np.nan),
        "fail": np.zeros(R, dtype=int),
    }


def init_diagnostics(R: int) -> Dict[str, np.ndarray]:
    return {
        "outside_share": np.full(R, np.nan),
        "price_xi_corr": np.full(R, np.nan),
        "seed": np.full(R, np.nan),
    }


def inject_market_size(markets: list, N_t: int) -> None:
    """Ensure each market dict has key 'N' so Lu25 MAP uses the correct pseudo-market-size."""
    for m in markets:
        m["N"] = int(N_t)


def summarize_mc(arr: np.ndarray, true_val: float) -> Dict[str, float]:
    x = arr[~np.isnan(arr)]
    if x.size == 0:
        return {"mean": np.nan, "bias": np.nan, "sd": np.nan, "rmse": np.nan}
    mean = float(np.mean(x))
    bias = float(mean - true_val)
    sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    rmse = float(np.sqrt(bias * bias + sd * sd))
    return {"mean": mean, "bias": bias, "sd": sd, "rmse": rmse}


def print_param_table(
    header: str, summ: Dict[str, Dict[str, float]], true_params: Dict[str, float]
):
    print(f"\n{'-' * 90}")
    print(header)
    print(f"{'-' * 90}")
    print(
        f"{'Param':<12} {'True':>10} {'Mean':>12} {'Bias':>12} {'SD':>12} {'RMSE':>12}"
    )
    print(f"{'-' * 90}")

    rows = [
        ("σ", "sigma", true_params["sigma"]),
        ("β_p", "beta_p", true_params["beta_p"]),
        ("β_w", "beta_w", true_params["beta_w"]),
    ]
    for sym, k, tv in rows:
        s = summ[k]
        print(
            f"{sym:<12} {tv:>10.4f} {s['mean']:>12.4f} {s['bias']:>12.4f} {s['sd']:>12.4f} {s['rmse']:>12.4f}"
        )


def run_blp_cell(
    mc: MCConfig, cfg: SimConfig, iv_type: str
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    res = init_storage(mc.R_mc)
    diag = init_diagnostics(mc.R_mc)

    R0 = int(mc.R0) if mc.R0 is not None else int(cfg.R0)

    for r in range(mc.R_mc):
        rep_seed = mc.seed + r
        np.random.seed(rep_seed)

        markets = simulate_dataset(mc.DGP, T=mc.T, J=mc.J, cfg=cfg)
        inject_market_size(markets, mc.N_t)

        # diagnostics from market 0
        diag["outside_share"][r] = float(1.0 - np.sum(markets[0]["s"]))
        diag["price_xi_corr"][r] = float(
            np.corrcoef(markets[0]["p"], markets[0]["xi"])[0, 1]
        )
        diag["seed"][r] = rep_seed

        try:
            sigma_hat, beta_hat, _ = estimate_blp_sigma(markets, iv_type=iv_type, R=R0)
            res["sigma"][r] = float(sigma_hat)
            res["beta_p"][r] = float(beta_hat[1])
            res["beta_w"][r] = float(beta_hat[2])
        except Exception:
            res["fail"][r] = 1

        print_progress_bar(
            r + 1,
            mc.R_mc,
            prefix=f"  {'BLP(' + iv_type + ')':12} T={mc.T:<3} J={mc.J:<3}",
            suffix=f"({r + 1}/{mc.R_mc})",
        )

    return res, diag


def run_shrinkage_cell(
    mc: MCConfig, cfg: SimConfig
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Optional[np.ndarray]]:
    res = init_storage(mc.R_mc)
    diag = init_diagnostics(mc.R_mc)

    R0 = int(mc.R0) if mc.R0 is not None else int(cfg.R0)

    gamma_list = []

    for r in range(mc.R_mc):
        rep_seed = mc.seed + r
        np.random.seed(rep_seed)

        markets = simulate_dataset(mc.DGP, T=mc.T, J=mc.J, cfg=cfg)
        inject_market_size(markets, mc.N_t)

        diag["outside_share"][r] = float(1.0 - np.sum(markets[0]["s"]))
        diag["price_xi_corr"][r] = float(
            np.corrcoef(markets[0]["p"], markets[0]["xi"])[0, 1]
        )
        diag["seed"][r] = rep_seed

        try:
            sigma_s, beta_s, _, gamma_prob = estimate_shrinkage_sigma(
                markets,
                R=R0,
                n_iter=mc.shrink_n_iter,
                burn=mc.shrink_burn,
                v0=mc.tau0_sq,
                v1=mc.shrink_v1,
            )
            res["sigma"][r] = float(sigma_s)
            res["beta_p"][r] = float(beta_s[1])
            res["beta_w"][r] = float(beta_s[2])
            gamma_list.append(np.asarray(gamma_prob))
        except Exception:
            res["fail"][r] = 1

        print_progress_bar(
            r + 1,
            mc.R_mc,
            prefix=f"  {'Shrinkage':12} T={mc.T:<3} J={mc.J:<3}",
            suffix=f"({r + 1}/{mc.R_mc})",
        )

    gamma_stack = None
    if len(gamma_list) > 0:
        gamma_stack = np.stack(gamma_list, axis=0)  # [R_eff, T*J]

    return res, diag, gamma_stack


def run_lu25_map_cell(
    mc: MCConfig, cfg: SimConfig
) -> Tuple[
    Dict[str, np.ndarray], Dict[str, np.ndarray], Optional[Dict[str, np.ndarray]]
]:
    if estimate_lu25_map is None or Lu25MapConfig is None:
        raise ImportError("estimators/lu25_map.py not importable in this environment")

    res = init_storage(mc.R_mc)
    diag = init_diagnostics(mc.R_mc)

    sp = {
        "signal_detect_rate": np.full(mc.R_mc, np.nan),
        "noise_detect_rate": np.full(mc.R_mc, np.nan),
        "sensitivity": np.full(mc.R_mc, np.nan),
        "specificity": np.full(mc.R_mc, np.nan),
        "detected": np.full(mc.R_mc, np.nan),
    }

    cutoff = int(cfg.sparse_frac * mc.J)
    R0 = int(mc.R0) if mc.R0 is not None else int(cfg.R0)

    for r in range(mc.R_mc):
        rep_seed = mc.seed + r
        np.random.seed(rep_seed)

        markets = simulate_dataset(mc.DGP, T=mc.T, J=mc.J, cfg=cfg)
        inject_market_size(markets, mc.N_t)

        diag["outside_share"][r] = float(1.0 - np.sum(markets[0]["s"]))
        diag["price_xi_corr"][r] = float(
            np.corrcoef(markets[0]["p"], markets[0]["xi"])[0, 1]
        )
        diag["seed"][r] = rep_seed

        try:
            lu_cfg = Lu25MapConfig(
                R=R0,
                steps=mc.lu_steps,
                lr=mc.lu_lr,
                l1_strength=mc.lu_l1_strength,
                mu_sd=mc.lu_mu_sd,
                default_market_size=mc.N_t,
                seed=rep_seed,
                tau_detect=mc.lu_tau_detect,
            )
            lu_res = estimate_lu25_map(markets, cfg=lu_cfg)

            sigma_lu = float(lu_res["sigma_hat"])
            beta_lu = np.asarray(lu_res["beta_hat"], dtype=float)

            res["sigma"][r] = sigma_lu
            res["beta_p"][r] = float(beta_lu[1])
            res["beta_w"][r] = float(beta_lu[2])

            if "gamma_hat" in lu_res and lu_res["gamma_hat"] is not None:
                gamma_hat = np.asarray(lu_res["gamma_hat"], dtype=int).reshape(
                    mc.T, mc.J
                )
                signal_detect = (
                    gamma_hat[:, :cutoff] if cutoff > 0 else gamma_hat[:, :0]
                )
                noise_detect = (
                    gamma_hat[:, cutoff:] if cutoff < mc.J else gamma_hat[:, :0]
                )

                sens = float(signal_detect.mean()) if signal_detect.size else np.nan
                noise_rate = float(noise_detect.mean()) if noise_detect.size else np.nan
                spec = float(1.0 - noise_rate) if noise_detect.size else np.nan

                sp["signal_detect_rate"][r] = sens
                sp["noise_detect_rate"][r] = noise_rate
                sp["sensitivity"][r] = sens
                sp["specificity"][r] = spec
                sp["detected"][r] = float(gamma_hat.sum())

        except Exception:
            res["fail"][r] = 1

        print_progress_bar(
            r + 1,
            mc.R_mc,
            prefix=f"  {'Lu25 MAP':12} T={mc.T:<3} J={mc.J:<3}",
            suffix=f"({r + 1}/{mc.R_mc})",
        )

    return res, diag, sp


def save_results_csv(filename: Path, summaries: Dict[str, Dict[str, Dict[str, float]]]):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("method,parameter,mean,bias,sd,rmse\n")
        for method_name, sum_method in summaries.items():
            for param in ["sigma", "beta_p", "beta_w"]:
                s = sum_method[param]
                f.write(
                    f"{method_name},{param},{s['mean']:.6f},{s['bias']:.6f},{s['sd']:.6f},{s['rmse']:.6f}\n"
                )


def print_comparison(summaries: Dict[str, Dict[str, Dict[str, float]]]):
    print(f"\n{'=' * 90}")
    print(f"{'COMPARISON SUMMARY - ALL METHODS':^90}")
    print(f"{'=' * 90}")
    print(f"{'Param':<10} {'Method':<18} {'Bias':>12} {'SD':>12} {'RMSE':>12}")
    print(f"{'-' * 90}")

    for param, sym in [("sigma", "σ"), ("beta_p", "β_p"), ("beta_w", "β_w")]:
        first = True
        for method_name, sum_method in summaries.items():
            s = sum_method[param]
            print(
                f"{(sym if first else ''):<10} {method_name:<18} {s['bias']:>12.4f} {s['sd']:>12.4f} {s['rmse']:>12.4f}"
            )
            first = False
        print(f"{'-' * 90}")

    print(f"{'=' * 90}")


def run_table1_cell(mc: MCConfig):
    cfg = SimConfig()
    results_dir = ensure_results_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_file = (
        results_dir / f"mc_{mc.DGP}_T{mc.T}_J{mc.J}_R{mc.R_mc}_{timestamp}.txt"
    )
    csv_file = results_dir / f"mc_{mc.DGP}_T{mc.T}_J{mc.J}_R{mc.R_mc}_{timestamp}.csv"

    logger = OutputLogger(output_file)
    old_stdout = sys.stdout
    sys.stdout = logger

    try:
        print("\n" + "#" * 90)
        print(f"{'MONTE CARLO SIMULATION (Table 1 alignment)':^90}")
        print("#" * 90)
        print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"R_mc: {mc.R_mc}")
        print(f"Base seed: {mc.seed}")
        print(f"Default market size N_t: {mc.N_t}")
        print(f"Consumer draws R0: {mc.R0 if mc.R0 is not None else cfg.R0}")
        print(
            f"Shrinkage: n_iter={mc.shrink_n_iter}, burn={mc.shrink_burn}, tau0^2(v0)={mc.tau0_sq}, v1={mc.shrink_v1}"
        )

        true_params = {
            "sigma": float(cfg.sigma_star),
            "beta_p": float(cfg.beta_p_star),
            "beta_w": float(cfg.beta_w_star),
        }

        print("\nTrue Parameters")
        print(f"  σ*   = {true_params['sigma']:.4f}")
        print(f"  β_p* = {true_params['beta_p']:.4f}")
        print(f"  β_w* = {true_params['beta_w']:.4f}")

        print("\n" + "=" * 90)
        print(
            f"Cell: {mc.DGP}, T={mc.T}, J={mc.J}, N_t={mc.N_t}, R0={mc.R0 if mc.R0 is not None else cfg.R0}"
        )
        print("=" * 90)

        # 1) BLP + cost IV
        blp_cost, _ = run_blp_cell(mc, cfg, iv_type="cost")

        # 2) BLP - cost IV
        blp_nocost, _ = run_blp_cell(mc, cfg, iv_type="nocost")

        # 3) Shrinkage
        shrink, _diag_sh, _gamma_stack = run_shrinkage_cell(mc, cfg)

        # 4) Lu25 MAP
        lu25, _diag_lu, lu25_sp = run_lu25_map_cell(mc, cfg)

        summaries: Dict[str, Dict[str, Dict[str, float]]] = {}
        for name, res in [
            ("BLP+IV", blp_cost),
            ("BLP-IV", blp_nocost),
            ("Shrinkage", shrink),
            ("Lu25MAP", lu25),
        ]:
            summaries[name] = {
                "sigma": summarize_mc(res["sigma"], true_params["sigma"]),
                "beta_p": summarize_mc(res["beta_p"], true_params["beta_p"]),
                "beta_w": summarize_mc(res["beta_w"], true_params["beta_w"]),
            }

        print_param_table("BLP + cost IV", summaries["BLP+IV"], true_params)
        print_param_table("BLP − cost IV", summaries["BLP-IV"], true_params)
        print_param_table("Shrinkage", summaries["Shrinkage"], true_params)
        print_param_table("Lu25 MAP", summaries["Lu25MAP"], true_params)

        if lu25_sp is not None:
            sens = float(np.nanmean(lu25_sp["sensitivity"]))
            spec = float(np.nanmean(lu25_sp["specificity"]))
            noise = float(np.nanmean(lu25_sp["noise_detect_rate"]))
            det = float(np.nanmean(lu25_sp["detected"]))

            print(f"\n{'-' * 90}")
            print("Lu25 MAP sparsity (avg across replications)")
            print(f"  tau_detect: {mc.lu_tau_detect:.4f}")
            print(f"  mean sensitivity: {sens:.4f}")
            print(f"  mean noise detect rate: {noise:.4f}")
            print(f"  mean specificity: {spec:.4f}")
            print(f"  mean # detected (T×J): {det:.1f}")
            print(f"{'-' * 90}")

        print_comparison(summaries)
        save_results_csv(csv_file, summaries)

        print("\nFailures (count / R):")
        for name, res in [
            ("BLP+IV", blp_cost),
            ("BLP-IV", blp_nocost),
            ("Shrinkage", shrink),
            ("Lu25MAP", lu25),
        ]:
            print(f"  {name:<10}: {int(np.sum(res['fail']))}/{mc.R_mc}")

        print("\n" + "=" * 90)
        print("Results saved to:")
        print(f"  {output_file.name}")
        print(f"  {csv_file.name}")
        print("=" * 90 + "\n")

    finally:
        sys.stdout = old_stdout
        logger.close()


def run_full_table1_grid(base: MCConfig, grid_TJ, dgps) -> None:
    """Optional: run all Table-1 cells (this can take many hours)."""
    cfg = SimConfig()
    results_dir = ensure_results_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_file = results_dir / f"mc_full_grid_R{base.R_mc}_{timestamp}.txt"
    logger = OutputLogger(output_file)
    old_stdout = sys.stdout
    sys.stdout = logger

    try:
        print("\n" + "#" * 90)
        print(f"{'FULL TABLE 1 GRID':^90}")
        print("#" * 90)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"R_mc: {base.R_mc}")
        print(f"Base seed: {base.seed}")
        print(
            f"N_t: {base.N_t}, tau0^2(v0): {base.tau0_sq}, R0: {base.R0 if base.R0 is not None else cfg.R0}\n"
        )

        cell_idx = 0
        total = len(dgps) * len(grid_TJ)

        for dgp in dgps:
            for T, J in grid_TJ:
                cell_idx += 1
                print("\n" + "=" * 90)
                print(f"Cell {cell_idx}/{total}: {dgp}, T={T}, J={J}")
                print("=" * 90)

                mc = MCConfig(**{**base.__dict__, "DGP": dgp, "T": T, "J": J})
                run_table1_cell(mc)

        print("\n" + "=" * 90)
        print("Full-grid run complete.")
        print(f"Log saved: {output_file.name}")
        print("=" * 90)

    finally:
        sys.stdout = old_stdout
        logger.close()


if __name__ == "__main__":
    # ======= EDIT HERE =======
    R_mc = 50  # can be 10 for a quick reference
    run_full_grid = True  # set to true if want full grid with all cells

    # Replace these with the (T,J) cells in Table 1.
    grid_TJ = [(25, 15)]
    dgps = ["DGP1"]

    # Base configuration (applies to all cells)
    base = MCConfig(
        DGP="DGP1",
        T=25,
        J=15,
        R_mc=R_mc,
        seed=123,
        N_t=1000,
        tau0_sq=1e-4,
        # if you want to override cfg.R0, set R0=200 (or whatever Table 1 uses)
        R0=None,
        shrink_n_iter=200,
        shrink_burn=100,
        shrink_v1=1.0,
        lu_steps=1200,
        lu_lr=0.05,
        lu_l1_strength=8.0,
        lu_tau_detect=0.25,
        lu_mu_sd=2.0,
    )

    if run_full_grid:
        print("\n" + "=" * 90)
        print("WARNING: Full grid can take many hours.")
        print("=" * 90)
        run_full_table1_grid(base, grid_TJ=grid_TJ, dgps=dgps)
    else:
        run_table1_cell(base)
