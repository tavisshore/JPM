from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np


def init_storage(R: int) -> Dict[str, np.ndarray]:
    return {
        "sigma": np.full(R, np.nan),
        "beta_p": np.full(R, np.nan),
        "beta_w": np.full(R, np.nan),
        "fail": np.zeros(R, dtype=int),
    }


def summarize_vec(x: np.ndarray, true_val: float) -> Dict[str, float]:
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {"mean": np.nan, "bias": np.nan, "sd": np.nan, "rmse": np.nan, "n": 0}
    mean = float(np.mean(x))
    bias = float(mean - true_val)
    sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    rmse = float(np.sqrt(bias * bias + sd * sd))
    return {"mean": mean, "bias": bias, "sd": sd, "rmse": rmse, "n": int(x.size)}


def print_method_table(
    title: str, summary: Dict[str, Dict[str, float]], true_params: Dict[str, float]
):
    print(f"\n{'-' * 90}")
    print(title)
    print(f"{'-' * 90}")
    print(
        f"{'Param':<8} {'True':>10} {'Mean':>10} {'Bias':>10} {'SD':>10} {'RMSE':>10} {'n':>6}"
    )
    print(f"{'-' * 90}")
    mapping = [("sigma", "σ"), ("beta_p", "β_p"), ("beta_w", "β_w")]
    for k, sym in mapping:
        s = summary[k]
        tv = true_params[k]
        print(
            f"{sym:<8} {tv:>10.4f} {s['mean']:>10.4f} {s['bias']:>10.4f} {s['sd']:>10.4f} {s['rmse']:>10.4f} {s['n']:>6d}"
        )


def save_summary_csv(
    path: Path, cell_key: str, summaries: Dict[str, Dict[str, Dict[str, float]]]
):
    # Append mode: one big CSV across the whole grid
    header = "cell,method,parameter,mean,bias,sd,rmse,n\n"
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(header)
        for method, summ in summaries.items():
            for param in ["sigma", "beta_p", "beta_w"]:
                s = summ[param]
                f.write(
                    f"{cell_key},{method},{param},{s['mean']:.6f},{s['bias']:.6f},{s['sd']:.6f},{s['rmse']:.6f},{s['n']}\n"
                )


def print_progress_bar(
    iteration: int, total: int, prefix: str = "", suffix: str = "", length: int = 36
):
    if total <= 0:
        return
    pct = 100.0 * (iteration / float(total))
    filled = int(length * iteration // total)
    bar = "█" * filled + "░" * (length - filled)
    print(f"\r{prefix} |{bar}| {pct:5.1f}% {suffix}", end="", flush=True)
    if iteration == total:
        print()


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
