"""Generate Figure 2c: order-dependent mean-independent disturbances."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np

from simulation_utils import sample_martingale_mean_independent_noise_uniform_average


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "results" / "figure2c"


def run(output_dir: str, n_samples: int, seed: int) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    e = sample_martingale_mean_independent_noise_uniform_average(rng, n_samples=n_samples, n_dim=3)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter(e[:, 1], e[:, 2], s=30, alpha=0.5, edgecolors="none")
    ax.set_xlabel(r"$\varepsilon_1$", fontsize=24)
    ax.set_ylabel(r"$\varepsilon_2$", fontsize=24)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.tick_params(labelsize=13)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal", adjustable="box")
    fig.savefig(output_path / "figure2c_scatter_e1_e2.pdf", bbox_inches="tight", pad_inches=0.15)
    fig.savefig(output_path / "figure2c_scatter_e1_e2.png", dpi=250, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.output_dir, args.samples, args.seed)
