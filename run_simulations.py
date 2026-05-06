"""Replicate the synthetic experiments for Figures 3 and 4.

Default settings match the paper/MATLAB replication: 100 Monte Carlo
replications, T=500, p in {2,3,4,5,6}, four shock distributions, and four
dependence designs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FixedLocator
import numpy as np
import pandas as pd

from direct_limiam import DirectLiMIAM
from metrics import (
    ci_binomial_95,
    frobenius_error,
    order_success,
    quartile_interval,
    structural_hamming_distance,
)
from simulation_utils import DEPENDENCE_NAMES, SHOCK_NAMES, generate_sem_data


METHOD_NAMES = ("DirectLiNGAM", "MI-Moment", "MI-Sieve", "MI-Kernel")
METHOD_DISPLAY_NAMES = {
    "DirectLiNGAM": "DirectLiNGAM",
    "MI-Moment": "DirectLiMIAM-Moment",
    "MI-Sieve": "DirectLiMIAM-Sieve",
    "MI-Kernel": "DirectLiMIAM-Kernel",
}
METHOD_COLORS = {
    "DirectLiNGAM": "#1f77b4",
    "MI-Moment": "#ff7f0e",
    "MI-Sieve": "#2ca02c",
    "MI-Kernel": "#7b3294",
}
ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "results" / "simulations"


@dataclass(frozen=True)
class TrialTask:
    n_idx: int
    shock_idx: int
    dependence_idx: int
    rep: int
    n_nodes: int
    n_samples: int
    max_parents: int
    coef_min: float
    coef_max: float
    edge_tol: float
    adjacency_estimator: str


def _method_factories(adjacency_estimator: str):
    return (
        ("DirectLiNGAM", lambda: DirectLiMIAM(score="pwling", adjacency_estimator=adjacency_estimator)),
        ("MI-Moment", lambda: DirectLiMIAM(score="moment", moment_powers=(2, 3), adjacency_estimator=adjacency_estimator)),
        (
            "MI-Sieve",
            lambda: DirectLiMIAM(
                score="sieve_cv",
                k_grid=(4, 5, 6, 8, 10),
                cv_folds=5,
                adjacency_estimator=adjacency_estimator,
            ),
        ),
        (
            "MI-Kernel",
            lambda: DirectLiMIAM(
                score="kernel_cv",
                h_grid=None,
                cv_folds=5,
                adjacency_estimator=adjacency_estimator,
            ),
        ),
    )


def run_trial(task: TrialTask) -> list[dict[str, float | int | str]]:
    seed = 100000 * (task.n_idx + 1) + 1000 * (task.shock_idx + 1) + 100 * (task.dependence_idx + 1) + (task.rep + 1)
    rng = np.random.default_rng(seed)
    shock_name = SHOCK_NAMES[task.shock_idx]
    dependence_name = DEPENDENCE_NAMES[task.dependence_idx]
    x, b_true, _ = generate_sem_data(
        rng,
        n_samples=task.n_samples,
        n_nodes=task.n_nodes,
        shock_name=shock_name,
        dependence_name=dependence_name,
        max_parents=task.max_parents,
        coef_min=task.coef_min,
        coef_max=task.coef_max,
    )
    truth_support = np.abs(b_true) > 0.0

    rows: list[dict[str, float | int | str]] = []
    for method_name, factory in _method_factories(task.adjacency_estimator):
        model = factory()
        start = time.perf_counter()
        fit_error = ""
        try:
            model.fit(x)
            pred_order = list(model.causal_order_)
            b_hat = np.asarray(model.adjacency_matrix_, dtype=float)
            exact_order = order_success(pred_order, task.n_nodes)
            frob = frobenius_error(b_hat, b_true)
            shd = structural_hamming_distance(b_hat, truth_support, task.edge_tol)
        except Exception as exc:  # pragma: no cover - long-run robustness
            pred_order = []
            exact_order = 0.0
            frob = float("nan")
            shd = float("nan")
            fit_error = f"{type(exc).__name__}: {exc}"
        runtime = time.perf_counter() - start

        rows.append(
            {
                "n_nodes": task.n_nodes,
                "n_samples": task.n_samples,
                "shock": shock_name,
                "dependence": dependence_name,
                "rep": task.rep,
                "seed": seed,
                "method": method_name,
                "order_success": exact_order,
                "frobenius_error": frob,
                "shd": shd,
                "runtime_seconds": runtime,
                "causal_order": repr(pred_order),
                "fit_error": fit_error,
            }
        )
    return rows


def build_tasks(args: argparse.Namespace) -> list[TrialTask]:
    return [
        TrialTask(
            n_idx=n_idx,
            shock_idx=shock_idx,
            dependence_idx=dependence_idx,
            rep=rep,
            n_nodes=n_nodes,
            n_samples=args.samples,
            max_parents=args.max_parents,
            coef_min=args.coef_min,
            coef_max=args.coef_max,
            edge_tol=args.edge_tol,
            adjacency_estimator=args.adjacency_estimator,
        )
        for n_idx, n_nodes in enumerate(args.nodes)
        for shock_idx in range(len(SHOCK_NAMES))
        for dependence_idx in range(len(DEPENDENCE_NAMES))
        for rep in range(args.mc)
    ]


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = raw.groupby(["n_nodes", "shock", "dependence", "method"], sort=False)
    for keys, group in grouped:
        order_lo, order_mean, order_hi = ci_binomial_95(group["order_success"].to_numpy())
        frob_lo, frob_mean, frob_hi = quartile_interval(group["frobenius_error"].to_numpy())
        shd_lo, shd_mean, shd_hi = quartile_interval(group["shd"].to_numpy())
        rows.append(
            {
                "n_nodes": keys[0],
                "shock": keys[1],
                "dependence": keys[2],
                "method": keys[3],
                "order_success_mean": order_mean,
                "order_success_lo": order_lo,
                "order_success_hi": order_hi,
                "frobenius_error_mean": frob_mean,
                "frobenius_error_q25": frob_lo,
                "frobenius_error_q75": frob_hi,
                "shd_mean": shd_mean,
                "shd_q25": shd_lo,
                "shd_q75": shd_hi,
                "runtime_seconds_mean": float(np.mean(group["runtime_seconds"])),
                "n_replications": int(len(group)),
                "n_failures": int(np.sum(group["fit_error"].astype(str) != "")),
            }
        )
    return pd.DataFrame(rows)


def _metric_columns(metric: str) -> tuple[str, str | None, str | None, str]:
    if metric == "order_success":
        return "order_success_mean", "order_success_lo", "order_success_hi", "Ordering success"
    if metric == "shd":
        return "shd_mean", "shd_q25", "shd_q75", "Mean SHD"
    if metric == "frobenius_error":
        return "frobenius_error_mean", "frobenius_error_q25", "frobenius_error_q75", "Mean Frobenius error"
    raise ValueError(metric)


def _padded_limits(values: np.ndarray, *, fraction: float = 0.08, minimum_pad: float = 0.1) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -minimum_pad, minimum_pad
    lower = float(np.min(values))
    upper = float(np.max(values))
    span = upper - lower
    pad = max(span * fraction, minimum_pad)
    return lower - pad, upper + pad


def _metric_ylim(summary: pd.DataFrame, metric: str, mean_col: str, lo_col: str | None, hi_col: str | None) -> tuple[float, float]:
    if metric == "order_success":
        return -0.1, 1.1
    values = [summary[mean_col].to_numpy(dtype=float)]
    if lo_col is not None and lo_col in summary:
        values.append(summary[lo_col].to_numpy(dtype=float))
    if hi_col is not None and hi_col in summary:
        values.append(summary[hi_col].to_numpy(dtype=float))
    return _padded_limits(np.concatenate(values), fraction=0.08, minimum_pad=0.1)


def plot_metric(summary: pd.DataFrame, output_dir: Path, metric: str, file_stem: str) -> None:
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "mathtext.fontset": "cm",
            "axes.labelsize": 20,
            "axes.titlesize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 20,
            "axes.unicode_minus": False,
        }
    )
    mean_col, lo_col, hi_col, y_label = _metric_columns(metric)
    fig, axes = plt.subplots(
        len(SHOCK_NAMES),
        len(DEPENDENCE_NAMES),
        figsize=(16, 10),
        sharex=True,
        sharey=False,
        squeeze=False,
    )
    legend_handles = {}
    node_ticks = np.array(sorted(summary["n_nodes"].unique()), dtype=int)
    x_min, x_max = _padded_limits(node_ticks, fraction=0.04, minimum_pad=0.25)

    for row, shock_name in enumerate(SHOCK_NAMES):
        for col, dependence_name in enumerate(DEPENDENCE_NAMES):
            ax = axes[row][col]
            panel = summary[(summary["shock"] == shock_name) & (summary["dependence"] == dependence_name)]
            y_min, y_max = _metric_ylim(panel, metric, mean_col, lo_col, hi_col)
            for method in METHOD_NAMES:
                method_panel = panel[panel["method"] == method].sort_values("n_nodes")
                if method_panel.empty:
                    continue
                x = method_panel["n_nodes"].to_numpy()
                y = method_panel[mean_col].to_numpy()
                (line,) = ax.plot(
                    x,
                    y,
                    color=METHOD_COLORS[method],
                    linewidth=1.8,
                    label=METHOD_DISPLAY_NAMES[method],
                )
                legend_handles.setdefault(method, line)
                if lo_col is not None and hi_col is not None:
                    lo = method_panel[lo_col].to_numpy()
                    hi = method_panel[hi_col].to_numpy()
                    yerr = np.vstack(
                        [
                            np.maximum(y - lo, 0.0),
                            np.maximum(hi - y, 0.0),
                        ]
                    )
                    ax.errorbar(
                        x,
                        y,
                        yerr=yerr,
                        fmt="none",
                        ecolor=METHOD_COLORS[method],
                        elinewidth=0.8,
                        capsize=2.5,
                        capthick=0.8,
                        alpha=0.85,
                    )

            ax.grid(True, alpha=0.3)
            ax.set_xlim(x_min, x_max)
            ax.xaxis.set_major_locator(FixedLocator(node_ticks))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
            ax.set_ylim(y_min, y_max)
            if row == 0:
                ax.set_title(dependence_name)
            if col == 0:
                ax.set_ylabel(f"{shock_name}\n{y_label}")
            if row == len(SHOCK_NAMES) - 1:
                ax.set_xlabel(r"Number of variables $p$")

    handles = [legend_handles[method] for method in METHOD_NAMES if method in legend_handles]
    labels = [METHOD_DISPLAY_NAMES[method] for method in METHOD_NAMES if method in legend_handles]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.998),
        ncol=len(labels),
        frameon=True,
        fancybox=False,
        edgecolor="black",
        borderaxespad=0.2,
        columnspacing=1.6,
        handlelength=2.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.925))
    fig.savefig(output_dir / f"{file_stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{file_stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_latex_table(summary: pd.DataFrame, output_dir: Path) -> None:
    path = output_dir / "directlimiam_compare_nodes_table.tex"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("% Auto-generated by run_simulations.py\n")
        handle.write("\\begin{table}[!htbp]\n\\centering\n\\scriptsize\n")
        handle.write("\\caption{Comparison of DirectLiNGAM and mean-independence variants.}\n")
        handle.write("\\label{tab:directlimiam_compare_nodes}\n")
        for shock_name in SHOCK_NAMES:
            handle.write(f"\\paragraph{{Shock distribution: {shock_name}}}\n")
            alignment = "cc" + "ccc" * len(DEPENDENCE_NAMES)
            handle.write(f"\\begin{{tabular}}{{{alignment}}}\n\\hline\\hline\n")
            handle.write("$p$ & Method ")
            for dependence_name in DEPENDENCE_NAMES:
                handle.write(f"& \\multicolumn{{3}}{{c}}{{{dependence_name}}} ")
            handle.write("\\\\\n & ")
            for _ in DEPENDENCE_NAMES:
                handle.write("& Order & Frob. & SHD ")
            handle.write("\\\\\n\\hline\n")
            for n_nodes in sorted(summary["n_nodes"].unique()):
                for method_idx, method in enumerate(METHOD_NAMES):
                    handle.write(f"{n_nodes if method_idx == 0 else ''} & {method} ")
                    for dependence_name in DEPENDENCE_NAMES:
                        row = summary[
                            (summary["shock"] == shock_name)
                            & (summary["dependence"] == dependence_name)
                            & (summary["n_nodes"] == n_nodes)
                            & (summary["method"] == method)
                        ]
                        if row.empty:
                            handle.write("& -- & -- & -- ")
                        else:
                            r = row.iloc[0]
                            handle.write(
                                f"& {r['order_success_mean']:.3f} "
                                f"& {r['frobenius_error_mean']:.3f} "
                                f"& {r['shd_mean']:.3f} "
                            )
                    handle.write("\\\\\n")
                handle.write("\\hline\n")
            handle.write("\\hline\\hline\n\\end{tabular}\n\\vspace{0.4cm}\n")
        handle.write("\\end{table}\n")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only_summary is not None:
        summary = pd.read_csv(args.plot_only_summary)
        plot_metric(summary, output_dir, "order_success", "figure3_order_success")
        plot_metric(summary, output_dir, "shd", "figure4_shd")
        plot_metric(summary, output_dir, "frobenius_error", "frobenius_error")
        write_latex_table(summary, output_dir)
        print(f"Rebuilt plots from {args.plot_only_summary} into {output_dir}")
        return

    tasks = build_tasks(args)
    rows: list[dict[str, float | int | str]] = []

    print(f"Running {len(tasks)} synthetic tasks; each task fits {len(METHOD_NAMES)} methods.")
    if args.jobs == 1:
        for idx, task in enumerate(tasks, start=1):
            rows.extend(run_trial(task))
            if idx % max(1, len(tasks) // 20) == 0:
                print(f"  completed {idx}/{len(tasks)} tasks")
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
            future_map = {executor.submit(run_trial, task): task for task in tasks}
            for idx, future in enumerate(concurrent.futures.as_completed(future_map), start=1):
                rows.extend(future.result())
                if idx % max(1, len(tasks) // 20) == 0:
                    print(f"  completed {idx}/{len(tasks)} tasks")

    raw_path = output_dir / "raw_results.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    summary.to_csv(output_dir / "summary_results.csv", index=False)
    plot_metric(summary, output_dir, "order_success", "figure3_order_success")
    plot_metric(summary, output_dir, "shd", "figure4_shd")
    plot_metric(summary, output_dir, "frobenius_error", "frobenius_error")
    write_latex_table(summary, output_dir)
    print(f"Saved results to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--mc", type=int, default=100, help="Monte Carlo replications per cell.")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--nodes", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--max-parents", type=int, default=8)
    parser.add_argument("--coef-min", type=float, default=0.30)
    parser.add_argument("--coef-max", type=float, default=0.80)
    parser.add_argument("--edge-tol", type=float, default=1e-11)
    parser.add_argument(
        "--adjacency-estimator",
        choices=["adaptive_lasso", "full_ols"],
        default="full_ols",
        help=(
            "Final B estimator. full_ols matches the collaborator MATLAB "
            "Replication code; adaptive_lasso matches the LiNGAM Python code path."
        ),
    )
    parser.add_argument(
        "--plot-only-summary",
        default=None,
        help="Read an existing simulation summary CSV and only rebuild plots/tables.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Override to a tiny smoke-test run: mc=1, samples=80, nodes=2 3.",
    )
    args = parser.parse_args()
    if args.quick:
        args.mc = 1
        args.samples = 80
        args.nodes = [2, 3]
    return args


if __name__ == "__main__":
    run(parse_args())
