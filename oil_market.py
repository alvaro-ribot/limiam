"""Replicate the oil-market SVAR application and Figure 5."""

from __future__ import annotations

import argparse
from pathlib import Path

from graphviz import Digraph
import numpy as np
import pandas as pd
from PIL import Image

from direct_limiam import (
    DirectLiMIAM,
    estimate_adjacency_full_ols_given_order,
    estimate_adjacency_given_order,
)


VARS_KEEP = ("SURPRISE", "OILPROD", "OILSTOCKS", "IP", "POIL", "CPI")
ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "VARdata.xlsx"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "oil_market"


def load_oil_data(path: str | Path, variables: tuple[str, ...] = VARS_KEEP) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0)
    numeric = df.select_dtypes(include=[np.number])
    missing = [name for name in variables if name not in numeric.columns]
    if missing:
        raise ValueError(f"Missing expected numeric columns: {missing}")
    return numeric.loc[:, list(variables)].dropna(axis=0, how="any").reset_index(drop=True)


def var_residuals(x: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    n_obs, n_vars = x.shape
    if n_obs <= lags:
        raise ValueError("Need more observations than VAR lags.")
    y = x[lags:, :]
    z = np.ones((n_obs - lags, 1 + n_vars * lags), dtype=float)
    for lag in range(1, lags + 1):
        cols = 1 + (lag - 1) * n_vars + np.arange(n_vars)
        z[:, cols] = x[lags - lag : n_obs - lag, :]
    beta, *_ = np.linalg.lstsq(z, y, rcond=None)
    intercept = beta[0, :]
    lag_coef = beta[1:, :]
    residual = y - z @ beta
    return residual, lag_coef, intercept, z, y


def standardize_residuals(u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(u, axis=0, keepdims=True)
    scale = np.std(u - mean, axis=0, ddof=1, keepdims=True)
    scale = np.where((scale > 1e-12) & np.isfinite(scale), scale, 1.0)
    return (u - mean) / scale, mean.reshape(-1), scale.reshape(-1)


def shocks_from_b(u: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Recover eps = (I - B) U for rows of contemporaneous residuals U."""

    return np.asarray(u, dtype=float) - np.asarray(u, dtype=float) @ np.asarray(b, dtype=float).T


def extract_edges(b: np.ndarray, edge_tol: float) -> list[tuple[int, int, float]]:
    edges: list[tuple[int, int, float]] = []
    n_vars = b.shape[0]
    for target in range(n_vars):
        for source in range(n_vars):
            if target != source and abs(b[target, source]) > edge_tol:
                edges.append((source, target, float(b[target, source])))
    return edges


def format_edge_label(coef: float) -> str:
    return f"{coef:.3f}"


def set_dag_graph_style(graph: Digraph) -> None:
    graph.attr(
        rankdir="TB",
        splines="true",
        bgcolor="white",
        nodesep="0.45",
        ranksep="0.85",
        margin="0.04",
        pad="0.08",
    )
    graph.attr("node", shape="ellipse", fontsize="30", margin="0.08,0.04")
    graph.attr("edge", fontsize="26")


def add_lingam_dot_panel(
    graph: Digraph,
    panel_name: str,
    title: str,
    adjacency: np.ndarray,
    variable_names: list[str],
    edge_tol: float,
) -> None:
    """Add one DAG panel using the LiNGAM ``make_dot`` convention.

    This mirrors the repository's ``utils.make_dot`` orientation:
    ``B[target, source]`` is drawn as ``source -> target`` with the edge
    coefficient as the label.
    """

    with graph.subgraph(name=f"cluster_{panel_name}") as panel:
        panel.attr(label=title, fontsize="34", labelloc="t", color="white")
        panel.attr("node", shape="ellipse", fontsize="30", margin="0.08,0.04")
        panel.attr("edge", fontsize="26")

        for name in variable_names:
            panel.node(f"{panel_name}_{name}", label=name)

        for source, target, coef in extract_edges(adjacency, edge_tol):
            panel.edge(
                f"{panel_name}_{variable_names[source]}",
                f"{panel_name}_{variable_names[target]}",
                label=format_edge_label(coef),
            )


def combine_dag_panel_images(output_dir: Path, panel_stems: tuple[str, str]) -> None:
    images = [
        Image.open(output_dir / f"{stem}.png").convert("RGBA")
        for stem in panel_stems
    ]
    gap = 90
    width = sum(image.width for image in images) + gap
    height = max(image.height for image in images)
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    x_offset = 0
    for image in images:
        y_offset = (height - image.height) // 2
        canvas.alpha_composite(image, (x_offset, y_offset))
        x_offset += image.width + gap

    canvas.save(output_dir / "figure5_oil_market_dags.png")
    canvas.convert("RGB").save(
        output_dir / "figure5_oil_market_dags.pdf",
        resolution=300.0,
    )


def plot_dags(
    b_direct: np.ndarray,
    order_direct: list[int],
    b_limiam: np.ndarray,
    order_limiam: list[int],
    variable_names: list[str],
    output_dir: Path,
    edge_tol_direct: float,
    edge_tol_limiam: float,
) -> None:
    del order_direct, order_limiam  # Graphviz computes the DAG layout.

    for stem, title, adjacency, edge_tol in (
        ("figure5_directlingam_dag", "DirectLiNGAM DAG", b_direct, edge_tol_direct),
        ("figure5_directlimiam_dag", "DirectLiMIAM DAG", b_limiam, edge_tol_limiam),
    ):
        panel_graph = Digraph(stem, engine="dot")
        set_dag_graph_style(panel_graph)
        add_lingam_dot_panel(panel_graph, "dag", title, adjacency, variable_names, edge_tol)
        for file_format in ("pdf", "png"):
            panel_graph.format = file_format
            panel_graph.render(filename=stem, directory=str(output_dir), cleanup=True)

    combine_dag_panel_images(
        output_dir,
        ("figure5_directlingam_dag", "figure5_directlimiam_dag"),
    )


def estimate_fixed_order_adjacency(
    u: np.ndarray, order: list[int], adjacency_estimator: str
) -> np.ndarray:
    estimator = str(adjacency_estimator).lower()
    if estimator in {"full_ols", "ols", "replication_ols"}:
        return estimate_adjacency_full_ols_given_order(u, order)
    if estimator in {"adaptive_lasso", "lasso", "lingam"}:
        return estimate_adjacency_given_order(u, order)
    raise ValueError(f"Unknown adjacency_estimator: {adjacency_estimator}")


def bootstrap_fixed_order_se(
    u: np.ndarray,
    order: list[int],
    n_bootstrap: int,
    seed: int,
    adjacency_estimator: str,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_samples, n_vars = u.shape
    draws = np.full((n_bootstrap, n_vars, n_vars), np.nan, dtype=float)
    for b_idx in range(n_bootstrap):
        indices = rng.integers(0, n_samples, size=n_samples)
        draws[b_idx] = estimate_fixed_order_adjacency(
            u[indices, :], order, adjacency_estimator
        )
    return np.nanstd(draws, axis=0, ddof=1)


def write_bootstrap_outputs(
    b_hat: np.ndarray,
    se: np.ndarray,
    variable_names: list[str],
    order: list[int],
    output_dir: Path,
    edge_tol: float,
) -> None:
    rows = []
    for target in range(len(variable_names)):
        for source in range(len(variable_names)):
            if target == source:
                continue
            se_value = float(se[target, source])
            b_value = float(b_hat[target, source])
            rows.append(
                {
                    "source": variable_names[source],
                    "target": variable_names[target],
                    "Bhat": b_value,
                    "SE_fixed_order": se_value,
                    "Tstat_fixed_order": b_value / se_value if se_value > 0 else np.nan,
                    "included_in_dag": abs(b_value) > edge_tol,
                }
            )
    pd.DataFrame(rows).sort_values("Bhat", ascending=False).to_csv(
        output_dir / "kernel_b_matrix_with_fixed_order_bootstrap_ses.csv",
        index=False,
    )

    ordered_names = [variable_names[idx] for idx in order]
    with (output_dir / "kernel_b_table_causal_order.tex").open("w", encoding="utf-8") as handle:
        handle.write("\\begin{table}[!htbp]\n\\centering\n\\small\n")
        handle.write("\\caption{DirectLiMIAM estimated contemporaneous $B$ matrix.}\n")
        handle.write("\\label{tab:kernelB}\n")
        handle.write("\\begin{tabular}{l" + "c" * (len(order) - 1) + "}\n\\toprule\n")
        handle.write(" & " + " & ".join(ordered_names[:-1]) + " \\\\\n\\midrule\n")
        for row_pos, target in enumerate(order[1:], start=1):
            cells = []
            for source in order[:-1]:
                if order.index(source) >= row_pos:
                    cells.append("--")
                    continue
                b_value = b_hat[target, source]
                se_value = se[target, source]
                cells.append(f"${b_value:.3f}_{{({se_value:.3f})}}$")
            handle.write(f"{variable_names[target]} & " + " & ".join(cells) + " \\\\\n")
        handle.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def _median_bandwidth(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if len(x) > 300:
        x = x[:300]
    sq_norm = np.sum(x * x, axis=1)
    dist2 = sq_norm[:, None] + sq_norm[None, :] - 2.0 * x @ x.T
    positive = dist2[dist2 > 1e-12]
    if positive.size == 0:
        return 1.0
    return max(float(np.sqrt(0.5 * np.median(positive))), 1e-6)


def _centered_rbf_gram(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    width = _median_bandwidth(x)
    sq_norm = np.sum(x * x, axis=1)
    dist2 = sq_norm[:, None] + sq_norm[None, :] - 2.0 * x @ x.T
    gram = np.exp(-dist2 / (2.0 * width**2))
    row_mean = np.mean(gram, axis=1, keepdims=True)
    col_mean = np.mean(gram, axis=0, keepdims=True)
    return gram - row_mean - col_mean + float(np.mean(gram))


def dhsic_permutation_test(x: np.ndarray, n_permutations: int, seed: int) -> tuple[float, float]:
    """Simple permutation dHSIC-style joint independence diagnostic."""

    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    grams = [_centered_rbf_gram(x[:, j]) for j in range(x.shape[1])]

    def statistic(permutations: list[np.ndarray] | None = None) -> float:
        product = grams[0].copy()
        for j in range(1, len(grams)):
            gram = grams[j] if permutations is None else grams[j][np.ix_(permutations[j - 1], permutations[j - 1])]
            product *= gram
        return float(np.mean(product))

    observed = statistic()
    exceed = 0
    for _ in range(n_permutations):
        perms = [rng.permutation(x.shape[0]) for _ in range(x.shape[1] - 1)]
        exceed += int(statistic(perms) >= observed)
    p_value = (exceed + 1.0) / (n_permutations + 1.0)
    return observed, p_value


def ordered_mean_independence_test(
    eps: np.ndarray,
    order: list[int],
    n_permutations: int,
    seed: int,
) -> tuple[float, float, pd.DataFrame]:
    """Kernel test for E[eps_i | previous ordered epsilons] = 0."""

    rng = np.random.default_rng(seed)
    eps_ordered = eps[:, order]
    eps_ordered = eps_ordered - np.mean(eps_ordered, axis=0, keepdims=True)
    terms = []
    observed = 0.0
    for idx in range(1, eps_ordered.shape[1]):
        y = eps_ordered[:, idx]
        gram = _centered_rbf_gram(eps_ordered[:, :idx])
        value = float(y @ gram @ y)
        observed += value
        terms.append((idx, order[idx], value, gram, y))

    exceed = 0
    for _ in range(n_permutations):
        stat = 0.0
        for _, _, _, gram, y in terms:
            yp = y[rng.permutation(len(y))]
            stat += float(yp @ gram @ yp)
        exceed += int(stat >= observed)
    p_value = (exceed + 1.0) / (n_permutations + 1.0)
    detail = pd.DataFrame(
        [
            {
                "ordered_position": idx + 1,
                "variable_index": variable,
                "statistic": value,
            }
            for idx, variable, value, _, _ in terms
        ]
    )
    return observed, p_value, detail


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_oil_data(args.data)
    names = list(df.columns)
    x = df.to_numpy(dtype=float)
    u_hat, lag_coef, intercept, z, y = var_residuals(x, args.lags)
    u_model = u_hat
    residual_mean = np.mean(u_hat, axis=0)
    residual_scale = np.std(u_hat - residual_mean, axis=0, ddof=1)
    if args.standardize_residuals:
        u_model, residual_mean, residual_scale = standardize_residuals(u_hat)

    print(f"Loaded {len(df)} monthly observations; VAR({args.lags}) residual sample has {len(u_hat)} rows.")
    print(f"Using {args.adjacency_estimator} adjacency estimation.")
    if args.standardize_residuals:
        print("Standardizing VAR residuals before DirectLiNGAM/DirectLiMIAM.")
    else:
        print("Using raw VAR residuals before DirectLiNGAM/DirectLiMIAM.")
    print("Running DirectLiNGAM...")
    direct = DirectLiMIAM(
        score="pwling",
        adjacency_estimator=args.adjacency_estimator,
    ).fit(u_model)
    print("Running DirectLiMIAM kernel-CV...")
    limiam = DirectLiMIAM(
        score="kernel_cv",
        cv_folds=args.cv_folds,
        adjacency_estimator=args.adjacency_estimator,
    ).fit(u_model)

    b_direct = np.asarray(direct.adjacency_matrix_, dtype=float)
    b_limiam = np.asarray(limiam.adjacency_matrix_, dtype=float)
    order_direct = list(direct.causal_order_)
    order_limiam = list(limiam.causal_order_)

    print("Computing bootstrap standard errors conditional on the DirectLiMIAM order...")
    se_limiam = bootstrap_fixed_order_se(
        u_model,
        order_limiam,
        args.bootstraps,
        args.seed,
        args.adjacency_estimator,
    )

    eps_direct = shocks_from_b(u_model, b_direct)
    eps_limiam = shocks_from_b(u_model, b_limiam)
    print("Running kernel permutation diagnostics...")
    dhsic_stat, dhsic_p = dhsic_permutation_test(eps_direct, args.permutations, args.seed + 1)
    mean_ind_stat, mean_ind_p, mean_ind_detail = ordered_mean_independence_test(
        eps_limiam,
        order_limiam,
        args.permutations,
        args.seed + 2,
    )

    pd.DataFrame(u_hat, columns=names).to_csv(output_dir / "var_residuals_uhat.csv", index=False)
    pd.DataFrame(u_model, columns=names).to_csv(output_dir / "var_residuals_used_for_lingam.csv", index=False)
    pd.DataFrame(
        {
            "variable": names,
            "raw_series_sd": df.std(axis=0, ddof=1).to_numpy(dtype=float),
            "var24_residual_mean": residual_mean,
            "var24_residual_sd": residual_scale,
            "standardized_before_lingam": args.standardize_residuals,
        }
    ).to_csv(output_dir / "oil_variable_scales.csv", index=False)
    pd.DataFrame(b_direct, index=names, columns=names).to_csv(output_dir / "bhat_directlingam.csv")
    pd.DataFrame(b_limiam, index=names, columns=names).to_csv(output_dir / "bhat_directlimiam_kernel.csv")
    pd.Series([names[idx] for idx in order_direct], name="directlingam_order").to_csv(
        output_dir / "directlingam_causal_order.csv",
        index=False,
    )
    pd.Series([names[idx] for idx in order_limiam], name="directlimiam_order").to_csv(
        output_dir / "directlimiam_causal_order.csv",
        index=False,
    )

    write_bootstrap_outputs(b_limiam, se_limiam, names, order_limiam, output_dir, args.edge_tol_limiam)
    mean_ind_detail["variable"] = mean_ind_detail["variable_index"].map(lambda idx: names[int(idx)])
    mean_ind_detail.to_csv(output_dir / "ordered_mean_independence_terms.csv", index=False)
    pd.DataFrame(
        [
            {
                "diagnostic": "directlingam_joint_independence_dhsic_permutation",
                "statistic": dhsic_stat,
                "p_value": dhsic_p,
                "permutations": args.permutations,
            },
            {
                "diagnostic": "directlimiam_ordered_mean_independence_kernel_permutation",
                "statistic": mean_ind_stat,
                "p_value": mean_ind_p,
                "permutations": args.permutations,
            },
        ]
    ).to_csv(output_dir / "specification_tests.csv", index=False)

    plot_dags(
        b_direct,
        order_direct,
        b_limiam,
        order_limiam,
        names,
        output_dir,
        args.edge_tol_direct,
        args.edge_tol_limiam,
    )

    print(f"DirectLiNGAM order: {[names[idx] for idx in order_direct]}")
    print(f"DirectLiMIAM order: {[names[idx] for idx in order_limiam]}")
    print(f"DirectLiNGAM dHSIC-style p-value: {dhsic_p:.4f}")
    print(f"DirectLiMIAM ordered mean-independence p-value: {mean_ind_p:.4f}")
    print(f"Saved oil-market outputs to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--lags", type=int, default=24)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--bootstraps", type=int, default=199)
    parser.add_argument("--permutations", type=int, default=499)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--adjacency-estimator",
        choices=["adaptive_lasso", "full_ols"],
        default="full_ols",
        help=(
            "Estimate B using full OLS on all predecessors in the learned "
            "order, or the LiNGAM adaptive LASSO + OLS refit."
        ),
    )
    parser.add_argument("--edge-tol-direct", type=float, default=0.001)
    parser.add_argument("--edge-tol-limiam", type=float, default=0.001)
    parser.add_argument(
        "--standardize-residuals",
        dest="standardize_residuals",
        action="store_true",
        default=True,
        help="Standardize VAR residuals before DirectLiNGAM/DirectLiMIAM. This is the default.",
    )
    parser.add_argument(
        "--raw-residuals",
        dest="standardize_residuals",
        action="store_false",
        help="Use raw VAR residuals before DirectLiNGAM/DirectLiMIAM.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use tiny bootstrap/permutation counts for a fast smoke test.",
    )
    args = parser.parse_args()
    if args.quick:
        args.bootstraps = 9
        args.permutations = 19
    return args


if __name__ == "__main__":
    run(parse_args())
