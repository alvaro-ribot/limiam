"""Metrics and small reporting helpers for the replication scripts."""

from __future__ import annotations

import numpy as np


Array = np.ndarray


def structural_hamming_distance(b_hat: Array, truth_support: Array, edge_tol: float = 1e-11) -> int:
    """Pairwise SHD used in the MATLAB replication.

    For each unordered node pair this counts one error if the estimated
    directed edge state differs from the true directed edge state.
    """

    a_hat = np.abs(np.asarray(b_hat, dtype=float)) > edge_tol
    a_true = np.asarray(truth_support).astype(bool)
    n_nodes = a_true.shape[0]
    shd = 0
    for i in range(n_nodes - 1):
        for j in range(i + 1, n_nodes):
            true_ij, true_ji = bool(a_true[i, j]), bool(a_true[j, i])
            hat_ij, hat_ji = bool(a_hat[i, j]), bool(a_hat[j, i])
            if not true_ij and not true_ji:
                shd += int(hat_ij or hat_ji)
            elif true_ij and not true_ji:
                shd += int(not (hat_ij and not hat_ji))
            elif not true_ij and true_ji:
                shd += int(not ((not hat_ij) and hat_ji))
    return int(shd)


def order_success(order: list[int], n_nodes: int) -> float:
    return float(list(order) == list(range(n_nodes)))


def frobenius_error(b_hat: Array, b_true: Array) -> float:
    return float(np.linalg.norm(np.asarray(b_hat) - np.asarray(b_true), ord="fro"))


def ci_binomial_95(values: Array) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    p_hat = float(np.mean(values)) if values.size else float("nan")
    se = np.sqrt(max(p_hat * (1.0 - p_hat) / max(len(values), 1), 0.0))
    return max(0.0, p_hat - 1.96 * se), p_hat, min(1.0, p_hat + 1.96 * se)


def quartile_interval(values: Array) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.percentile(values, 25)),
        float(np.mean(values)),
        float(np.percentile(values, 75)),
    )


__all__ = [
    "ci_binomial_95",
    "frobenius_error",
    "order_success",
    "quartile_interval",
    "structural_hamming_distance",
]
