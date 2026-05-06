"""Simulation designs used for Figures 2c, 3, and 4."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


SHOCK_NAMES = ("Uniform", "Beta05", "Beta2", "Bimodal")
DEPENDENCE_NAMES = (
    "Independent",
    "LaggedHeteroskedastic",
    "ThresholdScale",
    "ConditionalMixture",
)


@dataclass(frozen=True)
class SimulationParameters:
    gamma: float = 0.50
    rho: float = 0.70
    threshold_multiplier: float = 2.00
    mixture_multiplier: float = 2.50
    mixture_slope: float = 2.00


def draw_base_shock(rng: np.random.Generator, n_samples: int, shock_name: str) -> Array:
    shock_name = str(shock_name)
    if shock_name == "Uniform":
        return rng.uniform(-1.0, 1.0, size=n_samples)
    if shock_name == "Beta05":
        return 2.0 * rng.beta(0.5, 0.5, size=n_samples) - 1.0
    if shock_name == "Beta2":
        return 2.0 * rng.beta(2.0, 2.0, size=n_samples) - 1.0
    if shock_name == "Bimodal":
        low_component = rng.random(n_samples) < 0.5
        out = np.empty(n_samples, dtype=float)
        out[low_component] = -1.0 + 0.7 * rng.random(int(np.sum(low_component)))
        out[~low_component] = 0.3 + 0.7 * rng.random(int(np.sum(~low_component)))
        return out
    raise ValueError(f"Unknown shock distribution: {shock_name}")


def generate_sparse_lower_triangular_b(
    rng: np.random.Generator,
    n_nodes: int,
    max_parents: int = 8,
    coef_min: float = 0.30,
    coef_max: float = 0.80,
) -> Array:
    """Generate the MATLAB replication's lower triangular B matrix."""

    b = np.zeros((n_nodes, n_nodes), dtype=float)
    for target in range(1, n_nodes):
        n_candidates = target
        n_parent_cap = min(max_parents, n_candidates)
        n_parents = int(rng.integers(1, n_parent_cap + 1))
        parents = rng.choice(n_candidates, size=n_parents, replace=False)
        b[target, parents] = rng.uniform(coef_min, coef_max, size=n_parents)
    return b


def generate_disturbances(
    rng: np.random.Generator,
    n_samples: int,
    n_nodes: int,
    shock_name: str,
    dependence_name: str,
    params: SimulationParameters | None = None,
) -> Array:
    """Generate order-dependent mean-independent disturbances."""

    params = SimulationParameters() if params is None else params
    e = np.zeros((n_samples, n_nodes), dtype=float)
    dependence_name = str(dependence_name)

    if dependence_name == "Independent":
        for j in range(n_nodes):
            e[:, j] = draw_base_shock(rng, n_samples, shock_name)
        return e

    e[:, 0] = draw_base_shock(rng, n_samples, shock_name)

    if dependence_name == "LaggedHeteroskedastic":
        for j in range(1, n_nodes):
            history = np.zeros(n_samples, dtype=float)
            for ell in range(j):
                history += params.rho ** (j - ell - 1) * e[:, ell]
            sd_history = float(np.std(history, ddof=1))
            if sd_history < 1e-8 or not np.isfinite(sd_history):
                sd_history = 1.0
            history = history / sd_history
            scale = np.exp(0.5 * params.gamma * history)
            scale = np.minimum(scale, 5.0)
            e[:, j] = scale * draw_base_shock(rng, n_samples, shock_name)
        return e

    if dependence_name == "ThresholdScale":
        for j in range(1, n_nodes):
            mean_history = np.mean(e[:, :j], axis=1)
            scale = np.ones(n_samples, dtype=float)
            scale[mean_history > 0.0] = params.threshold_multiplier
            e[:, j] = scale * draw_base_shock(rng, n_samples, shock_name)
        return e

    if dependence_name == "ConditionalMixture":
        for j in range(1, n_nodes):
            mean_history = np.mean(e[:, :j], axis=1)
            probability_low = 1.0 / (1.0 + np.exp(-params.mixture_slope * mean_history))
            choose_low = rng.random(n_samples) < probability_low
            low = draw_base_shock(rng, n_samples, shock_name)
            high = draw_base_shock(rng, n_samples, shock_name)
            current = low.copy()
            current[~choose_low] = params.mixture_multiplier * high[~choose_low]
            e[:, j] = current
        return e

    raise ValueError(f"Unknown dependence design: {dependence_name}")


def build_system_from_disturbances(e: Array, b: Array) -> Array:
    """Simulate X = B X + e from lower-triangular B and centered shocks."""

    e = np.asarray(e, dtype=float)
    b = np.asarray(b, dtype=float)
    n_samples, n_nodes = e.shape
    x = np.zeros((n_samples, n_nodes), dtype=float)
    for target in range(n_nodes):
        if target == 0:
            x[:, target] = e[:, target]
        else:
            x[:, target] = e[:, target] + x[:, :target] @ b[target, :target]
    return x - np.mean(x, axis=0, keepdims=True)


def generate_sem_data(
    rng: np.random.Generator,
    n_samples: int,
    n_nodes: int,
    shock_name: str,
    dependence_name: str,
    max_parents: int = 8,
    coef_min: float = 0.30,
    coef_max: float = 0.80,
    params: SimulationParameters | None = None,
) -> tuple[Array, Array, Array]:
    b = generate_sparse_lower_triangular_b(rng, n_nodes, max_parents, coef_min, coef_max)
    e = generate_disturbances(rng, n_samples, n_nodes, shock_name, dependence_name, params)
    x = build_system_from_disturbances(e, b)
    return x, b, e


def sample_martingale_mean_independent_noise_uniform_average(
    rng: np.random.Generator,
    n_samples: int,
    n_dim: int,
) -> Array:
    """Noise used in Figure 2c."""

    u = rng.uniform(-1.0, 1.0, size=(n_samples, n_dim))
    e = np.zeros((n_samples, n_dim), dtype=float)
    e[:, 0] = u[:, 0]
    running_sum = e[:, 0].copy()
    for j in range(1, n_dim):
        e[:, j] = (running_sum / j) * u[:, j]
        running_sum += u[:, j]
    return e


__all__ = [
    "DEPENDENCE_NAMES",
    "SHOCK_NAMES",
    "SimulationParameters",
    "build_system_from_disturbances",
    "draw_base_shock",
    "generate_disturbances",
    "generate_sem_data",
    "generate_sparse_lower_triangular_b",
    "sample_martingale_mean_independent_noise_uniform_average",
]
