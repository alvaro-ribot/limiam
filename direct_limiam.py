"""DirectLiMIAM estimator and DirectLiNGAM-style score.

The orientation convention throughout this folder is

    X_i = sum_j B[i, j] X_j + eps_i,

so ``B[target, source]`` is the directed effect ``source -> target``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi, sqrt
from typing import Iterable

import numpy as np
from sklearn.linear_model import LassoLarsIC, LinearRegression
from sklearn.preprocessing import SplineTransformer
from sklearn.preprocessing import StandardScaler


Array = np.ndarray


@dataclass
class FitResult:
    """Convenience return type for functional use."""

    causal_order: list[int]
    adjacency_matrix: Array
    scores: list[float]
    selected_params: list[float | int | None]


def _as_array(x: Array) -> Array:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("X must be a 2D array with rows as observations.")
    if x.shape[0] < 3:
        raise ValueError("X must contain at least three observations.")
    if not np.all(np.isfinite(x)):
        raise ValueError("X contains non-finite values.")
    return x


def center_columns(x: Array) -> Array:
    return np.asarray(x, dtype=float) - np.mean(x, axis=0, keepdims=True)


def safe_std(x: Array, ddof: int = 1) -> float:
    value = float(np.std(x, ddof=ddof)) if len(x) > ddof else float(np.std(x))
    return value if np.isfinite(value) and value > 1e-12 else 1.0


def standardize_vector(x: Array) -> Array:
    x = np.asarray(x, dtype=float).reshape(-1)
    return (x - np.mean(x)) / safe_std(x)


def standardize_columns(x: Array) -> Array:
    x = np.asarray(x, dtype=float)
    centered = center_columns(x)
    scale = np.std(centered, axis=0, ddof=1, keepdims=True)
    scale = np.where((scale > 1e-12) & np.isfinite(scale), scale, 1.0)
    return centered / scale


def residual(y: Array, x: Array) -> Array:
    """Residual from regressing y on x with no explicit intercept.

    Inputs are centered in the covariance formula.
    """

    y = np.asarray(y, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)
    var_x = float(np.mean((x - np.mean(x)) ** 2))
    if var_x < 1e-12 or not np.isfinite(var_x):
        return y.copy()
    cov_yx = float(np.mean((y - np.mean(y)) * (x - np.mean(x))))
    return y - (cov_yx / var_x) * x


def predict_adaptive_lasso(x: Array, predictors: Iterable[int], target: int, gamma: float = 1.0) -> Array:
    """Adaptive-Lasso parent selection followed by OLS refit.

    Standardizes X, builds adaptive weights from an initial OLS fit,
    selects parents with LassoLarsIC(criterion="bic"), and refits
    ordinary least squares on selected parents in the original scale.
    """

    x = _as_array(x)
    predictors = list(predictors)
    if len(predictors) == 0:
        return np.zeros(0, dtype=float)

    scaler = StandardScaler()
    x_std = scaler.fit_transform(x)

    initial_lr = LinearRegression()
    initial_lr.fit(x_std[:, predictors], x_std[:, target])
    weight = np.power(np.abs(initial_lr.coef_), gamma)

    selector = LassoLarsIC(criterion="bic")
    selector.fit(x_std[:, predictors] * weight, x_std[:, target])
    selected = np.abs(selector.coef_ * weight) > 0.0

    coef = np.zeros(len(predictors), dtype=float)
    if np.sum(selected) > 0:
        pred = np.asarray(predictors)
        refit_lr = LinearRegression()
        refit_lr.fit(x[:, pred[selected]], x[:, target])
        coef[selected] = refit_lr.coef_
    return coef


def estimate_adjacency_ols_given_order(x: Array, order: Iterable[int]) -> Array:
    """Estimate B by OLS on all predecessors in ``order``."""

    x = center_columns(_as_array(x))
    order = list(order)
    n_features = x.shape[1]
    if sorted(order) != list(range(n_features)):
        raise ValueError("order must be a permutation of feature indices.")

    b_hat = np.zeros((n_features, n_features), dtype=float)
    for k, target in enumerate(order):
        if k == 0:
            continue
        parents = order[:k]
        coef, *_ = np.linalg.lstsq(x[:, parents], x[:, target], rcond=None)
        b_hat[target, parents] = coef
    return b_hat


def estimate_adjacency_given_order(x: Array, order: Iterable[int], gamma: float = 1.0) -> Array:
    """Estimate B using the LiNGAM adaptive-Lasso and OLS-refit approach."""

    x = center_columns(_as_array(x))
    order = list(order)
    n_features = x.shape[1]
    if sorted(order) != list(range(n_features)):
        raise ValueError("order must be a permutation of feature indices.")

    b_hat = np.zeros((n_features, n_features), dtype=float)
    for k, target in enumerate(order):
        predictors = order[:k]
        if len(predictors) == 0:
            continue
        b_hat[target, predictors] = predict_adaptive_lasso(x, predictors, target, gamma=gamma)
    return b_hat


def fit_order_and_adjacency(
    x: Array,
    order: Iterable[int],
    adjacency_estimator: str = "ols",
) -> FitResult:
    order = list(order)
    estimator = str(adjacency_estimator).lower()
    if estimator == "ols":
        adjacency = estimate_adjacency_ols_given_order(x, order)
    elif estimator == "adaptive_lasso":
        adjacency = estimate_adjacency_given_order(x, order)
    else:
        raise ValueError(f"Unknown adjacency_estimator: {adjacency_estimator}. Choose 'ols' or 'adaptive_lasso'.")
    return FitResult(order, adjacency, [], [])


def make_folds(n_samples: int, n_folds: int) -> Array:
    """Deterministic fold assignment for cross-validation."""

    n_folds = int(max(2, min(n_folds, n_samples)))
    fold_id = np.arange(n_samples) % n_folds
    permutation = np.r_[np.arange(0, n_samples, 2), np.arange(1, n_samples, 2)]
    return fold_id[permutation]


def _mse(y: Array, y_hat: Array) -> float:
    return float(np.mean((np.asarray(y) - np.asarray(y_hat)) ** 2))


def _cross_fitted_constant_mse(y: Array, fold_id: Array) -> tuple[Array, float]:
    pred = np.empty_like(y, dtype=float)
    for fold in np.unique(fold_id):
        test = fold_id == fold
        train = ~test
        pred[test] = float(np.mean(y[train])) if np.any(train) else float(np.mean(y))
    return pred, _mse(y, pred)


def _local_linear_predict(y_train: Array, x_train: Array, x_eval: Array, bandwidth: float) -> Array:
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    x_train = np.asarray(x_train, dtype=float).reshape(-1)
    x_eval = np.asarray(x_eval, dtype=float).reshape(-1)
    bandwidth = max(float(bandwidth), 1e-8)
    train_mean = float(np.mean(y_train))
    pred = np.empty(len(x_eval), dtype=float)

    # Closed-form local-linear intercept. Chunking keeps memory modest when
    # users run larger empirical examples.
    chunk_size = 512
    for start in range(0, len(x_eval), chunk_size):
        stop = min(start + chunk_size, len(x_eval))
        x0 = x_eval[start:stop, None]
        dx = x_train[None, :] - x0
        weights = np.exp(-0.5 * (dx / bandwidth) ** 2) / sqrt(2.0 * pi)

        s0 = np.sum(weights, axis=1)
        s1 = np.sum(weights * dx, axis=1)
        s2 = np.sum(weights * dx**2, axis=1)
        t0 = weights @ y_train
        t1 = (weights * dx) @ y_train

        denom = s0 * s2 - s1**2
        weighted_mean = np.divide(t0, s0, out=np.full_like(t0, train_mean), where=s0 > 1e-12)
        local_linear = np.divide(
            s2 * t0 - s1 * t1,
            denom,
            out=weighted_mean.copy(),
            where=np.abs(denom) > 1e-12,
        )
        pred[start:stop] = local_linear

    return pred


def _bandwidth_grid(x: Array) -> Array:
    x = np.asarray(x, dtype=float).reshape(-1)
    n = len(x)
    sd = safe_std(x)
    q75, q25 = np.percentile(x, [75, 25])
    iqr = float(q75 - q25)
    robust_sd = min(sd, iqr / 1.349) if iqr > 1e-12 else sd
    h_rot = 1.06 * robust_sd * n ** (-1.0 / 5.0)
    h_min = max(0.20 * h_rot, 0.05 * robust_sd * n ** (-1.0 / 5.0), 1e-3)
    h_max = max(3.00 * h_rot, 2.0 * h_min)
    if not np.isfinite(h_min) or h_min <= 0:
        h_min = 1e-3
    if not np.isfinite(h_max) or h_max <= h_min:
        h_max = 10.0 * h_min
    return np.exp(np.linspace(np.log(h_min), np.log(h_max), 10))


def _kernel_cross_fitted_mse(y: Array, x: Array, bandwidth: float, fold_id: Array) -> float:
    pred = np.empty_like(y, dtype=float)
    for fold in np.unique(fold_id):
        test = fold_id == fold
        train = ~test
        if not np.any(test):
            continue
        pred[test] = _local_linear_predict(y[train], x[train], x[test], bandwidth)
    return _mse(y, pred)


def _spline_predict(
    y_train: Array,
    x_train: Array,
    x_eval: Array,
    n_basis: int,
    degree: int,
    ridge: float,
) -> Array:
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    x_train = np.asarray(x_train, dtype=float).reshape(-1, 1)
    x_eval = np.asarray(x_eval, dtype=float).reshape(-1, 1)

    unique_values = np.unique(x_train[:, 0])
    if len(unique_values) <= 2:
        return np.full(len(x_eval), float(np.mean(y_train)))

    degree = int(max(1, min(degree, len(unique_values) - 1)))
    n_basis = int(max(degree + 1, n_basis))
    n_knots = int(max(2, n_basis - degree + 1))
    n_knots = min(n_knots, len(unique_values))

    try:
        transformer = SplineTransformer(
            n_knots=n_knots,
            degree=degree,
            knots="quantile",
            extrapolation="constant",
            include_bias=True,
        )
        train_basis = transformer.fit_transform(x_train)
        eval_basis = transformer.transform(x_eval)
        gram = train_basis.T @ train_basis
        rhs = train_basis.T @ y_train
        coef = np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), rhs)
        return eval_basis @ coef
    except (ValueError, np.linalg.LinAlgError):
        return np.full(len(x_eval), float(np.mean(y_train)))


def _spline_cross_fitted_mse(
    y: Array,
    x: Array,
    n_basis: int,
    fold_id: Array,
    degree: int,
    ridge: float,
) -> float:
    pred = np.empty_like(y, dtype=float)
    for fold in np.unique(fold_id):
        test = fold_id == fold
        train = ~test
        if not np.any(test):
            continue
        pred[test] = _spline_predict(y[train], x[train], x[test], n_basis, degree, ridge)
    return _mse(y, pred)


class DirectLiMIAM:
    """DirectLiMIAM estimator with interchangeable source scores.

    Parameters
    ----------
    score:
        ``"moment"``, ``"sieve_cv"``, ``"kernel_cv"``, or ``"pwling"``.
        The first three compute mean-independence scores. ``"pwling"`` uses
        the DirectLiNGAM pairwise-likelihood/entropy approach. The CV variants
        choose a smoothing parameter per candidate via K-fold CV, then score by
        out-of-fold improvement over a constant predictor. Smaller scores
        indicate more exogenous variables.
    adjacency_estimator:
        ``"ols"`` (default) for OLS estimation on all predecessors in the
        learned order, or ``"adaptive_lasso"`` for the adaptive-Lasso
        parent-selection plus OLS-refit approach.
    """

    def __init__(
        self,
        score: str = "kernel_cv",
        moment_powers: tuple[int, ...] = (2, 3),
        k_grid: tuple[int, ...] = (4, 5, 6, 8, 10),
        h_grid: tuple[float, ...] | None = None,
        cv_folds: int = 5,
        spline_degree: int = 3,
        ridge: float = 1e-6,
        adjacency_estimator: str = "ols",
    ):
        self.score = score
        self.moment_powers = tuple(moment_powers)
        self.k_grid = tuple(k_grid)
        self.h_grid = None if h_grid is None else tuple(float(h) for h in h_grid)
        self.cv_folds = int(cv_folds)
        self.spline_degree = int(spline_degree)
        self.ridge = float(ridge)
        self.adjacency_estimator = adjacency_estimator
        self.causal_order_: list[int] | None = None
        self.adjacency_matrix_: Array | None = None
        self.scores_: list[float] = []
        self.selected_params_: list[float | int | None] = []

    def fit(self, x: Array) -> "DirectLiMIAM":
        x = _as_array(x)
        x_centered = center_columns(x)
        x_work = standardize_columns(x_centered)
        active = list(range(x.shape[1]))
        order: list[int] = []
        scores: list[float] = []
        params: list[float | int | None] = []

        for _ in range(x.shape[1] - 1):
            candidate_scores, candidate_params = self._candidate_scores(x_work)
            j_star = int(np.argmin(candidate_scores))
            order.append(active[j_star])
            scores.append(float(candidate_scores[j_star]))
            params.append(candidate_params[j_star])

            selected = x_work[:, j_star]
            keep = [j for j in range(x_work.shape[1]) if j != j_star]
            residualized = np.column_stack([residual(x_work[:, j], selected) for j in keep])
            x_work = standardize_columns(residualized)
            active = [active[j] for j in keep]

        order.append(active[0])
        self.causal_order_ = order
        estimator = str(self.adjacency_estimator).lower()
        if estimator == "ols":
            self.adjacency_matrix_ = estimate_adjacency_ols_given_order(x_centered, order)
        elif estimator == "adaptive_lasso":
            self.adjacency_matrix_ = estimate_adjacency_given_order(x_centered, order)
        else:
            raise ValueError(f"Unknown adjacency_estimator: {self.adjacency_estimator}. Choose 'ols' or 'adaptive_lasso'.")
        self.scores_ = scores
        self.selected_params_ = params
        return self

    def as_result(self) -> FitResult:
        if self.causal_order_ is None or self.adjacency_matrix_ is None:
            raise RuntimeError("The model has not been fit yet.")
        return FitResult(
            causal_order=list(self.causal_order_),
            adjacency_matrix=np.asarray(self.adjacency_matrix_, dtype=float),
            scores=list(self.scores_),
            selected_params=list(self.selected_params_),
        )

    def _candidate_scores(self, x_work: Array) -> tuple[Array, list[float | int | None]]:
        score_name = self.score.lower()
        if score_name == "moment":
            return self._moment_candidate_scores(x_work)
        if score_name == "sieve_cv":
            return self._sieve_candidate_scores(x_work)
        if score_name == "kernel_cv":
            return self._kernel_candidate_scores(x_work)
        if score_name == "pwling":
            return self._pwling_candidate_scores(x_work)
        raise ValueError(f"Unknown score: {self.score}. Choose 'moment', 'sieve_cv', 'kernel_cv', or 'pwling'.")

    def _standardized_residuals_for_candidate(self, x_work: Array, candidate: int) -> tuple[Array, list[Array]]:
        x_candidate = standardize_vector(x_work[:, candidate])
        residuals = []
        for i in range(x_work.shape[1]):
            if i == candidate:
                continue
            x_i = standardize_vector(x_work[:, i])
            residuals.append(residual(x_i, x_candidate))
        return x_candidate, residuals

    def _moment_candidate_scores(self, x_work: Array) -> tuple[Array, list[None]]:
        scores = np.zeros(x_work.shape[1], dtype=float)
        for candidate in range(x_work.shape[1]):
            x_candidate, residuals = self._standardized_residuals_for_candidate(x_work, candidate)
            score = 0.0
            for r_i in residuals:
                for power in self.moment_powers:
                    moment = float(np.mean(r_i * (x_candidate ** int(power))))
                    score += moment**2
            scores[candidate] = score
        return scores, [None] * x_work.shape[1]

    def _kernel_candidate_scores(self, x_work: Array) -> tuple[Array, list[float]]:
        scores = np.zeros(x_work.shape[1], dtype=float)
        params: list[float] = []
        fold_id = make_folds(x_work.shape[0], self.cv_folds)

        for candidate in range(x_work.shape[1]):
            x_candidate, residuals = self._standardized_residuals_for_candidate(x_work, candidate)
            grid = np.asarray(self.h_grid if self.h_grid is not None else _bandwidth_grid(x_candidate), dtype=float)

            cv_losses = []
            for bandwidth in grid:
                cv_losses.append(
                    sum(_kernel_cross_fitted_mse(r_i, x_candidate, bandwidth, fold_id) for r_i in residuals)
                )
            best_h = float(grid[int(np.argmin(cv_losses))])

            score = 0.0
            for r_i in residuals:
                _, mse_const = _cross_fitted_constant_mse(r_i, fold_id)
                mse_kernel = _kernel_cross_fitted_mse(r_i, x_candidate, best_h, fold_id)
                score += max(0.0, 1.0 - mse_kernel / mse_const) if mse_const > 1e-12 else 0.0
            scores[candidate] = score
            params.append(best_h)

        return scores, params

    def _sieve_candidate_scores(self, x_work: Array) -> tuple[Array, list[int]]:
        scores = np.zeros(x_work.shape[1], dtype=float)
        params: list[int] = []
        fold_id = make_folds(x_work.shape[0], self.cv_folds)

        for candidate in range(x_work.shape[1]):
            x_candidate, residuals = self._standardized_residuals_for_candidate(x_work, candidate)
            grid = np.asarray(self.k_grid, dtype=int)

            cv_losses = []
            for n_basis in grid:
                cv_losses.append(
                    sum(
                        _spline_cross_fitted_mse(
                            r_i,
                            x_candidate,
                            int(n_basis),
                            fold_id,
                            self.spline_degree,
                            self.ridge,
                        )
                        for r_i in residuals
                    )
                )
            best_k = int(grid[int(np.argmin(cv_losses))])

            score = 0.0
            for r_i in residuals:
                _, mse_const = _cross_fitted_constant_mse(r_i, fold_id)
                mse_sieve = _spline_cross_fitted_mse(
                    r_i,
                    x_candidate,
                    best_k,
                    fold_id,
                    self.spline_degree,
                    self.ridge,
                )
                score += max(0.0, 1.0 - mse_sieve / mse_const) if mse_const > 1e-12 else 0.0
            scores[candidate] = score
            params.append(best_k)

        return scores, params

    @staticmethod
    def _entropy(u: Array) -> float:
        u = np.asarray(u, dtype=float).reshape(-1)
        k1 = 79.047
        k2 = 7.4129
        gamma = 0.37457
        return (
            (1.0 + log(2.0 * pi)) / 2.0
            - k1 * (float(np.mean(np.log(np.cosh(u)))) - gamma) ** 2
            - k2 * float(np.mean(u * np.exp(-(u**2) / 2.0))) ** 2
        )

    @classmethod
    def _diff_mutual_info(cls, x_candidate: Array, x_other: Array, r_candidate_on_other: Array, r_other_on_candidate: Array) -> float:
        r_candidate_std = r_candidate_on_other / safe_std(r_candidate_on_other)
        r_other_std = r_other_on_candidate / safe_std(r_other_on_candidate)
        return (
            cls._entropy(x_other)
            + cls._entropy(r_candidate_std)
            - cls._entropy(x_candidate)
            - cls._entropy(r_other_std)
        )

    @classmethod
    def _pwling_score(cls, x_work: Array, candidate: int) -> float:
        x_candidate = standardize_vector(x_work[:, candidate])
        score = 0.0
        for i in range(x_work.shape[1]):
            if i == candidate:
                continue
            x_other = standardize_vector(x_work[:, i])
            r_candidate_on_other = residual(x_candidate, x_other)
            r_other_on_candidate = residual(x_other, x_candidate)
            diff = cls._diff_mutual_info(
                x_candidate,
                x_other,
                r_candidate_on_other,
                r_other_on_candidate,
            )
            score += min(0.0, diff) ** 2
        return float(score)

    def _pwling_candidate_scores(self, x_work: Array) -> tuple[Array, list[None]]:
        scores = np.array(
            [self._pwling_score(x_work, candidate) for candidate in range(x_work.shape[1])],
            dtype=float,
        )
        return scores, [None] * x_work.shape[1]


__all__ = [
    "DirectLiMIAM",
    "FitResult",
    "center_columns",
    "estimate_adjacency_ols_given_order",
    "estimate_adjacency_given_order",
    "fit_order_and_adjacency",
    "make_folds",
    "predict_adaptive_lasso",
    "residual",
    "standardize_columns",
    "standardize_vector",
]
