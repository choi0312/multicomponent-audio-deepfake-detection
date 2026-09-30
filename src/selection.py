"""Training and model-selection utilities for the lightweight fusion heads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def fit_nonnegative_logistic_head(
    matrix: np.ndarray,
    labels: Sequence[int],
    sample_weights: Sequence[float],
    feature_names: Sequence[str],
    regularization: float,
) -> dict[str, object]:
    """Fit the constrained head used in the candidate sweep.

    Coefficients are non-negative so positive fake evidence cannot be inverted.
    A previous-model feature is capped at 0.5 and receives a stronger penalty;
    this keeps it as an anchor instead of allowing it to absorb the new experts.
    Normalization is estimated from the training rows only and exported with the
    coefficients. No target-batch normalization is performed at inference.
    """

    x = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    weights = np.asarray(sample_weights, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(y) or len(y) != len(weights):
        raise ValueError("Incompatible training arrays")
    if len(feature_names) != x.shape[1]:
        raise ValueError("Feature names do not match matrix columns")

    mean = np.average(x, axis=0, weights=weights)
    scale = np.maximum(
        0.3, np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weights))
    )
    standardized = (x - mean) / scale

    penalty = np.asarray(
        [4.0 if "previous" in name else 1.0 for name in feature_names]
    )
    bounds = [
        (0.0, 0.5 if "previous" in name else 3.0)
        for name in feature_names
    ] + [(-8.0, 8.0)]

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        margin = standardized @ beta[:-1] + beta[-1]
        probability = expit(margin)
        loss = np.sum(weights * (np.logaddexp(0.0, margin) - y * margin))
        loss += 0.5 * regularization * np.sum(penalty * beta[:-1] ** 2)
        gradient = np.r_[
            standardized.T @ (weights * (probability - y))
            + regularization * penalty * beta[:-1],
            np.sum(weights * (probability - y)),
        ]
        return float(loss), gradient

    result = minimize(
        objective,
        np.zeros(x.shape[1] + 1),
        jac=True,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 1500, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(result.message)

    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coef": result.x[:-1].tolist(),
        "bias": float(result.x[-1]),
        "features": list(feature_names),
        "regularization": float(regularization),
    }


def robust_component_error(
    major_group_eers: Sequence[float], reliable_subgroup_eers: Sequence[float]
) -> float:
    """Combine average performance and a tail-risk penalty.

    Callers must exclude groups with fewer than ten real or ten fake examples
    before passing values here.
    """

    major = np.asarray(major_group_eers, dtype=np.float64)
    robust = np.asarray(reliable_subgroup_eers, dtype=np.float64)
    if not len(major) or not len(robust):
        raise ValueError("At least one major group and subgroup are required")
    if not np.isfinite(major).all() or not np.isfinite(robust).all():
        raise ValueError("EER inputs must be finite")
    return float(0.5 * major.mean() + 0.5 * np.quantile(robust, 0.9))


def ads_selection_error(component_errors: Mapping[str, float]) -> float:
    """Apply the official ADS component weights to robust validation errors."""

    return float(
        0.5 * component_errors["file"]
        + 0.2 * component_errors["voice"]
        + 0.3 * component_errors["music"]
    )

