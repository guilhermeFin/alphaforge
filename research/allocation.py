"""Native, auditable portfolio-allocation primitives."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform


def diagonal_shrinkage_covariance(
    returns: pd.DataFrame,
    *,
    shrinkage: float = 0.10,
    min_observations: int = 20,
) -> pd.DataFrame:
    """Blend sample covariance toward its diagonal for a stable risk estimate."""
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be between 0 and 1.")
    frame = pd.DataFrame(returns).apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if frame.shape[1] < 2:
        raise ValueError("At least two assets are required for hierarchical allocation.")
    if len(frame) < min_observations:
        raise ValueError(f"At least {min_observations} aligned return observations are required.")
    sample = frame.cov().to_numpy(dtype=float)
    target = np.diag(np.diag(sample))
    covariance = (1.0 - shrinkage) * sample + shrinkage * target
    covariance = (covariance + covariance.T) / 2.0
    diagonal_floor = max(float(np.nanmedian(np.diag(covariance))) * 1e-8, 1e-12)
    covariance += np.eye(len(covariance)) * diagonal_floor
    return pd.DataFrame(covariance, index=frame.columns, columns=frame.columns)


def _cluster_variance(covariance: np.ndarray, items: list[int]) -> float:
    sub_covariance = covariance[np.ix_(items, items)]
    inverse_variance = 1.0 / np.clip(np.diag(sub_covariance), 1e-12, None)
    inverse_variance /= inverse_variance.sum()
    return float(inverse_variance @ sub_covariance @ inverse_variance)


def _cluster_order(linkage_matrix: np.ndarray) -> list[int]:
    root, _ = to_tree(linkage_matrix, rd=True)
    return [int(item) for item in root.pre_order()]


def hierarchical_risk_parity(
    returns: pd.DataFrame,
    *,
    shrinkage: float = 0.10,
    linkage_method: str = "single",
    min_observations: int = 20,
) -> dict[str, Any]:
    """Produce long-only HRP weights using only the supplied historical returns."""
    covariance_frame = diagonal_shrinkage_covariance(
        returns, shrinkage=shrinkage, min_observations=min_observations
    )
    covariance = covariance_frame.to_numpy(dtype=float)
    standard_deviation = np.sqrt(np.clip(np.diag(covariance), 1e-12, None))
    correlation = covariance / np.outer(standard_deviation, standard_deviation)
    correlation = np.clip((correlation + correlation.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    distances = np.sqrt(np.clip((1.0 - correlation) / 2.0, 0.0, 1.0))
    hierarchy = linkage(squareform(distances, checks=False), method=linkage_method)
    ordered = _cluster_order(hierarchy)

    weights = np.ones(len(ordered), dtype=float)
    clusters: list[list[int]] = [ordered]
    while clusters:
        next_clusters: list[list[int]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            split = len(cluster) // 2
            left, right = cluster[:split], cluster[split:]
            left_variance = _cluster_variance(covariance, left)
            right_variance = _cluster_variance(covariance, right)
            allocation = 1.0 - left_variance / max(left_variance + right_variance, 1e-12)
            weights[[ordered.index(item) for item in left]] *= allocation
            weights[[ordered.index(item) for item in right]] *= 1.0 - allocation
            next_clusters.extend([left, right])
        clusters = next_clusters

    result = pd.Series(0.0, index=covariance_frame.columns, dtype=float)
    result.iloc[ordered] = weights
    result /= result.sum()
    return {
        "weights": result,
        "cluster_order": [str(covariance_frame.columns[item]) for item in ordered],
        "covariance_method": f"{int(shrinkage * 100)}% diagonal shrinkage",
        "n_observations": int(len(pd.DataFrame(returns).dropna(how="any"))),
    }
