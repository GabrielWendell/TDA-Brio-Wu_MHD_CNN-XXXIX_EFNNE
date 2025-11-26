"""
src/tda/persistence_tools.py

Wrappers around ripser for Vietoris-Rips persistent homology,
plus utilities for Betti curves and persistence landscapes.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple

from ripser import ripser


# -------------------------------------------------------------------------
# 1. Vietoris–Rips persistence
# -------------------------------------------------------------------------
def compute_vr_persistence(
    point_cloud: np.ndarray,
    maxdim: int = 1,
    thresh: float | None = None,
) -> List[np.ndarray]:
    """
    Compute Vietoris-Rips persistent homology of a point cloud using ripser.

    Parameters
    ----------
    point_cloud : np.ndarray, shape (N_points, dim)
        Point cloud in R^dim.
    maxdim : int, optional
        Maximum homology dimension to compute. Default is 1 (H0 and H1).
    thresh : float or None, optional
        Maximum filtration value (distance threshold) to use. If None,
        we let ripser choose its default.

    Returns
    -------
    diagrams : list of np.ndarray
        diagrams[d] is an array of shape (N_features_d, 2) with birth-death
        pairs for H_d.
    """
    X = np.asarray(point_cloud, dtype = float)

    if thresh is None:
        # Do NOT pass thresh at all – some ripser versions don’t accept None.
        res = ripser(X, maxdim = maxdim, distance_matrix = False)
    else:
        # Ensure thresh is a float
        res = ripser(X, maxdim = maxdim, thresh = float(thresh), distance_matrix = False)

    diagrams = res["dgms"]
    return diagrams


# -------------------------------------------------------------------------
# 2. Betti curves
# -------------------------------------------------------------------------
def betti_curve(
    diagram: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    """
    Compute the Betti curve β(ε) from a single persistence diagram.

    Parameters
    ----------
    diagram : np.ndarray, shape (N_features, 2)
        Array of birth-death pairs (b_i, d_i). Death may be np.inf.
    radii : np.ndarray, shape (N_r,)
        Filtration radii ε at which to evaluate the Betti number.

    Returns
    -------
    beta : np.ndarray, shape (N_r,)
        Betti numbers β(ε_j) = number of features alive at scale ε_j.

    Definition
    ----------
    A feature (b_i, d_i) is counted as alive at ε if

        b_i <= ε < d_i      (with the convention d_i = +∞ allowed).

    This is the standard convention for persistent homology.
    """
    if diagram.size == 0:
        return np.zeros_like(radii, dtype=int)

    b = diagram[:, 0].reshape(-1, 1)  # (N_features, 1)
    d = diagram[:, 1].reshape(-1, 1)  # (N_features, 1)

    eps = radii.reshape(1, -1)        # (1, N_r)

    alive = (b <= eps) & (eps < d)    # (N_features, N_r)
    beta = alive.sum(axis=0)          # (N_r,)

    return beta


def betti_curves_from_diagrams(
    diagrams: List[np.ndarray],
    radii: np.ndarray,
    maxdim: int = 1,
) -> Dict[int, np.ndarray]:
    """
    Compute Betti curves for all dimensions up to maxdim.

    Parameters
    ----------
    diagrams : list of np.ndarray
        Output from compute_vr_persistence.
    radii : np.ndarray, shape (N_r,)
        Filtration radii.
    maxdim : int, optional
        Maximum dimension to consider.

    Returns
    -------
    betti_dict : dict
        betti_dict[d] = β_d(ε) as a numpy array of shape (N_r,)
        for each 0 <= d <= maxdim.
    """
    betti_dict: Dict[int, np.ndarray] = {}
    for d in range(maxdim + 1):
        diag_d = diagrams[d]
        beta_d = betti_curve(diag_d, radii)
        betti_dict[d] = beta_d
    return betti_dict


# -------------------------------------------------------------------------
# 3. Persistence landscapes (optional)
# -------------------------------------------------------------------------
def persistence_landscape(
    diagram: np.ndarray,
    radii: np.ndarray,
    n_layers: int = 3,
) -> np.ndarray:
    """
    Compute the first n_layers of the persistence landscape λ_k(ε)
    for a single persistence diagram.

    Parameters
    ----------
    diagram : np.ndarray, shape (N_features, 2)
        Birth-death pairs (b_i, d_i).
    radii : np.ndarray, shape (N_r,)
        Grid of ε values at which to evaluate the landscape.
    n_layers : int, optional
        Number of landscape layers k = 1,...,n_layers to compute.

    Returns
    -------
    L : np.ndarray, shape (n_layers, N_r)
        L[k-1, j] = λ_k(ε_j), where λ_1 ≥ λ_2 ≥ ... at each ε.
    """
    radii = np.asarray(radii, dtype = float)
    N_r = radii.shape[0]
    L = np.zeros((n_layers, N_r), dtype = float)

    # No features ⇒ landscape identically zero
    if diagram.size == 0:
        return L

    # Ensure diagram is 2D (N_features, 2)
    diag = np.asarray(diagram, dtype = float)
    if diag.ndim != 2 or diag.shape[1] != 2:
        raise ValueError(f"Diagram must have shape (N_features, 2), got {diag.shape}")

    # Column vectors (N_features, 1)
    b = diag[:, 0].reshape(-1, 1)
    d = diag[:, 1].reshape(-1, 1)

    # Ignore infinite deaths (features that never die)
    finite = np.isfinite(d)
    b = b[finite]
    d = d[finite]
    if b.size == 0:
        return L
    
    # Reshape again after boolean indexing
    b = b.reshape(-1, 1)
    d = d.reshape(-1, 1)

    # Row vector (1, N_r)
    eps = radii.reshape(1, -1)

    # For each feature, compute triangle height at each ε:
    left  = eps - b        # > 0 on right of birth
    right = d - eps        # > 0 on left of death
    tent  = np.minimum(left, right)
    tent[tent < 0] = 0.0   # outside [b, d]

    # tent: (N_features, N_r). For each ε, sort descending over features.
    tent_sorted = np.sort(tent, axis = 0)[::-1, :]  # descending per ε

    n_feat = tent_sorted.shape[0]
    k_max = min(n_layers, n_feat)
    L[:k_max, :] = tent_sorted[:k_max, :]

    return L