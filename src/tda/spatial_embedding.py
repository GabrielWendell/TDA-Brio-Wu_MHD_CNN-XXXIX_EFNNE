"""
src/tda/spatial_embedding.py

Spatial Takens-style delay embeddings for 1D fields f(x_i).

Given a discrete profile f[i] = f(x_i), we construct

    Φ_{m,τ}(i) = (f[i], f[i+τ], ..., f[i+(m-1)τ]) ∈ R^m

for all i such that i + (m-1)τ < N_x.

Optionally we can subsample the resulting point cloud (stride > 1) to speed up
persistent homology computations.
"""

from __future__ import annotations

import numpy as np
from typing import Literal


def spatial_delay_embedding(
    field_1d: np.ndarray,
    m: int,
    tau: int,
    stride: int = 1,
    normalize: Literal["none", "center", "zscore"] = "none",
) -> np.ndarray:
    """
    Build a spatial delay embedding for a 1D field.

    Parameters
    ----------
    field_1d : np.ndarray, shape (N_x,)
        1D array f[i] = f(x_i) at fixed time.
        It is assumed to be already in the desired normalization
        (e.g. z-score over (t,x)); we can optionally normalize each
        window again if desired.
    m : int
        Embedding dimension (number of delays).
        Typical values: 3-8.
    tau : int
        Spatial lag in grid points between delays.
        Typical values: 1-4.
    stride : int, optional
        Subsampling step for starting indices i. If stride = 1 we use all i
        such that i + (m-1)*tau < N_x. If stride > 1 we use only every stride-th i.
        This is useful to reduce the point-cloud size for TDA.
    normalize : {"none", "center", "zscore"}, optional
        Optional *per-window* normalization:
        - "none": do nothing.
        - "center": subtract window mean.
        - "zscore": subtract window mean and divide by window std
          (if std>0; otherwise leave window as zeros).

    Returns
    -------
    points : np.ndarray, shape (N_points, m)
        Embedded point cloud in R^m.

    Notes
    -----
    The number of available starting indices is

        N_start = N_x - (m-1)*tau

    If N_start <= 0 the function raises a ValueError.
    """
    f = np.asarray(field_1d).astype(float).ravel()
    N_x = f.shape[0]

    if m <= 0 or tau <= 0:
        raise ValueError("Embedding parameters m and tau must be positive integers.")
    max_shift = (m - 1) * tau
    N_start = N_x - max_shift
    if N_start <= 0:
        raise ValueError(
            f"Embedding not possible: N_x = {N_x}, m = {m}, tau = {tau} "
            f"⇒ N_start = {N_start} <= 0."
        )

    # Indices of starting points
    start_indices = np.arange(0, N_start, stride, dtype = int)
    N_points = start_indices.shape[0]

    # Allocate point cloud
    points = np.empty((N_points, m), dtype = float)

    for j, i0 in enumerate(start_indices):
        window = f[i0 : i0 + max_shift + 1 : tau]  # length m
        if window.shape[0] != m:
            # This should not happen if the logic above is correct
            raise RuntimeError(
                f"Internal error: expected window of length {m}, got {window.shape[0]}"
            )
        if normalize == "center":
            window = window - window.mean()
        elif normalize == "zscore":
            mu = window.mean()
            sigma = window.std()
            if sigma > 0:
                window = (window - mu) / sigma
            else:
                window = window * 0.0  # All equal values
        points[j, :] = window

    return points