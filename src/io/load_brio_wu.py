"""
src/io/load_brio_wu.py

Utilities to load Brio-Wu shock tube data from MATLAB .mat files.

Expected filename pattern (already used in the uploaded files):

    BrioWuShockTube_Output_Time_0.00.mat
    BrioWuShockTube_Output_Time_0.05.mat
    ...
    BrioWuShockTube_Output_Time_2.00.mat

This module builds time-sorted tensors:

    rho[t_idx, x_idx]
    p[t_idx, x_idx]

and, if available, also returns the spatial grid x[x_idx].
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.io import loadmat


# ---------- Data structure ------------------------------------------------- #

@dataclass
class BrioWuDataset:
    """
    Container for the Brio-Wu MHD shock tube simulation data.

    Attributes
    ----------
    times : np.ndarray, shape (N_t,)
        Time grid: t_k from filenames.
    x : np.ndarray, shape (N_x,)
        Spatial grid. If not found in the .mat files, a uniform grid
        on [0, 1] is constructed as a placeholder.
    rho : np.ndarray, shape (N_t, N_x)
        Density field rho(t_k, x_i).
    p : np.ndarray, shape (N_t, N_x)
        Pressure field p(t_k, x_i).
    extra : Dict[str, np.ndarray]
        Optional additional fields (e.g. velocity, magnetic field).
        Keys are variable names, values have shape (N_t, N_x).
    """
    times: np.ndarray
    x: np.ndarray
    rho: np.ndarray
    p: np.ndarray
    extra: Dict[str, np.ndarray]


# ---------- Helper functions ---------------------------------------------- #

_TIME_RE = re.compile(
    r"BrioWuShockTube_Output_Time_([0-9]+(?:\.[0-9]+)?)\.mat$"
)

def _parse_time_from_filename(filename: str) -> float:
    """
    Extract the simulation time from a file name.

    Examples
    --------
    'BrioWuShockTube_Output_Time_0.00.mat' -> 0.0
    'BrioWuShockTube_Output_Time_1.25.mat' -> 1.25
    """
    m = _TIME_RE.search(os.path.basename(filename))
    if m is None:
        raise ValueError(f"Could not parse time from filename: {filename}")
    return float(m.group(1))


def _guess_variable(mat_dict: Dict, candidates: Sequence[str]) -> Optional[str]:
    """
    Given a dict returned by scipy.io.loadmat, try to guess which key
    corresponds to a given physical quantity by checking a list of
    candidate names.

    This is robust if, for instance, the variable is called 'rho' in
    one version and 'density' in another.
    """
    keys = set(mat_dict.keys())
    for name in candidates:
        if name in keys:
            return name
    return None


# ---------- Main loading routine ----------------------------------------- #

def load_brio_wu_data(
    data_dir: str,
    rho_candidates: Sequence[str] = ("rho", "density", "rho_array", "Density"),
    p_candidates: Sequence[str]   = ("p", "pressure", "P", "Pressure"),
    x_candidates: Sequence[str]   = ("x", "x_grid", "X")
) -> BrioWuDataset:
    """
    Load all Brio-Wu .mat files in `data_dir` and build time-ordered tensors.

    Parameters
    ----------
    data_dir : str
        Directory containing the .mat files.
    rho_candidates : sequence of str, optional
        Possible variable names for the density field in the .mat files.
    p_candidates : sequence of str, optional
        Possible variable names for the pressure field.
    x_candidates : sequence of str, optional
        Possible variable names for the spatial grid array.

    Returns
    -------
    BrioWuDataset
        A data structure with times, grid, rho, p, and optional extra fields.

    Notes
    -----
    If the spatial grid `x` is not present in the .mat files, a uniform
    grid on [0, 1] with N_x points is generated as a placeholder.
    """

    # --- 1. Collect and sort files by time ---
    all_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".mat") and "BrioWuShockTube_Output_Time_" in f
    ]
    if not all_files:
        raise RuntimeError(f"No Brio-Wu .mat files found in directory: {data_dir}")

    # Pair each path with its time
    time_file_pairs: List[Tuple[float, str]] = []
    for path in all_files:
        t = _parse_time_from_filename(path)
        time_file_pairs.append((t, path))

    # Sort by time
    time_file_pairs.sort(key = lambda pair: pair[0])
    times = np.array([pair[0] for pair in time_file_pairs], dtype = float)

    # --- 2. Load first file to discover variable names and shapes ---
    sample_t, sample_path = time_file_pairs[0]
    sample_mat = loadmat(sample_path)

    # Remove MATLAB metadata keys (those starting with '__')
    sample_mat_clean = {
        k: v for k, v in sample_mat.items()
        if not k.startswith("__")
    }

    if not sample_mat_clean:
        raise RuntimeError(f"MAT-file {sample_path} contains no user variables.")

    rho_key = _guess_variable(sample_mat_clean, rho_candidates)
    p_key   = _guess_variable(sample_mat_clean, p_candidates)
    x_key   = _guess_variable(sample_mat_clean, x_candidates)

    if rho_key is None:
        raise KeyError(
            f"Could not find a density variable in {sample_path}. "
            f"Tried candidates: {rho_candidates}"
        )
    if p_key is None:
        raise KeyError(
            f"Could not find a pressure variable in {sample_path}. "
            f"Tried candidates: {p_candidates}"
        )

    # Extract sample arrays and infer shapes
    rho_sample = np.asarray(sample_mat_clean[rho_key]).squeeze()
    p_sample   = np.asarray(sample_mat_clean[p_key]).squeeze()

    if rho_sample.shape != p_sample.shape:
        raise ValueError(
            f"Density and pressure shapes do not match in {sample_path}: "
            f"rho {rho_sample.shape}, p {p_sample.shape}"
        )

    # We expect either shape (N_x,) or (1, N_x) or (N_x, 1).
    if rho_sample.ndim != 1:
        raise ValueError(
            f"Expected 1D arrays for rho and p in {sample_path}, "
            f"got rho.ndim = {rho_sample.ndim}"
        )

    N_x = rho_sample.shape[0]
    N_t = len(time_file_pairs)

    # Spatial grid
    if x_key is not None:
        x_sample = np.asarray(sample_mat_clean[x_key]).squeeze()
        if x_sample.shape[0] != N_x:
            raise ValueError(
                f"Spatial grid x has incompatible length in {sample_path}: "
                f"len(x) = {x_sample.shape[0]}, expected N_x = {N_x}"
            )
        x = x_sample.astype(float)
    else:
        # Construct a placeholder uniform grid on [0, 1]
        x = np.linspace(0.0, 1.0, N_x, endpoint = False)

    # --- 3. Allocate tensors ---
    rho = np.zeros((N_t, N_x), dtype=float)
    p   = np.zeros((N_t, N_x), dtype=float)

    # Optional extra fields: anything 1D of length N_x that is not rho/p/x
    extra_keys = []
    for key, arr in sample_mat_clean.items():
        if key in {rho_key, p_key, x_key}:
            continue
        arr = np.asarray(arr).squeeze()
        if arr.ndim == 1 and arr.shape[0] == N_x:
            extra_keys.append(key)

    extra = {k: np.zeros((N_t, N_x), dtype=float) for k in extra_keys}

    # Fill tensors time-step by time-step
    for t_idx, (t, path) in enumerate(time_file_pairs):
        mat = loadmat(path)
        mat_clean = {k: v for k, v in mat.items() if not k.startswith("__")}

        rho_arr = np.asarray(mat_clean[rho_key]).squeeze().astype(float)
        p_arr   = np.asarray(mat_clean[p_key]).squeeze().astype(float)
        if rho_arr.shape != (N_x,) or p_arr.shape != (N_x,):
            raise ValueError(
                f"Inconsistent shapes in {path}: "
                f"rho {rho_arr.shape}, p {p_arr.shape}, expected ({N_x},)"
            )
        rho[t_idx, :] = rho_arr
        p[t_idx, :]   = p_arr

        for k in extra_keys:
            arr = np.asarray(mat_clean[k]).squeeze().astype(float)
            if arr.shape != (N_x,):
                raise ValueError(
                    f"Inconsistent shape for extra field '{k}' in {path}: "
                    f"{arr.shape}, expected ({N_x},)"
                )
            extra[k][t_idx, :] = arr

    return BrioWuDataset(times = times, x = x, rho = rho, p = p, extra = extra)
