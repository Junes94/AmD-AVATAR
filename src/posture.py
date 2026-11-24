"""Six posture states on the normalized length-height plane."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

LENGTH_BINS = ["short", "mid", "long"]
HEIGHT_BINS = ["low", "high"]
STATES = [f"{length}_{height}" for length in LENGTH_BINS for height in HEIGHT_BINS]


def length_cuts(length: np.ndarray) -> np.ndarray:
    """Tertiles of the reference distribution."""
    return np.percentile(length[np.isfinite(length)], [100 / 3, 200 / 3])


def height_cut(height: np.ndarray, bandwidth: float, max_frames: int, seed: int) -> float:
    """Antimode between the low and high modes of the reference height density."""
    values = height[np.isfinite(height)]
    sample = np.random.default_rng(seed).choice(values, min(max_frames, len(values)), replace=False)
    grid = np.linspace(0.0, 1.05, 600)
    density = gaussian_kde(sample, bw_method=bandwidth)(grid)
    for i in range(2, len(grid) - 2):
        if density[i] < density[i - 1] and density[i] < density[i + 1] and 0.4 < grid[i] < 0.95:
            return float(grid[i])
    raise ValueError("The reference height density has no antimode.")


def assign_states(length: np.ndarray, height: np.ndarray, l_cuts: np.ndarray, h_cut: float) -> np.ndarray:
    """State index of each frame, in the order of STATES."""
    return np.digitize(length, l_cuts) * len(HEIGHT_BINS) + np.digitize(height, [h_cut])


def state_occupancy(frames: pd.DataFrame, l_cuts: np.ndarray, h_cut: float) -> pd.DataFrame:
    """Fraction of frames in each state, per session."""
    valid = frames[np.isfinite(frames["length"]) & np.isfinite(frames["height"])]
    states = assign_states(valid["length"].to_numpy(), valid["height"].to_numpy(), l_cuts, h_cut)
    counts = (
        pd.DataFrame({"name": valid["name"].to_numpy(), "state": states})
        .groupby(["name", "state"]).size().unstack(fill_value=0)
        .reindex(columns=range(len(STATES)), fill_value=0)
    )
    occupancy = counts.div(counts.sum(axis=1), axis=0)
    occupancy.columns = STATES
    return occupancy.reset_index()


def clr(composition: np.ndarray) -> np.ndarray:
    """Centered log-ratio transform. Zeros are replaced by half the smallest positive value."""
    pseudocount = 0.5 * composition[composition > 0].min()
    log_x = np.log(np.where(composition > 0, composition, pseudocount))
    return log_x - log_x.mean(axis=1, keepdims=True)
