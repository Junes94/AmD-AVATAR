"""Per-frame kinematic scalars from 3D keypoint trajectories."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SCALARS = ["length", "height", "speed_xy", "speed_z", "position"]


def read_keypoints(path: str | Path, keypoints: list[str]) -> pd.DataFrame:
    """Read a headerless file with one x, y, z column triplet per keypoint."""
    columns = [f"{keypoint}_{axis}" for keypoint in keypoints for axis in "xyz"]
    pose = pd.read_csv(path, header=None).iloc[:, : len(columns)]
    pose.columns = columns
    return pose


def compute_scalars(pose: pd.DataFrame, fps: int, smoothing_window: int) -> pd.DataFrame:
    """Compute length, height, speeds and position for every frame.

    Trajectories are smoothed with a centered moving average. Length is the horizontal
    head to tail-base distance and height the signed vertical difference. Speeds are
    taken from the midpoint between head and torso. Position is the horizontal distance
    of the torso from the arena center, which is the origin of the coordinates.
    """
    p = pose.rolling(window=smoothing_window, min_periods=1, center=True).mean()
    dt = 1 / fps
    mid_x = (p["head_x"] + p["torso_x"]) / 2
    mid_y = (p["head_y"] + p["torso_y"]) / 2
    mid_z = (p["head_z"] + p["torso_z"]) / 2
    return pd.DataFrame({
        "length": np.sqrt((p["head_x"] - p["tail_base_x"]) ** 2 + (p["head_y"] - p["tail_base_y"]) ** 2),
        "height": p["head_z"] - p["tail_base_z"],
        "speed_xy": np.sqrt(mid_x.diff() ** 2 + mid_y.diff() ** 2) / dt,
        "speed_z": (mid_z.diff() / dt).abs(),
        "position": np.sqrt(p["torso_x"] ** 2 + p["torso_y"] ** 2),
    })
