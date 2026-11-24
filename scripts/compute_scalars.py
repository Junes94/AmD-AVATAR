"""Compute per-frame kinematic scalars from the 3D keypoints of each session."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import parse_config, results_path  # noqa: E402
from src.kinematics import SCALARS, compute_scalars, read_keypoints  # noqa: E402
from src.sessions import load_sessions  # noqa: E402


def main(cfg: dict) -> None:
    params = cfg["kinematics"]
    tables = []
    for name in load_sessions(cfg)["name"]:
        pose = read_keypoints(cfg["paths"]["coordinates"] / f"{name}.csv", cfg["keypoints"])
        scalars = compute_scalars(pose, params["fps"], params["smoothing_window"])
        scalars.insert(0, "name", name)
        tables.append(scalars)
    pd.concat(tables, ignore_index=True)[["name", *SCALARS]].to_csv(
        results_path(cfg, "frame_scalars.csv"), index=False
    )


if __name__ == "__main__":
    main(parse_config(__doc__))
