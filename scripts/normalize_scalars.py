"""Express length and height as fractions of each animal's baseline body size.

Body size (Lmax) is the 99.5th percentile of the 3D head to tail-base distance over the
early session of the animal. Speeds and position stay in physical units.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import parse_config, results_path  # noqa: E402
from src.kinematics import SCALARS  # noqa: E402
from src.sessions import load_sessions  # noqa: E402


def main(cfg: dict) -> None:
    sessions = load_sessions(cfg)
    frames = pd.read_csv(results_path(cfg, "frame_scalars.csv"), dtype={"name": str})
    frames = frames.merge(sessions[["name", "animal_id", "timepoint"]], on="name", how="inner")

    frames["distance_3d"] = np.hypot(frames["length"].to_numpy(float), frames["height"].to_numpy(float))
    lmax = (
        frames[frames["timepoint"] == "early"]
        .groupby("animal_id")["distance_3d"]
        .quantile(cfg["normalization"]["body_size_quantile"])
        .rename("lmax")
    )
    frames = frames.merge(lmax, on="animal_id", how="inner")
    for column in ("length", "height"):
        frames[column] = frames[column] / frames["lmax"]

    frames[["name", *SCALARS]].to_csv(results_path(cfg, "frame_scalars_normalized.csv"), index=False)
    body_size = lmax.reset_index().merge(sessions[["animal_id", "genotype"]].drop_duplicates(), on="animal_id")
    body_size.to_csv(results_path(cfg, "body_size.csv"), index=False)


if __name__ == "__main__":
    main(parse_config(__doc__))
