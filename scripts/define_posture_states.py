"""Define the six posture states on the reference group and compute their occupancy."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import parse_config, results_path  # noqa: E402
from src.posture import height_cut, length_cuts, state_occupancy  # noqa: E402
from src.sessions import load_frames  # noqa: E402


def main(cfg: dict) -> None:
    params = cfg["posture_states"]
    frames = load_frames(cfg)
    reference = frames[frames["group"] == params["reference_group"]]

    l_cuts = length_cuts(reference["length"].to_numpy(float))
    h_cut = height_cut(
        reference["height"].to_numpy(float),
        params["height_kde_bandwidth"],
        params["height_kde_max_frames"],
        params["height_kde_seed"],
    )
    pd.DataFrame({"cut": ["length_short_mid", "length_mid_long", "height_low_high"],
                  "value": [l_cuts[0], l_cuts[1], h_cut]}).to_csv(
        results_path(cfg, "posture_cuts.csv"), index=False
    )
    state_occupancy(frames, l_cuts, h_cut).to_csv(results_path(cfg, "posture_occupancy.csv"), index=False)


if __name__ == "__main__":
    main(parse_config(__doc__))
