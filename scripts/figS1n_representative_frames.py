"""Fig. S1N: select representative frames of the long+high and short+low posture states.

For each state, the frames closest to the center of the state are selected, one per animal.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import parse_config, results_path, source_data_path  # noqa: E402
from src.posture import HEIGHT_BINS, STATES, assign_states  # noqa: E402
from src.sessions import load_frames  # noqa: E402


def select_frames(frames: pd.DataFrame, l_cuts: np.ndarray, h_cut: float, params: dict) -> pd.DataFrame:
    frames = frames.assign(frame=frames.groupby("name").cumcount())
    frames["state"] = assign_states(frames["length"].to_numpy(), frames["height"].to_numpy(), l_cuts, h_cut)
    length_edges = [0, *l_cuts, frames["length"].max()]
    height_edges = [0, h_cut, frames["height"].max()]

    selected = []
    for state in params["states"]:
        state_index = STATES.index(state)
        length_bin, height_bin = divmod(state_index, len(HEIGHT_BINS))
        center_length = (length_edges[length_bin] + length_edges[length_bin + 1]) / 2
        center_height = (height_edges[height_bin] + height_edges[height_bin + 1]) / 2
        candidates = frames[(frames["group"] == params["group"]) & (frames["state"] == state_index)]
        distance = (candidates["length"] - center_length) ** 2 + (candidates["height"] - center_height) ** 2
        ranked = candidates.assign(distance=distance).sort_values("distance")
        best = ranked.drop_duplicates("animal_id").head(params["n_per_state"])
        selected.append(best.assign(state_name=state, rank=range(1, len(best) + 1)))
    columns = ["name", "animal_id", "group", "state_name", "rank", "frame", "length", "height"]
    return pd.concat(selected)[columns].rename(columns={"state_name": "state"})


def main(cfg: dict) -> None:
    cuts = pd.read_csv(results_path(cfg, "posture_cuts.csv"), float_precision="round_trip").set_index("cut")["value"]
    l_cuts = cuts[["length_short_mid", "length_mid_long"]].to_numpy()
    selection = select_frames(load_frames(cfg), l_cuts, cuts["height_low_high"], cfg["representative_frames"])
    selection.to_csv(source_data_path(cfg, "figS1n_selected_frames.csv"), index=False)


if __name__ == "__main__":
    main(parse_config(__doc__))
