"""Session metadata and loaders for the per-frame and per-session tables."""
from __future__ import annotations

import pandas as pd

from src.config import results_path
from src.kinematics import SCALARS
from src.posture import STATES

GROUPS = ["WT_early", "WT_late", "cKO_early", "cKO_late"]
NAME_PATTERN = r"^(?P<prefix>ko|wt)_(?P<cohort>s\d+)d\d+_(?P<mouse>\d+)$"
GENOTYPE_PREFIX = {"ko": "cKO", "wt": "WT"}


def load_sessions(cfg: dict) -> pd.DataFrame:
    """Return the analyzed sessions with animal, genotype and timepoint labels.

    Recording names follow <ko|wt>_<cohort>d<day>_<mouse>. Early and late are taken
    from the group label because the recording day differs between mice.
    """
    index = pd.read_csv(cfg["paths"]["index"], dtype={"name": str})
    sessions = index[index["group"].isin(GROUPS)].reset_index(drop=True)
    parts = sessions["name"].str.extract(NAME_PATTERN)
    if parts.isna().any(axis=None):
        raise ValueError("Recording names do not match the expected pattern.")

    sessions[["genotype", "timepoint"]] = sessions["group"].str.split("_", expand=True)
    if not parts["prefix"].map(GENOTYPE_PREFIX).equals(sessions["genotype"]):
        raise ValueError("Genotype in the recording name disagrees with the group label.")
    sessions["animal_id"] = sessions["genotype"] + "_" + parts["cohort"] + "_" + parts["mouse"]
    return sessions


def load_frames(cfg: dict) -> pd.DataFrame:
    """Per-frame normalized scalars with session labels."""
    frames = pd.read_csv(results_path(cfg, "frame_scalars_normalized.csv"), dtype={"name": str})
    return frames.merge(load_sessions(cfg), on="name", how="inner")


def load_session_means(cfg: dict) -> pd.DataFrame:
    """Mean of each normalized scalar per session, in index order."""
    frames = pd.read_csv(results_path(cfg, "frame_scalars_normalized.csv"), dtype={"name": str})
    means = frames.groupby("name")[SCALARS].mean().reset_index()
    return load_sessions(cfg).merge(means, on="name", how="left")


def load_occupancy(cfg: dict) -> pd.DataFrame:
    """Posture-state occupancy per session, in index order."""
    occupancy = pd.read_csv(results_path(cfg, "posture_occupancy.csv"), dtype={"name": str})
    return load_sessions(cfg).merge(occupancy[["name", *STATES]], on="name", how="left")
