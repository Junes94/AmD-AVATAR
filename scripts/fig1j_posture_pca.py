"""Fig. 1J: PCA of the CLR-transformed posture-state occupancy of each session."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import parse_config, source_data_path  # noqa: E402
from src.posture import STATES, clr  # noqa: E402
from src.sessions import GROUPS, load_occupancy  # noqa: E402


def main(cfg: dict) -> None:
    occupancy = load_occupancy(cfg)
    features = clr(occupancy[sorted(STATES)].to_numpy(float))
    centered = features - features.mean(axis=0)
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    scores = centered @ components[:2].T
    explained = singular_values ** 2 / (singular_values ** 2).sum() * 100

    sessions = occupancy[["name", "group", "animal_id"]].assign(PC1=scores[:, 0], PC2=scores[:, 1])
    sessions.to_csv(source_data_path(cfg, "fig1j_pca_sessions.csv"), index=False)

    groups = sessions.groupby("group")[["PC1", "PC2"]].agg(["mean", "sem", "count"]).reindex(GROUPS)
    groups.columns = ["_".join(column) for column in groups.columns]
    groups.reset_index().to_csv(source_data_path(cfg, "fig1j_pca_groups.csv"), index=False)

    pd.DataFrame({"component": ["PC1", "PC2"], "explained_variance_percent": explained[:2]}).to_csv(
        source_data_path(cfg, "fig1j_pca_variance.csv"), index=False
    )


if __name__ == "__main__":
    main(parse_config(__doc__))
