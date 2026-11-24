"""Fig. 1K: within-animal early-to-late shift in posture composition.

Plotted values: Aitchison distance between the early and late session of each animal.
Tests: sign-flip permutation test of the mean change vector within each genotype, label permutation
test of the difference between the two genotype-mean change vectors, and exact Mann-Whitney U test
of the per-animal distances.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import parse_config, source_data_path  # noqa: E402
from src.posture import STATES, clr  # noqa: E402
from src.sessions import load_occupancy  # noqa: E402
from src.stats import bh_fdr, group_difference_test, sign_flip_test  # noqa: E402


def main(cfg: dict) -> None:
    stats = cfg["statistics"]
    n_permutations = stats["n_permutations"]
    occupancy = load_occupancy(cfg)
    features = clr(occupancy[sorted(STATES)].to_numpy(float))

    animals, changes = [], []
    for animal_id, rows in occupancy.groupby("animal_id"):
        early = features[rows.index[rows["timepoint"] == "early"]].mean(axis=0)
        late = features[rows.index[rows["timepoint"] == "late"]].mean(axis=0)
        animals.append({"animal_id": animal_id, "genotype": rows["genotype"].iloc[0],
                        "aitchison_distance": np.linalg.norm(late - early)})
        changes.append(late - early)
    animals = pd.DataFrame(animals)
    changes = np.vstack(changes)
    is_cko = (animals["genotype"] == "cKO").to_numpy()

    rng = np.random.default_rng(stats["permutation_seed"])
    p_cko = sign_flip_test(changes[is_cko], n_permutations, rng)
    p_wt = sign_flip_test(changes[~is_cko], n_permutations, rng)
    p_between = group_difference_test(changes, is_cko, n_permutations, rng)
    q_wt, q_cko = bh_fdr([p_wt, p_cko])
    distance = animals["aitchison_distance"].to_numpy()
    p_distance = mannwhitneyu(distance[is_cko], distance[~is_cko], alternative="two-sided", method="exact").pvalue

    animals.to_csv(source_data_path(cfg, "fig1k_per_animal.csv"), index=False)
    summary = animals.groupby("genotype")["aitchison_distance"].agg(["mean", "sem", "count"]).reset_index()
    summary.to_csv(source_data_path(cfg, "fig1k_groups.csv"), index=False)
    pd.DataFrame({
        "test": ["WT_direction", "cKO_direction", "mean_vector_cKO_vs_WT", "distance_cKO_vs_WT"],
        "p": [p_wt, p_cko, p_between, p_distance],
        "q": [q_wt, q_cko, np.nan, np.nan],
    }).to_csv(source_data_path(cfg, "fig1k_tests.csv"), index=False)


if __name__ == "__main__":
    main(parse_config(__doc__))
