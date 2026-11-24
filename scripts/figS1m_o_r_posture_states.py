"""Fig. S1M and S1O-R: posture-state occupancy and its changes.

S1M: mean occupancy per group.
S1O, P: early-to-late change within WT and cKO (conditional exact Wilcoxon signed-rank test).
S1Q, R: cKO minus WT at the early and late timepoints (conditional exact rank-sum test).
Both tests keep tied values. The six states of each panel form one Benjamini-Hochberg family.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import parse_config, source_data_path  # noqa: E402
from src.posture import STATES  # noqa: E402
from src.sessions import GROUPS, load_occupancy  # noqa: E402
from src.stats import bh_fdr, cliffs_delta, conditional_rank_sum_p, conditional_signed_rank_p  # noqa: E402


def within_genotype(occupancy: pd.DataFrame, genotype: str) -> pd.DataFrame:
    subset = occupancy[occupancy["genotype"] == genotype]
    early = subset[subset["timepoint"] == "early"].set_index("animal_id")[STATES]
    late = subset[subset["timepoint"] == "late"].set_index("animal_id")[STATES]
    change = late - early.loc[late.index]
    # Differences of equal size in frames can differ in the last digits; rounding makes them tie.
    p = [conditional_signed_rank_p(change[state].round(12)) for state in STATES]
    return pd.DataFrame({"genotype": genotype, "state": STATES, "change": change.mean().to_numpy(),
                         "p": p, "q": bh_fdr(p)})


def between_genotypes(occupancy: pd.DataFrame, timepoint: str) -> pd.DataFrame:
    subset = occupancy[occupancy["timepoint"] == timepoint]
    cko = subset[subset["genotype"] == "cKO"]
    wt = subset[subset["genotype"] == "WT"]
    p = [conditional_rank_sum_p(cko[state], wt[state]) for state in STATES]
    return pd.DataFrame({
        "timepoint": timepoint,
        "state": STATES,
        "difference": [cko[state].mean() - wt[state].mean() for state in STATES],
        "cliffs_delta": [cliffs_delta(cko[state], wt[state]) for state in STATES],
        "p": p,
        "q": bh_fdr(p),
    })


def main(cfg: dict) -> None:
    occupancy = load_occupancy(cfg)
    occupancy.groupby("group")[STATES].mean().reindex(GROUPS).T.rename_axis("state").reset_index().to_csv(
        source_data_path(cfg, "figS1m_occupancy.csv"), index=False
    )
    pd.concat([within_genotype(occupancy, g) for g in ("WT", "cKO")]).to_csv(
        source_data_path(cfg, "figS1o_p_within_genotype.csv"), index=False
    )
    pd.concat([between_genotypes(occupancy, t) for t in ("early", "late")]).to_csv(
        source_data_path(cfg, "figS1q_r_between_genotypes.csv"), index=False
    )


if __name__ == "__main__":
    main(parse_config(__doc__))
