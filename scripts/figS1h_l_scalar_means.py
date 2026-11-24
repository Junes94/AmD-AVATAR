"""Fig. S1H-L: group means of each kinematic scalar at the early and late timepoints."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import parse_config, source_data_path  # noqa: E402
from src.kinematics import SCALARS  # noqa: E402
from src.sessions import load_session_means  # noqa: E402
from src.stats import bh_fdr, cliffs_delta, cliffs_delta_ci  # noqa: E402


def main(cfg: dict) -> None:
    stats = cfg["statistics"]
    means = load_session_means(cfg)
    rows = []
    for scalar in SCALARS:
        pair = []
        for timepoint in ("early", "late"):
            subset = means[means["timepoint"] == timepoint]
            wt = subset.loc[subset["genotype"] == "WT", scalar].to_numpy()
            cko = subset.loc[subset["genotype"] == "cKO", scalar].to_numpy()
            ci_low, ci_high = cliffs_delta_ci(cko, wt, stats["n_bootstrap"], stats["bootstrap_seed"])
            pair.append({
                "scalar": scalar,
                "timepoint": timepoint,
                "WT_mean": wt.mean(),
                "WT_sem": wt.std(ddof=1) / len(wt) ** 0.5,
                "n_WT": len(wt),
                "cKO_mean": cko.mean(),
                "cKO_sem": cko.std(ddof=1) / len(cko) ** 0.5,
                "n_cKO": len(cko),
                "cliffs_delta": cliffs_delta(cko, wt),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p": mannwhitneyu(cko, wt, alternative="two-sided", method="exact").pvalue,
            })
        for row, q in zip(pair, bh_fdr([row["p"] for row in pair])):
            row["q"] = q
        rows.extend(pair)

    pd.DataFrame(rows).to_csv(source_data_path(cfg, "figS1h_l_summary.csv"), index=False)
    means[["name", "animal_id", "genotype", "timepoint", "group", *SCALARS]].to_csv(
        source_data_path(cfg, "figS1h_l_per_animal.csv"), index=False
    )


if __name__ == "__main__":
    main(parse_config(__doc__))
