"""Fig. 1I: early-to-late change of each kinematic scalar, cKO versus WT."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import parse_config, source_data_path  # noqa: E402
from src.kinematics import SCALARS  # noqa: E402
from src.sessions import load_session_means  # noqa: E402
from src.stats import cliffs_delta, cliffs_delta_ci  # noqa: E402


def within_animal_change(means: pd.DataFrame, scalar: str, genotype: str):
    """Late minus early value per animal, ordered by animal."""
    table = means[means["genotype"] == genotype].pivot(index="animal_id", columns="timepoint", values=scalar)
    return (table["late"] - table["early"]).to_numpy()


def main(cfg: dict) -> None:
    stats = cfg["statistics"]
    means = load_session_means(cfg)
    rows = []
    for scalar in SCALARS:
        cko = within_animal_change(means, scalar, "cKO")
        wt = within_animal_change(means, scalar, "WT")
        ci_low, ci_high = cliffs_delta_ci(cko, wt, stats["n_bootstrap"], stats["bootstrap_seed"])
        rows.append({
            "scalar": scalar,
            "change_cKO": cko.mean(),
            "change_WT": wt.mean(),
            "cliffs_delta": cliffs_delta(cko, wt),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_cKO": len(cko),
            "n_WT": len(wt),
            "p": mannwhitneyu(cko, wt, alternative="two-sided", method="exact").pvalue,
        })
    pd.DataFrame(rows).to_csv(source_data_path(cfg, "fig1i_scalar_change.csv"), index=False)


if __name__ == "__main__":
    main(parse_config(__doc__))
