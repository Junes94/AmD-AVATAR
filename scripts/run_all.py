"""Run every analysis step in order."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import parse_config  # noqa: E402

STEPS = [
    "compute_scalars",
    "normalize_scalars",
    "define_posture_states",
    "fig1i_scalar_change",
    "fig1j_posture_pca",
    "fig1k_posture_shift",
    "figS1h_l_scalar_means",
    "figS1m_o_r_posture_states",
    "figS1n_representative_frames",
]


def main() -> None:
    cfg = parse_config(__doc__)
    for step in STEPS:
        print(step, flush=True)
        importlib.import_module(step).main(cfg)


if __name__ == "__main__":
    main()
