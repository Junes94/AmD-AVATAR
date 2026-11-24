# AVATAR behavioral analysis

Code for the three-dimensional behavioral analysis in Lee et al. (in submission). Male wild-type and
conditional knockout mice were recorded for 10 min in an open field with the AVATAR system at an early and
a late timepoint. The code computes kinematic measures from 3D keypoints, normalizes them to body size, defines six
posture states and runs the statistical tests of the paper. It writes the values of each figure panel as tables and
does not draw figures.

## Quick start

Python 3.13 (tested on Windows 11). Run every command from the repository root.

```
pip install -r requirements.txt
python scripts/run_all.py
```

The inputs are described in [`data/README.md`](data/README.md). All settings are in `configs/config.yaml`; another
config can be passed with `--config`. Each script can also be run on its own, in the order listed in
`scripts/run_all.py`.

## Layout

| Folder | Content |
|---|---|
| `configs/` | Analysis parameters and input locations. |
| `data/` | Inputs (not distributed, see `data/README.md`). |
| `src/` | Shared code: session metadata, kinematics, posture states, statistics. |
| `scripts/` | One script per analysis step or figure panel, and `run_all.py`. |
| `results/` | Everything the scripts write (not distributed). |

## Analyses

| Step | Script | Output in `results/` |
|---|---|---|
| Kinematic measures per frame | `compute_scalars.py` | `frame_scalars.csv` |
| Body-size normalization | `normalize_scalars.py` | `frame_scalars_normalized.csv`, `body_size.csv` |
| Posture states | `define_posture_states.py` | `posture_cuts.csv`, `posture_occupancy.csv` |
| Fig. 1I | `fig1i_scalar_change.py` | `source_data/fig1i_scalar_change.csv` |
| Fig. 1J | `fig1j_posture_pca.py` | `source_data/fig1j_pca_*.csv` |
| Fig. 1K | `fig1k_posture_shift.py` | `source_data/fig1k_*.csv` |
| Fig. S1H-L | `figS1h_l_scalar_means.py` | `source_data/figS1h_l_*.csv` |
| Fig. S1M, O-R | `figS1m_o_r_posture_states.py` | `source_data/figS1m_occupancy.csv`, `figS1o_p_within_genotype.csv`, `figS1q_r_between_genotypes.csv` |
| Fig. S1N (frame selection) | `figS1n_representative_frames.py` | `source_data/figS1n_selected_frames.csv` |

## Methods in brief

- **Kinematics.** Keypoint trajectories are smoothed with a centered 5-frame moving average. Length is the horizontal
  head to tail-base distance and height the signed vertical difference between them. Horizontal and vertical speeds are
  computed from the midpoint of head and torso. Position is the horizontal distance of the torso from the arena center.
- **Normalization.** Length and height are divided by the 99.5th percentile of the 3D head to tail-base distance in the
  early session of the same animal.
- **Posture states.** Three length bins (WT-early tertiles) are crossed with two height bins (antimode of the WT-early
  density). Occupancy vectors are centered-log-ratio transformed, with zeros replaced by half the smallest positive
  value.
- **Statistics.** Scalar measures and per-animal distances use exact Mann-Whitney U tests. Posture-state occupancy
  uses conditional exact tests that keep ties. Genotypes are compared with rank-sum tests that count every label
  assignment preserving the two group sizes. Early-to-late changes are tested with Wilcoxon signed-rank tests that
  drop zero differences and count every sign assignment. Equal values receive average ranks, and two-sided p values are
  twice the smaller inclusive tail probability, capped at one. Cliff's delta CIs use a percentile bootstrap over
  animals (2,000 resamples). Sign-flip and label permutation tests on posture-change
  vectors use 10,000 random permutations and p = (b + 1) / (m + 1). Benjamini-Hochberg correction is applied within each
  family. Random seeds are set in the config.

The statistical checks can be run with `python -m unittest discover -s tests`.

## Steps outside this repository

Keypoint detection with AVATARnet in five camera views and triangulation in MATLAB R2022a, as described in the Methods
of the paper.

## License

MIT (see `LICENSE`).
