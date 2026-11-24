# Input data

The recordings are not distributed with this repository. They are available from the lead contact of the paper on
request. Place them as follows:

```
data/
  index.csv
  coordinates/
    <name>.csv
```

## `index.csv`

One row per recording, with two columns.

| Column | Content |
|---|---|
| `name` | Recording name, `<ko or wt>_s<cohort>d<day>_<mouse>`, for example `ko_s1d54_111`. |
| `group` | `WT_early`, `WT_late`, `cKO_early` or `cKO_late`. Rows with any other label are not analyzed. |

## `coordinates/<name>.csv`

3D keypoints of one recording at 20 frames per second, in cm, with the origin at the arena center. The file has no
header and one row per frame. Columns are x, y and z of each keypoint in this order: nose, head, tail base, torso, right
forelimb, left forelimb, right hindlimb, left hindlimb, tail.
