# Sparse time-camera supervision in a small D4RT-style model

Mean 3D end-point error (scene units; spheres have radius 0.55 to 0.8) on held-out test scenes, for queries whose t_src, t_tgt and t_cam are all different. For the sparse designs these combinations were never supervised; the all-36 model saw every combination. Mean ± sd over seeds; acc = share of queries within 0.25 units.

| Supervision | Combinations per t_src | Seeds | Moving points | Static points | Moving, acc | Static, acc | Tracks in camera 0 (moving) | Seen combinations (moving / static) |
|---|---|---|---|---|---|---|---|---|
| All 36 | 36 | 2 | 0.756 ± 0.012 | 0.311 ± 0.008 | 6% | 53% | - | 0.682 / 0.291 |
| Row + column | 11 | 2 | 0.779 ± 0.009 | 0.320 ± 0.005 | 6% | 51% | 0.702 ± 0.012 | 0.538 / 0.244 |
| Random | 11 | 2 | 0.771 ± 0.069 | 0.309 ± 0.011 | 6% | 53% | 0.681 ± 0.092 | 0.673 / 0.276 |
| Tracks + depth | 11 | 2 | 0.765 ± 0.010 | 0.313 ± 0.007 | 6% | 52% | 0.689 ± 0.010 | 0.510 / 0.297 |
| Tracks only | 6 | 2 | 0.892 ± 0.005 | 0.615 ± 0.023 | 5% | 15% | 0.884 ± 0.040 | 0.629 / 0.288 |
| Depth only | 6 | 2 | 0.895 ± 0.008 | 0.653 ± 0.019 | 6% | 13% | 0.814 ± 0.002 | 0.328 / 0.294 |

Baseline that assumes nothing moves (the point where it was, in its own camera): moving 1.180, static 1.003.
