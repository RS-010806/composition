# Sparse time-camera supervision in a small D4RT-style model

Mean 3D end-point error (scene units; spheres have radius 0.55 to 0.8) on held-out test scenes, for queries whose t_src, t_tgt and t_cam are all different. For the sparse designs these combinations were never supervised; the all-36 model saw every combination. Mean ± sd over seeds; acc = share of queries within 0.25 units.

| Supervision | Combinations per t_src | Seeds | Moving points | Static points | Moving, acc | Static, acc | Tracks in camera 0 (moving) | Seen combinations (moving / static) |
|---|---|---|---|---|---|---|---|---|
| All 36 | 36 | 3 | 0.754 ± 0.009 | 0.307 ± 0.009 | 6% | 54% | - | 0.681 / 0.286 |
| Row + column | 11 | 3 | 0.775 ± 0.010 | 0.319 ± 0.004 | 6% | 52% | 0.698 ± 0.011 | 0.533 / 0.241 |
| Random | 11 | 3 | 0.765 ± 0.050 | 0.307 ± 0.008 | 6% | 54% | 0.684 ± 0.065 | 0.677 / 0.278 |
| Tracks + depth | 11 | 3 | 0.762 ± 0.009 | 0.311 ± 0.006 | 6% | 53% | 0.686 ± 0.008 | 0.507 / 0.294 |
| Tracks only | 6 | 3 | 0.906 ± 0.024 | 0.642 ± 0.051 | 4% | 14% | 0.909 ± 0.051 | 0.623 / 0.288 |
| Depth only | 6 | 3 | 0.895 ± 0.006 | 0.650 ± 0.014 | 6% | 13% | 0.804 ± 0.016 | 0.324 / 0.292 |

Baseline that assumes nothing moves (the point where it was, in its own camera): moving 1.180, static 1.003.
