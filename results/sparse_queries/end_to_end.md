# Sparse time-camera supervision in a small D4RT-style model

Mean 3D end-point error (scene units; spheres have radius 0.55 to 0.8) on held-out test scenes, for queries whose t_src, t_tgt and t_cam are all different. For the sparse designs these combinations were never supervised; the all-36 model saw every combination. Mean ± sd over seeds; acc = share of queries within 0.25 units.

| Supervision | Combinations per t_src | Seeds | Moving points | Static points | Moving, acc | Static, acc | Tracks in camera 0 (moving) | Seen combinations (moving / static) |
|---|---|---|---|---|---|---|---|---|
| All 36 | 36 | 3 | 0.710 ± 0.007 | 0.242 ± 0.003 | 9% | 67% | - | 0.637 / 0.220 |
| Row + column | 11 | 3 | 0.745 ± 0.008 | 0.230 ± 0.018 | 7% | 70% | 0.665 ± 0.007 | 0.491 / 0.156 |
| Tracks + depth | 11 | 3 | 0.734 ± 0.007 | 0.222 ± 0.009 | 7% | 71% | 0.652 ± 0.004 | 0.469 / 0.205 |
| Tracks only | 6 | 3 | 0.851 ± 0.014 | 0.617 ± 0.021 | 5% | 15% | 0.876 ± 0.030 | 0.553 / 0.254 |

Baseline that assumes nothing moves (the point where it was, in its own camera): moving 1.180, static 1.003.
