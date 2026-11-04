# Sparse time-camera supervision in a small D4RT-style model

Mean 3D end-point error (scene units; spheres have radius 0.55 to 0.8) on held-out test scenes, for queries whose t_src, t_tgt and t_cam are all different. For the sparse designs these combinations were never supervised; the all-36 model saw every combination. Mean ± sd over seeds; acc = share of queries within 0.25 units.

| Supervision | Combinations per t_src | Seeds | Moving points | Static points | Moving, acc | Static, acc | Tracks in camera 0 (moving) | Seen combinations (moving / static) |
|---|---|---|---|---|---|---|---|---|
| All 36 | 36 | 1 | 0.708 ± 0.000 | 0.243 ± 0.000 | 9% | 67% | - | 0.636 / 0.221 |
| Row + column | 11 | 1 | 0.741 ± 0.000 | 0.211 ± 0.000 | 7% | 74% | 0.662 ± 0.000 | 0.484 / 0.145 |
| Tracks + depth | 11 | 1 | 0.729 ± 0.000 | 0.218 ± 0.000 | 7% | 72% | 0.649 ± 0.000 | 0.462 / 0.201 |
| Tracks only | 6 | 1 | 0.835 ± 0.000 | 0.596 ± 0.000 | 6% | 16% | 0.841 ± 0.000 | 0.550 / 0.246 |

Baseline that assumes nothing moves (the point where it was, in its own camera): moving 1.180, static 1.003.
