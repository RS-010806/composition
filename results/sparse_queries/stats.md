# Statistics

3D error on held-out combinations (t_src, t_tgt, t_cam all different), 100 test scenes. Mean ± sd over seeds.
Difference to all 36: per scene, errors averaged over seeds, then mean difference with a 95% bootstrap interval
over scenes and a two-sided Wilcoxon signed-rank test. Welch p: t-test on the seed means.
Both p columns are Holm-corrected across all comparisons with the all-36 model in the same table.

## end to end

| Label set | Points | Seeds | Error (mean ± sd) | Difference to all 36 [95% CI] | Ratio [95% CI] | Scenes worse | Wilcoxon p | Welch p (seeds) |
|---|---|---|---|---|---|---|---|---|
| full | static | 1 | 0.243 ± 0.000 | - | - | - | - | - |
| full | moving | 1 | 0.708 ± 0.000 | - | - | - | - | - |
| rowcol | static | 1 | 0.211 ± 0.000 | -0.031 [-0.044, -0.020] | 0.87 [0.83, 0.92] | 32/100 | 1.7e-05 | - |
| rowcol | moving | 1 | 0.741 ± 0.000 | +0.032 [+0.017, +0.047] | 1.05 [1.02, 1.07] | 67/100 | 3.7e-04 | - |
| tiedrow | static | 1 | 0.218 ± 0.000 | -0.024 [-0.038, -0.010] | 0.90 [0.85, 0.96] | 29/100 | 8.9e-04 | - |
| tiedrow | moving | 1 | 0.729 ± 0.000 | +0.020 [+0.007, +0.033] | 1.03 [1.01, 1.05] | 58/100 | 0.021 | - |
| tied | static | 1 | 0.596 ± 0.000 | +0.351 [+0.323, +0.379] | 2.45 [2.29, 2.61] | 100/100 | <1e-15 | - |
| tied | moving | 1 | 0.835 ± 0.000 | +0.127 [+0.099, +0.156] | 1.18 [1.14, 1.23] | 83/100 | 7.8e-11 | - |

## decoder only

| Label set | Points | Seeds | Error (mean ± sd) | Difference to all 36 [95% CI] | Ratio [95% CI] | Scenes worse | Wilcoxon p | Welch p (seeds) |
|---|---|---|---|---|---|---|---|---|
| full | static | 2 | 0.311 ± 0.008 | - | - | - | - | - |
| full | moving | 2 | 0.756 ± 0.012 | - | - | - | - | - |
| rowcol | static | 2 | 0.320 ± 0.005 | - | - | - | - | 1.000 |
| rowcol | moving | 2 | 0.779 ± 0.009 | - | - | - | - | 0.988 |
| random | static | 2 | 0.309 ± 0.011 | - | - | - | - | 1.000 |
| random | moving | 2 | 0.771 ± 0.069 | - | - | - | - | 1.000 |
| tiedrow | static | 2 | 0.313 ± 0.007 | - | - | - | - | 1.000 |
| tiedrow | moving | 2 | 0.765 ± 0.010 | - | - | - | - | 1.000 |
| tied | static | 2 | 0.615 ± 0.023 | - | - | - | - | 0.157 |
| tied | moving | 2 | 0.892 ± 0.005 | - | - | - | - | 0.157 |
| row | static | 2 | 0.653 ± 0.019 | - | - | - | - | 0.092 |
| row | moving | 2 | 0.895 ± 0.008 | - | - | - | - | 0.087 |

