# Statistics

3D error on held-out combinations (t_src, t_tgt, t_cam all different), 100 test scenes. Mean ± sd over seeds.

- Difference and ratio to all 36, seeds and scenes: 95% interval from a two-level bootstrap that resamples the
  test scenes and each label set's training seeds. This is the headline interval.
- Welch p: t-test on the seed means (does the difference replicate across training runs?).
- Scene-level columns condition on the trained models: per scene, errors averaged over seeds, then a 95% bootstrap
  interval over scenes and a two-sided Wilcoxon signed-rank test.
- Both p columns are Holm-corrected across all comparisons with the all-36 model in the same table.

## end to end

| Label set | Points | Seeds | Error (mean ± sd) | Difference, seeds and scenes [95% CI] | Ratio, seeds and scenes [95% CI] | Welch p | Difference, scenes only [95% CI] | Scenes worse | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|---|
| full | static | 3 | 0.242 ± 0.003 | - | - | - | - | - | - |
| full | moving | 3 | 0.710 ± 0.007 | - | - | - | - | - | - |
| rowcol | static | 3 | 0.230 ± 0.018 | -0.012 [-0.031, +0.007] | 0.95 [0.87, 1.03] | 0.724 | -0.012 [-0.019, -0.005] | 43/100 | 0.010 |
| rowcol | moving | 3 | 0.745 ± 0.008 | +0.035 [+0.018, +0.052] | 1.05 [1.03, 1.07] | 0.027 | +0.035 [+0.022, +0.048] | 67/100 | 6.6e-05 |
| tiedrow | static | 3 | 0.222 ± 0.009 | -0.019 [-0.033, -0.005] | 0.92 [0.87, 0.98] | 0.177 | -0.019 [-0.027, -0.011] | 34/100 | 7.7e-05 |
| tiedrow | moving | 3 | 0.734 ± 0.007 | +0.023 [+0.008, +0.039] | 1.03 [1.01, 1.06] | 0.052 | +0.023 [+0.012, +0.035] | 67/100 | 0.002 |
| split | static | 3 | 0.247 ± 0.008 | +0.005 [-0.007, +0.018] | 1.02 [0.97, 1.08] | 0.724 | +0.005 [-0.002, +0.012] | 60/100 | 0.112 |
| split | moving | 3 | 0.735 ± 0.005 | +0.024 [+0.010, +0.039] | 1.03 [1.01, 1.06] | 0.041 | +0.024 [+0.013, +0.035] | 65/100 | 8.0e-04 |
| tied | static | 3 | 0.617 ± 0.021 | +0.372 [+0.337, +0.410] | 2.54 [2.37, 2.73] | 0.006 | +0.372 [+0.342, +0.404] | 100/100 | <1e-15 |
| tied | moving | 3 | 0.851 ± 0.014 | +0.141 [+0.108, +0.174] | 1.20 [1.15, 1.25] | 0.006 | +0.141 [+0.114, +0.169] | 86/100 | 1.8e-12 |

## decoder only

| Label set | Points | Seeds | Error (mean ± sd) | Difference to all 36 | Welch p |
|---|---|---|---|---|---|
| full | static | 3 | 0.307 ± 0.009 | reference | - |
| full | moving | 3 | 0.754 ± 0.009 | reference | - |
| rowcol | static | 3 | 0.319 ± 0.004 | +0.012 | 0.702 |
| rowcol | moving | 3 | 0.775 ± 0.010 | +0.021 | 0.333 |
| random | static | 3 | 0.307 ± 0.008 | +0.000 | 1.000 |
| random | moving | 3 | 0.765 ± 0.050 | +0.011 | 1.000 |
| tiedrow | static | 3 | 0.311 ± 0.006 | +0.005 | 1.000 |
| tiedrow | moving | 3 | 0.762 ± 0.009 | +0.008 | 1.000 |
| tied | static | 3 | 0.642 ± 0.051 | +0.336 | 0.043 |
| tied | moving | 3 | 0.906 ± 0.024 | +0.151 | 0.031 |
| row | static | 3 | 0.650 ± 0.014 | +0.343 | 1.5e-04 |
| row | moving | 3 | 0.895 ± 0.006 | +0.141 | 6.0e-04 |

## Same clips or different clips

Camera-frame tracks and posed depth on different clips (split) against both on every clip (tiedrow), end to end.

| Points | Split | Same clips | Difference, seeds and scenes [95% CI] | Ratio [95% CI] | Welch p | Wilcoxon p |
|---|---|---|---|---|---|---|
| static | 0.247 ± 0.008 | 0.222 ± 0.009 | +0.024 [+0.009, +0.039] | 1.11 [1.04, 1.18] | 0.053 | 1.0e-07 |
| moving | 0.735 ± 0.005 | 0.734 ± 0.007 | +0.001 [-0.009, +0.011] | 1.00 [0.99, 1.02] | 0.866 | 0.842 |

