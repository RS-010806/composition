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
| rowcol | static | 3 | 0.230 ± 0.018 | -0.012 [-0.031, +0.007] | 0.95 [0.87, 1.03] | 0.362 | -0.012 [-0.019, -0.005] | 43/100 | 0.005 |
| rowcol | moving | 3 | 0.745 ± 0.008 | +0.035 [+0.018, +0.052] | 1.05 [1.03, 1.07] | 0.018 | +0.035 [+0.022, +0.048] | 67/100 | 4.4e-05 |
| tiedrow | static | 3 | 0.222 ± 0.009 | -0.019 [-0.033, -0.005] | 0.92 [0.87, 0.98] | 0.118 | -0.019 [-0.027, -0.011] | 34/100 | 4.6e-05 |
| tiedrow | moving | 3 | 0.734 ± 0.007 | +0.023 [+0.008, +0.039] | 1.03 [1.01, 1.06] | 0.039 | +0.023 [+0.012, +0.035] | 67/100 | 0.001 |
| tied | static | 3 | 0.617 ± 0.021 | +0.372 [+0.337, +0.410] | 2.54 [2.37, 2.73] | 0.005 | +0.372 [+0.342, +0.404] | 100/100 | <1e-15 |
| tied | moving | 3 | 0.851 ± 0.014 | +0.141 [+0.108, +0.174] | 1.20 [1.15, 1.25] | 0.005 | +0.141 [+0.114, +0.169] | 86/100 | 1.3e-12 |

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

