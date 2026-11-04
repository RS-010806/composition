# Sparse training queries in a D4RT-style model

D4RT ([Zhang et al., CVPR 2026](https://arxiv.org/abs/2512.08924)) answers queries
q = (u, v, t_src, t_tgt, t_cam): where the point seen at pixel (u, v) of frame t_src is at time
t_tgt, in the coordinates of the camera of frame t_cam. Its training draws t_tgt and t_cam
independently (and sets them equal 40% of the time), so every time-camera combination is supervised.

Real labels are often narrower. Camera-frame 3D tracks, from 2D tracks plus per-frame depth, only give
queries with t_tgt = t_cam. Posed depth only gives t_tgt = t_src. This folder asks whether a D4RT-style
model answers the combinations it was never trained on, with the same test as the color-shape grid in
the main README: supervise one row and one column of the (t_tgt, t_cam) grid, or only its diagonal,
and score the rest.

## Results

Errors are 3D distances in scene units (the spheres have radius 0.55 to 0.8) on 100 new scenes, for combinations with
t_src, t_tgt and t_cam all different, which none of the sparse label sets contains. Static points give the cleanest
readout: their position depends only on the camera, and the model learns camera motion well. It tracks moving objects
only partly (see the caveats), so moving points are noisier but show the same ordering.

**End to end** (1 seeds per label set; each model trained from scratch with the same number of training queries).
Differences to the all-36 model are computed per test scene after averaging over seeds, with 95% bootstrap intervals over
the 100 scenes and two-sided Wilcoxon signed-rank p-values (Holm-corrected across the six comparisons).

| Supervised combinations | Cells per t_src | Static points | Difference to all 36 | Moving points | Difference to all 36 |
|---|---|---|---|---|---|
| All 36 (D4RT's sampling) | 36 | 0.24 ± 0.00 | reference | 0.71 ± 0.00 | reference |
| One row + one column | 11 | 0.21 ± 0.00 | -0.03 [-0.04, -0.02], p 2e-05 | 0.74 ± 0.00 | +0.03 [+0.02, +0.05], p 4e-04 |
| Camera-frame tracks + posed depth | 11 | 0.22 ± 0.00 | -0.02 [-0.04, -0.01], p 9e-04 | 0.73 ± 0.00 | +0.02 [+0.01, +0.03], p 0.02 |
| Camera-frame tracks only | 6 | 0.60 ± 0.00 | +0.35 [+0.32, +0.38], p < 1e-15 | 0.83 ± 0.00 | +0.13 [+0.10, +0.16], p 8e-11 |
| Assume nothing moves | - | 1.00 | | 1.18 | |

![Static-point error maps, end to end](../results/sparse_queries/end_to_end_error_maps_static.png)

**Post-training.** A fresh decoder on the frozen encoder of the seed-0 all-36 model, trained with each label set
(2,500 steps; mean ± sd over seeds; p from Welch's t-test on the seed means against the all-36 decoder, Holm-corrected).

| Supervised combinations | Seeds | Static points | Versus all 36 | Moving points |
|---|---|---|---|---|
| All 36 (D4RT's sampling) | 2 | 0.31 ± 0.01 | reference | 0.76 ± 0.01 |
| One row + one column | 2 | 0.32 ± 0.00 | p 1.00 | 0.78 ± 0.01 |
| Random | 2 | 0.31 ± 0.01 | p 1.00 | 0.77 ± 0.07 |
| Camera-frame tracks + posed depth | 2 | 0.31 ± 0.01 | p 1.00 | 0.76 ± 0.01 |
| Camera-frame tracks only | 2 | 0.61 ± 0.02 | p 0.16 | 0.89 ± 0.00 |
| Posed depth only | 2 | 0.65 ± 0.02 | p 0.09 | 0.89 ± 0.01 |

![Static-point error maps, decoder only](../results/sparse_queries/decoder_only_error_maps_static.png)

Every label set in which the time indices vary independently somewhere matches all 36, random cells included. The two
that keep a pair of indices tied fail in different ways. With camera-frame tracks only (t_tgt = t_cam) the error grows
away from the diagonal: nothing in the labels says which index sets the camera. With posed depth only (t_tgt = t_src)
it grows with the target time: the model never saw time change, so it moves static points with it.

For static points the right answer is the point in camera t_cam. The end-to-end tracks-only model (seed 0) lands between that and the same point in camera t_tgt: closer to the truth in 57% of queries and to the camera-t_tgt answer in 43% ([readings_tied.json](../results/sparse_queries/readings_tied.json)).

All statistics: [stats.md](../results/sparse_queries/stats.md).

## Setup

- **Scenes** (`scenes.py`). 6 frames of 32 x 32 video: a camera at fixed height and tilt that slides and pans over a
  tiled floor, 3 static spheres and 2 moving ones. Every pixel keeps its object id and 3D point, so any query has
  exact ground truth.
- **Model** (`d4rt_small.py`). D4RT's interface at small scale. Encoder: a 3x3 convolution over each frame and its
  difference from the previous frame (D4RT's tubelets span two frames), 8 x 8 patches, and a 3-layer transformer
  over all frames. Decoder: the query is the sum of a Fourier embedding of (u, v), learned embeddings of t_src, t_tgt
  and t_cam, and an embedding of the 9 x 9 patch around (u, v); 3 layers of cross-attention to the encoder tokens,
  with no attention between queries. Loss: L1 on sign(x) log(1 + |x|) of the 3D point.
- **Label sets.** For every source frame, training queries are restricted to a subset of the 6 x 6 (t_tgt, t_cam)
  grid. All label sets get the same number of training queries (8 clips x 128 per step, half on moving objects).

| Label set | Cells per t_src | Supervised cells | Real labels that give it |
|---|---|---|---|
| All 36 | 36 | every (t_tgt, t_cam); t_tgt = t_cam 40% of the time | D4RT's sampling |
| One row + one column | 11 | t_tgt = t_src, or t_cam = t_src | posed depth plus tracks in the source camera |
| Random | 11 | 11 cells drawn at random for each t_src | |
| Camera-frame tracks + posed depth | 11 | t_tgt = t_cam, or t_tgt = t_src | 2D tracks and per-frame depth, plus posed depth |
| Camera-frame tracks only | 6 | t_tgt = t_cam | 2D tracks and per-frame depth, no camera poses |
| Posed depth only | 6 | t_tgt = t_src | depth maps with camera poses |

- **Test.** 100 new scenes; for every frame, 6 points on moving objects and 6 static points, each queried at all 36
  combinations. The headline numbers use the combinations with t_src, t_tgt and t_cam all different, which no sparse
  label set supervises.

## Files

| File | What it does |
|---|---|
| `scenes.py` | Procedural clips and exact ground truth for any query |
| `d4rt_small.py` | The model, the supervision designs, end-to-end training and evaluation |
| `decoder_only.py` | Fresh decoders on a frozen encoder, one per label set and seed (post-training version) |
| `stats.py` | Seed spread, per-scene paired differences with bootstrap intervals, Wilcoxon and Welch tests |
| `readings.py` | Where the tracks-only model's answers land between the readings its labels cannot tell apart |
| `reeval.py` | Re-scores saved models (adds per-scene errors) |
| `analyze.py` | Tables and figures in `../results/sparse_queries/` |
| `runs/`, `runs_dec/` | Per-run results (`*_eval.json`, `*_train.json`, supervised-cell masks) |
| `report/` | Two-page summary (PDF) and its source |

## Reproduce

```bash
pip install -r ../requirements.txt
for s in 0 1 2; do for d in full rowcol tied tiedrow; do
  python d4rt_small.py --design $d --seed $s --steps 5000 --out runs
done; done
python decoder_only.py --encoder runs/full_s0 --designs full rowcol random tiedrow tied row --seeds 0 1 2 --steps 2500 --out runs_dec
python readings.py --run runs/tied_s0 --out ../results/sparse_queries/readings_tied.json
python analyze.py --runs runs --name end_to_end
python analyze.py --runs runs_dec --name decoder_only
python stats.py
```

On 2 CPU cores an end-to-end run takes about 11 minutes and a decoder-only run about 2. Model weights are not
committed; every run can be retrained from its seed.
