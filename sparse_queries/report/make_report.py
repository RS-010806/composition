"""Fill the report template with numbers from ../../results/sparse_queries and write report.html.

python make_report.py && node pdf.js "$PWD/report.html" sparse_queries_report.pdf
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
RES = HERE.parent.parent / "results" / "sparse_queries"
E = {r["design"]: r for r in json.load(open(RES / "end_to_end.json"))}
S = json.load(open(RES / "stats.json"))
SE, SD = S["end_to_end"], S["decoder_only"]
DR = json.load(open(RES / "drift.json"))
DRD = json.load(open(RES / "drift_decoder_only.json")) if (RES / "drift_decoder_only.json").exists() else {}
T = 6
N2, K = T * T, 2 * T - 1


def f2(x):
    return f"{x:.2f}"


def fp(p):
    """p-value for running text; non-breaking spaces keep it on one line."""
    if p < 1e-15:
        return "p&nbsp;&lt;&nbsp;10<sup>&minus;15</sup>"
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return f"p&nbsp;=&nbsp;{m}&nbsp;&times;&nbsp;10<sup>&minus;{abs(int(e))}</sup>"
    return f"p&nbsp;=&nbsp;{p:.3f}" if p < 0.01 else f"p&nbsp;=&nbsp;{p:.2f}"



def m(d, w="static", src=SE):
    return src[d][w]["mean"]


base_m, base_s = E["full"]["base_mov"][0], E["full"]["base_sta"][0]
ns_e = SE["full"]["static"]["seeds"]
ns_d = SD["full"]["static"]["seeds"]
tied = SE["tied"]["static"]
ROW_DRIFT = (f"; it moves a static point by {DRD['row']['mean']:.2f} units across target times, against {DRD['full']['mean']:.2f} for the all-{N2} decoder"
             if all(len(DRD.get(d, {}).get("per_seed", {})) == ns_d for d in ("row", "full")) else "")
every = (f"on every one of the {tied['scenes']} test scenes" if tied["scenes_worse"] == tied["scenes"]
         else f"on {tied['scenes_worse']} of {tied['scenes']} test scenes")

DEK = (f"Trained on one row and one column of the {T} &times; {T} time-camera grid ({K} of {N2} combinations), a small D4RT-style "
       f"model answered the combinations it never saw as well as one trained on all {N2}. Trained only where time and camera "
       f"coincide, which is all that camera-frame tracks provide, it confused the two; one row of posed depth fixed that, even "
       f"when it came from other clips.")

TILES = f"""
<div class="tile"><div class="num">{K} <small>of {N2}</small></div>
<div class="lab">combinations, one row and one column: error <b>{f2(m('rowcol'))}</b> on held-out static points,
against <b>{f2(m('full'))}</b> when all {N2} are supervised.</div></div>
<div class="tile bad"><div class="num">{tied['ratio']:.1f}&times; <small>the error</small></div>
<div class="lab">when only t<sub>tgt</sub> = t<sub>cam</sub> is supervised, as with camera-frame tracks: worse on
<b>{tied['scenes_worse']} of {tied['scenes']}</b> test scenes{" and in every seed" if tied['every_seed_worse'] else ""}.</div></div>
<div class="tile fix"><div class="num">+1 <small>row</small></div>
<div class="lab">of posed depth (t<sub>tgt</sub> = t<sub>src</sub>) added to those tracks brings the error back to <b>{f2(m('tiedrow'))}</b>,
and to <b>{f2(m('split'))}</b> when it comes from other clips.</div></div>
"""

BODY1 = f"""
<h2><span class="n">01</span>The question</h2>
<p>D4RT answers queries <span class="q">(u, v, t<sub>src</sub>, t<sub>tgt</sub>, t<sub>cam</sub>)</span>: where the point seen at pixel (u, v)
of frame t<sub>src</sub> is at time t<sub>tgt</sub>, in the coordinates of the camera of frame t<sub>cam</sub>. Its training draws t<sub>tgt</sub>
and t<sub>cam</sub> independently and sets them equal 40% of the time, so every time-camera combination is supervised. Real labels are often
narrower. Camera-frame 3D tracks (2D tracks plus per-frame depth) only give queries with <b>t<sub>tgt</sub> = t<sub>cam</sub></b>; posed depth
only gives <b>t<sub>tgt</sub> = t<sub>src</sub></b>. Does the model answer combinations it was never trained on? The color-shape test suggests
it does, as long as time and camera vary independently somewhere in the labels.</p>

<img src="../../results/sparse_queries/end_to_end_error_maps_static.png" alt="Error maps on static points for each label set">
<div class="cap"><b>Figure 1.</b> Error on static points by target time (rows) and camera (columns), for points first seen in frame 0 of 100 new
scenes; mean of {ns_e} models per panel, each trained from scratch with the same number of training queries. Outlined cells were supervised; the
rest were never seen in training. A static point's position depends only on the camera, so a model that keeps time and camera apart shows
errors that change across columns but not down them. Numbers below the panels: error on all held-out combinations and source frames.</div>

<h2><span class="n">02</span>What I ran</h2>
<p>A D4RT-style model at small scale, with D4RT's query interface: an encoder over all {T} frames of 32 &times; 32 video (a 3 &times; 3 convolution
over each frame and its difference from the previous one, 8 &times; 8 patches, a 3-layer transformer), and a decoder in which each query sums a
Fourier embedding of (u, v), learned embeddings of t<sub>src</sub>, t<sub>tgt</sub> and t<sub>cam</sub>, and an embedding of the 9 &times; 9 patch
around (u, v), then cross-attends to the encoder tokens, never to other queries. The clips are synthetic so that every query has exact ground
truth: a camera slides and pans over a tiled floor with 3 static and 2 moving spheres.</p>
<p>Each label set restricts which (t<sub>tgt</sub>, t<sub>cam</sub>) cells are supervised for every source frame; all get the same number of
training queries (5,000 steps of 8 clips &times; 128 queries, half on moving objects) and {ns_e} seeds. Errors are 3D distances in scene units
(the spheres have radius 0.55 to 0.8) on 100 new scenes, for combinations with t<sub>src</sub>, t<sub>tgt</sub> and t<sub>cam</sub> all different,
which none of the sparse label sets contains.</p>
"""


def row(d, label):
    s, mo = SE[d]["static"], SE[d]["moving"]
    rt = lambda c: "reference" if d == "full" else f"{c['ratio']:.2f}&times; [{c['ratio_ci_2level'][0]:.2f}, {c['ratio_ci_2level'][1]:.2f}]"
    cells = f"{T} per clip" if d == "split" else E[d]["cells"]
    return (f"<tr><td>{label}</td><td class='num'>{cells}</td><td class='num'><b>{s['mean']:.3f}</b> &plusmn; {s['sd']:.3f}</td>"
            f"<td class='num'>{rt(s)}</td><td class='num'>{mo['mean']:.3f} &plusmn; {mo['sd']:.3f}</td><td class='num'>{rt(mo)}</td></tr>")


LABEL = {"full": f"All {N2} (D4RT's sampling)", "rowcol": "One row + one column",
         "tiedrow": "Camera-frame tracks + posed depth", "split": "Tracks + posed depth, on different clips",
         "tied": "Camera-frame tracks only"}
E2E = [d for d in ["full", "rowcol", "tiedrow", "split", "tied"] if d in SE]
INDEP = [d for d in E2E if d != "tied"]
sp = SE["split"]["static"]
lo, hi = sp["ratio_ci_2level"]
SPLIT_VS_FULL = ("no worse than with all " + str(N2)) if hi < 1.10 and lo <= 1.0 else ("close to all " + str(N2))
sv = S["split_vs_tiedrow"]["static"]
SPLIT_VS_SAME = (f"slightly better ({f2(m('tiedrow'))}; {fp(sv['welch_p_seeds_holm'])})" if sv["welch_p_seeds_holm"] < 0.05
                 else f"not clearly better ({f2(m('tiedrow'))}; {fp(sv['welch_p_seeds_holm'])})")
TABLE = "".join(row(d, LABEL[d]) for d in E2E)
rc_m, tr_m, ti_m = SE["rowcol"]["moving"], SE["tiedrow"]["moving"], SE["tied"]["moving"]

BODY2 = f"""
<h2 style="margin-top:0"><span class="n">03</span>What I got</h2>
<table>
<tr><th>Supervised combinations</th><th class="num">Cells</th><th class="num">Static points</th><th class="num">Ratio to all {N2} [95% CI]</th><th class="num">Moving points</th><th class="num">Ratio to all {N2} [95% CI]</th></tr>
{TABLE}
<tr class="base"><td>Assume nothing moves (baseline)</td><td class="num">&ndash;</td><td class="num">{base_s:.3f}</td><td></td><td class="num">{base_m:.3f}</td><td></td></tr>
</table>
<div class="cap" style="margin-top:-1mm">Mean &plusmn; sd over {ns_e} seeds. Ratios are errors relative to the all-{N2} model, with 95% intervals from a bootstrap that
resamples both the training seeds and the 100 test scenes. p-values in the text are Welch&rsquo;s t-tests across seeds, Holm-corrected across the
comparisons; scene-level tests are in the repository.</div>
<ul>
<li><b>Varying independently is enough.</b> One row and one column, and camera-frame tracks plus posed depth ({K} cells each), are no worse
than all {N2} on static points: error ratios {SE['rowcol']['static']['ratio']:.2f} and {SE['tiedrow']['static']['ratio']:.2f}, with 95% intervals that end at
{SE['rowcol']['static']['ratio_ci_2level'][1]:.2f} and {SE['tiedrow']['static']['ratio_ci_2level'][1]:.2f}. On moving points they cost a small
{100 * (rc_m['ratio'] - 1):.0f}% and {100 * (tr_m['ratio'] - 1):.0f}%, consistent across seeds ({fp(rc_m['welch_p_seeds_holm'])} and {fp(tr_m['welch_p_seeds_holm'])}).</li>
<li><b>Tied labels confound time and camera.</b> With only t<sub>tgt</sub> = t<sub>cam</sub>, the static-point error is {tied['ratio']:.1f} times
higher (95% CI {tied['ratio_ci_2level'][0]:.1f} to {tied['ratio_ci_2level'][1]:.1f}), {every}{" and in every seed" if tied['every_seed_worse'] else ""}
({fp(tied['welch_p_seeds_holm'])}), and it grows with the gap between
t<sub>tgt</sub> and t<sub>cam</sub> (Figure 1): nothing in the labels says which index sets the camera. The model reads time as camera motion:
asked about a static point from one camera at different times, it moves the point by {DR['tied']['mean']:.2f} units on average, where the other
models keep it still ({min(DR[d]['mean'] for d in INDEP):.2f} to {max(DR[d]['mean'] for d in INDEP):.2f}).
Moving points: {100 * (ti_m['ratio'] - 1):.0f}% worse ({fp(ti_m['welch_p_seeds_holm'])}).</li>
<li><b>One row of posed depth resolves it, even from other clips.</b> With tracks on half of the clips and posed depth on the other half,
so that no clip has both, the static-point error is {f2(sp['mean'])}, {SPLIT_VS_FULL} (ratio {sp['ratio']:.2f}, 95% CI {sp['ratio_ci_2level'][0]:.2f} to
{sp['ratio_ci_2level'][1]:.2f}); having both on the same clips is {SPLIT_VS_SAME}.</li>
</ul>

<img class="fig2" src="../../results/sparse_queries/decoder_only_error_maps_static.png" alt="Static-point error maps for six label sets, decoder-only training">
<div class="cap"><b>Figure 2.</b> The post-training version: a fresh decoder on the frozen encoder of an all-{N2} model, trained with each label set
(mean of {ns_d} seeds). Every label set in which the time indices vary independently somewhere matches all {N2}, random cells included
({f2(m('random', 'static', SD))} against {f2(m('full', 'static', SD))}). The two that keep a pair tied fail in different ways: tracks only blurs time
into the camera (error grows away from the diagonal, {f2(m('tied', 'static', SD))}); posed depth only never sees time change, so it moves static
points with it (error grows down the rows, {f2(m('row', 'static', SD))}{ROW_DRIFT}). Both differ from all {N2} across seeds (Welch
{fp(SD['tied']['static']['welch_p_seeds_holm'])} and {fp(SD['row']['static']['welch_p_seeds_holm'])}).</div>

<h2><span class="n">04</span>What it suggests for D4RT</h2>
<ul>
<li><b>Camera-frame tracks alone may not teach a change of reference frame.</b> Queries with t<sub>tgt</sub> &ne; t<sub>cam</sub>, as in
world-coordinate tracking, need labels in which time and camera vary independently. This matters most for real videos without camera
poses, where camera-frame tracks may be the only 3D labels.</li>
<li><b>World-coordinate tracks may not be needed for it.</b> Camera-frame tracks plus posed depth, each easier to obtain, covered the rest,
even from different clips: tracks from videos without camera poses could be paired with posed depth from other footage.</li>
<li><b>A cheap check at full scale:</b> hold out queries with t<sub>src</sub>, t<sub>tgt</sub> and t<sub>cam</sub> all different during training and score them.</li>
</ul>

<h2><span class="n">05</span>Limits and next steps</h2>
<p class="note">A small model on synthetic 32 &times; 32 clips; the camera keeps a fixed height and tilt so that metric depth is recoverable from one
frame. At this scale the model learns camera motion well but tracks moving objects only partly: even with all {N2} combinations its error on moving
points grows with the time gap ({f2(m('full', 'moving'))} against {f2(base_m)} for assuming nothing moves). Static points, whose answer depends only on
the camera, therefore give the cleanest comparison; moving points show the same ordering. Three seeds per label set: the intervals resample seeds
as well as scenes, but the seed-to-seed spread is itself only roughly known. Next: Kubric clips at higher resolution, and how small the share of
posed-depth clips can be.</p>

<div class="refs">
<p>C. Zhang et al. Efficiently reconstructing dynamic scenes one D4RT at a time. CVPR, 2026.</p>
<p>Code, data, per-run results and statistics: github.com/RS-010806/composition/tree/main/sparse_queries</p>
</div>
"""

html = (HERE / "template.html").read_text()
html = html.replace("{{DEK}}", DEK).replace("{{TILES}}", TILES).replace("{{BODY1}}", BODY1).replace("{{BODY2}}", BODY2)
(HERE / "report.html").write_text(html)
print("wrote", HERE / "report.html")
