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

DEK = (f"Trained on one row and one column of the {T} &times; {T} grid of (target time, camera) combinations, {K} of {N2}, a small "
       f"D4RT-style model answered the combinations it never saw as well as one trained on all {N2}. Trained only where target time and camera "
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
<p>D4RT (Zhang et al., CVPR 2026) answers queries <span class="q">(u, v, t<sub>src</sub>, t<sub>tgt</sub>, t<sub>cam</sub>)</span>: where the point seen at pixel (u, v)
of frame t<sub>src</sub> is at time t<sub>tgt</sub>, in the coordinates of the camera of frame t<sub>cam</sub>. Its training draws t<sub>tgt</sub>
and t<sub>cam</sub> independently and sets them equal 40% of the time, so every time-camera combination is supervised. Real labels are often
narrower. Camera-frame 3D tracks (2D tracks plus per-frame depth) only give queries with <b>t<sub>tgt</sub> = t<sub>cam</sub></b>; posed depth
only gives <b>t<sub>tgt</sub> = t<sub>src</sub></b>. Does the model answer combinations it was never trained on? In an earlier test on a
{T} &times; {T} grid of colors and shapes, labels for one row and one column taught a model all {N2} combinations, while the diagonal, where each
color always comes with the same shape, taught none. The same should hold here if time and camera vary independently somewhere in the labels.</p>

<img src="../../results/sparse_queries/end_to_end_error_maps_static.png" alt="Error maps on static points for each label set">
<div class="cap"><b>Figure 1.</b> Error on static points by target time (rows) and camera (columns), for points first seen in frame 0 of 100 new
scenes; mean of {ns_e} models per panel, each trained from scratch with the same number of training queries. Outlined cells were supervised (in the
different-clips model, each clip had only the row or only the diagonal); the rest were never seen in training. A static point's position depends only on the camera, so a model that keeps time and camera apart shows
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


def row(d):
    name, sub = LABEL[d]
    s, mo = SE[d]["static"], SE[d]["moving"]

    def rt(c, hi=False):
        if d == "full":
            return "<td class='num ref'>reference</td>"
        return (f"<td class='num{' hi' if hi else ''}'>{c['ratio']:.2f}&times; "
                f"<span class='ci'>[{c['ratio_ci_2level'][0]:.2f}, {c['ratio_ci_2level'][1]:.2f}]</span></td>")
    cells = f"{K} of {N2}<span class='sub'>{T} per clip</span>" if d == "split" else f"{E[d]['cells']} of {N2}"
    return (f"<tr><td>{name}<span class='sub'>{sub}</span></td><td class='num'>{cells}</td>"
            f"<td class='num'><b>{s['mean']:.3f}</b> <span class='sd'>&plusmn; {s['sd']:.3f}</span></td>{rt(s, d == 'tied')}"
            f"<td class='num'>{mo['mean']:.3f} <span class='sd'>&plusmn; {mo['sd']:.3f}</span></td>{rt(mo)}</tr>")


TT, TC, TS = "t<sub>tgt</sub>", "t<sub>cam</sub>", "t<sub>src</sub>"
LABEL = {"full": (f"All {N2} combinations", "D4RT&rsquo;s sampling; the reference"),
         "rowcol": ("One row + one column", f"{TT} = {TS} or {TC} = {TS}"),
         "tiedrow": ("Camera-frame tracks + posed depth", f"{TT} = {TC} or {TT} = {TS}, both on every clip"),
         "split": ("Tracks + posed depth, different clips", f"{TT} = {TC} on half of the clips, {TT} = {TS} on the rest"),
         "tied": ("Camera-frame tracks only", f"{TT} = {TC}")}
E2E = [d for d in ["full", "rowcol", "tiedrow", "split", "tied"] if d in SE]
INDEP = [d for d in E2E if d != "tied"]
sp = SE["split"]["static"]
sv = S["split_vs_tiedrow"]["static"]
pct = lambda c: f"{100 * (c['ratio_ci_2level'][0] - 1):.0f} to {100 * (c['ratio_ci_2level'][1] - 1):.0f}%"
TABLE = "".join(row(d) for d in E2E)
rc_m, tr_m, ti_m = SE["rowcol"]["moving"], SE["tiedrow"]["moving"], SE["tied"]["moving"]

BODY2 = f"""
<h2 style="margin-top:0"><span class="n">03</span>What I got</h2>
<div class="tlead">Error at the combinations with {TS}, {TT} and {TC} all different, which only the all-{N2} model was trained on, on 100 new scenes:
mean 3D distance, lower is better.</div>
<table class="res">
<tr class="grp"><th rowspan="2">Labels used in training</th><th rowspan="2" class="num">Trained on</th>
<th colspan="2" class="gh">Static points</th><th colspan="2" class="gh">Moving points</th></tr>
<tr><th class="num">Error</th><th class="num">vs all {N2} [95% CI]</th><th class="num">Error</th><th class="num">vs all {N2} [95% CI]</th></tr>
{TABLE}
<tr class="base"><td>Baseline: assume nothing moves</td><td class="num">none</td>
<td class="num">{base_s:.3f}</td><td></td><td class="num">{base_m:.3f}</td><td></td></tr>
</table>
<div class="cap" style="margin-top:-1mm">Mean &plusmn; sd over {ns_e} seeds. vs all {N2}: ratio to the all-{N2} error, with a 95% bootstrap interval over seeds and test scenes.
Baseline: the point where it was seen, as if nothing moved. p-values in the text: Welch&rsquo;s t-tests across seeds, Holm-corrected.</div>
<ul>
<li><b>Varying independently is enough.</b> One row and one column, or camera-frame tracks plus posed depth, are no worse than all {N2} on
static points (95% intervals end at {SE['rowcol']['static']['ratio_ci_2level'][1]:.2f}&times; and {SE['tiedrow']['static']['ratio_ci_2level'][1]:.2f}&times;) and cost only
{100 * (rc_m['ratio'] - 1):.0f}% and {100 * (tr_m['ratio'] - 1):.0f}% on moving points.</li>
<li><b>Tied labels confound time and camera.</b> With camera-frame tracks only, the error on static points is {tied['ratio']:.1f} times higher,
{every}{" and in every seed" if tied['every_seed_worse'] else ""} ({fp(tied['welch_p_seeds_holm'])}): nothing in the labels says which index sets the camera.
The model reads time as camera motion: asked about a static point from one camera at different times, it moves the point by
{DR['tied']['mean']:.2f} units on average, against {min(DR[d]['mean'] for d in INDEP):.2f} to {max(DR[d]['mean'] for d in INDEP):.2f} for the other models.</li>
<li><b>One row of posed depth resolves it, even from other clips.</b> With tracks and posed depth on different clips, the error matches all {N2}
({sp['ratio']:.2f}&times;). Both on the same clips is better still ({f2(m('tiedrow'))}; different clips give {sv['ratio']:.2f} times that, 95% CI
{sv['ratio_ci_2level'][0]:.2f} to {sv['ratio_ci_2level'][1]:.2f}).</li>
</ul>

<img class="fig2" src="../../results/sparse_queries/decoder_only_error_maps_static.png" alt="Static-point error maps for six label sets, decoder-only training">
<div class="cap"><b>Figure 2.</b> The post-training version: a fresh decoder on the frozen encoder of an all-{N2} model, trained with each label set
(mean of {ns_d} seeds). Label sets in which the time indices vary independently somewhere match all {N2}, random cells included.
The two that keep a pair tied fail in different ways: tracks only blurs time
into the camera (error grows away from the diagonal, {f2(m('tied', 'static', SD))}); posed depth only never sees time change, so it moves static
points with it (error grows down the rows, {f2(m('row', 'static', SD))}{ROW_DRIFT}). Both differ from all {N2} across seeds (Welch
{fp(SD['tied']['static']['welch_p_seeds_holm'])} and {fp(SD['row']['static']['welch_p_seeds_holm'])}).</div>

<h2><span class="n">04</span>What it suggests for D4RT</h2>
<ul>
<li><b>Camera-frame tracks alone may not teach a change of reference frame.</b> Queries with {TT} &ne; {TC}, as in world-coordinate
tracking, need labels in which time and camera vary independently.</li>
<li><b>Posed depth from other footage may supply it.</b> Tracks plus posed depth covered those queries even from different clips, so
unposed videos could contribute their tracks, with no need for world-coordinate tracks.</li>
<li><b>A cheap check at full scale:</b> hold out queries with t<sub>src</sub>, t<sub>tgt</sub> and t<sub>cam</sub> all different during training and score them.</li>
</ul>

<h2><span class="n">05</span>Limits and next steps</h2>
<p class="note">A small model on synthetic 32 &times; 32 clips, {ns_e} seeds per label set; the camera keeps a fixed height and tilt so that metric depth
is recoverable from one frame. The model learns camera motion well but tracks moving objects only partly ({f2(m('full', 'moving'))} against {f2(base_m)} for
assuming nothing moves, even with all {N2}), so static points, which depend only on the camera, give the cleanest comparison; moving points show the
same ordering. Next: Kubric clips at higher resolution, and how small the share of posed-depth clips can be.</p>
"""

html = (HERE / "template.html").read_text()
html = html.replace("{{DEK}}", DEK).replace("{{TILES}}", TILES).replace("{{BODY1}}", BODY1).replace("{{BODY2}}", BODY2)
(HERE / "report.html").write_text(html)
print("wrote", HERE / "report.html")
