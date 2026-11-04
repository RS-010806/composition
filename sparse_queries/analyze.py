"""Summaries and figures for the sparse-query experiment.

python analyze.py --runs runs --out ../results/sparse_queries
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from scenes import T

# Okabe-Ito colours (colour-blind safe)
BLUE, GREEN, VERMILION, AMBER, SKY, GREY, INK = "#0072B2", "#009E73", "#D55E00", "#E69F00", "#56B4E9", "#9AA0A6", "#1B1F23"
ORDER = ["full", "rowcol", "random", "tiedrow", "tied", "row"]
N2, K = T * T, 2 * T - 1
NAMES = {
    "full": f"All {N2} combinations\n(D4RT's sampling)",
    "rowcol": f"One row + one column\n({K} of {N2})",
    "random": f"Random\n({K} of {N2})",
    "tiedrow": f"Camera-frame tracks\n+ posed depth ({K} of {N2})",
    "tied": f"Camera-frame tracks\nonly ({T} of {N2})",
    "row": f"Posed depth\nonly ({T} of {N2})",
}
SHORT = {"full": f"All {N2}", "rowcol": "Row + column", "random": "Random", "tiedrow": "Tracks + depth", "tied": "Tracks only", "row": "Depth only"}
COLOR = {"full": GREY, "rowcol": GREEN, "random": AMBER, "tiedrow": BLUE, "tied": VERMILION, "row": SKY}
TEXT = {"full": "#5F6670", "rowcol": "#007A5A", "random": "#A86A00", "tiedrow": "#005A8C", "tied": "#B04700", "row": "#1F78AA"}

plt.rcParams.update({"mathtext.default": "regular", "font.family": "Poppins", "font.size": 8.5, "axes.edgecolor": "#C9CED6", "axes.linewidth": 0.8,
                     "xtick.color": "#4A5059", "ytick.color": "#4A5059", "axes.labelcolor": "#2B3036", "text.color": INK})


def load(runs):
    """Results of end-to-end runs (<design>_s<seed>_eval.json) or of decoder-only runs (enc-*_dec-<design>_s<seed>.json)."""
    res = defaultdict(list)
    for f in sorted(Path(runs).glob("*.json")):
        if f.name.endswith("_train.json"):
            continue
        r = json.load(open(f))
        stem = f.name[:-len("_eval.json")] if f.name.endswith("_eval.json") else f.name[:-len(".json")]
        r["A"] = np.load(f.with_name(stem + "_A.npy"))
        res[r["design"]].append(r)
    return res


def ms(vals):
    v = np.array([x for x in vals if x is not None], float)
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v)) if len(v) else (None, None, 0)


def table(res):
    rows = []
    for d in ORDER:
        if d not in res:
            continue
        rs = res[d]
        key = "general_all" if d == "full" else "general_heldout"
        rows.append(dict(
            design=d, seeds=len(rs), cells=rs[0]["supervised_cells_per_tsrc"],
            mov=ms([r[key]["moving"]["epe"] for r in rs]), sta=ms([r[key]["static"]["epe"] for r in rs]),
            mov_acc=ms([r[key]["moving"]["acc25"] for r in rs]), sta_acc=ms([r[key]["static"]["acc25"] for r in rs]),
            seen_mov=ms([r["seen_moving"]["epe"] for r in rs]), seen_sta=ms([r["seen_static"]["epe"] for r in rs]),
            track=ms([r["track_cam0_heldout_moving"]["epe"] if d != "full" else None for r in rs]),
            base_mov=ms([r["baseline_general"]["moving"]["epe"] for r in rs]),
            base_sta=ms([r["baseline_general"]["static"]["epe"] for r in rs])))
    return rows


def heatmaps(res, path, rows=None, which="moving"):
    designs = [d for d in ORDER if d in res]
    key = "mov" if which == "moving" else "sta"
    held = {r["design"]: r[key][0] for r in rows} if rows else {}
    fig, axes = plt.subplots(1, len(designs), figsize=(2.05 * len(designs), 2.7), dpi=240)
    maps = {d: np.mean([np.array(r[f"map_{which}"]) for r in res[d]], 0) for d in designs}
    vmax = max(1.0, np.percentile(np.concatenate([m.ravel() for m in maps.values()]), 97))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("err", ["#F4FAF7", "#9ED9C3", "#F2C46B", "#E07B39", "#B23A12"])
    for ax, d in zip(np.atleast_1d(axes), designs):
        A = res[d][0]["A"][0]                       # supervised cells for t_src = 0 (seed 0)
        im = ax.imshow(maps[d], cmap=cmap, vmin=0, vmax=vmax, origin="upper")
        if d == "full":
            ax.add_patch(Rectangle((-0.5, -0.5), T, T, fill=False, ec=INK, lw=1.1))
        else:
            for g in range(T):
                for c in range(T):
                    if A[g, c]:
                        ax.add_patch(Rectangle((c - 0.5, g - 0.5), 1, 1, fill=False, ec=INK, lw=1.1))
        ax.set_xticks([0, T - 1], ["0", str(T - 1)])
        ax.set_yticks([0, T - 1], ["0", str(T - 1)])
        ax.tick_params(length=0, labelsize=7)
        ax.set_xlabel("camera $t_{cam}$", fontsize=7.4, labelpad=1)
        if d == designs[0]:
            ax.set_ylabel("target time $t_{tgt}$", fontsize=7.4, labelpad=1)
        ax.set_title(NAMES[d], fontsize=8, fontweight="medium", color=INK, pad=4, linespacing=1.15)
        if d in held:
            lab = "error, all cells supervised" if d == "full" else "held-out error"
            ax.text(0.5, -0.36, f"{lab} {held[d]:.2f}", transform=ax.transAxes, ha="center", va="top", fontsize=7.4,
                    color=TEXT[d], fontweight="semibold")
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.012)
    cb.ax.tick_params(labelsize=6.5, length=0)
    cb.outline.set_visible(False)
    cb.set_label(f"3D error, {which} points", fontsize=7)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def bars(rows, path):
    """Held-out error per design, moving and static points side by side, with the 'nothing moves' baseline."""
    rows = rows[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 0.36 * len(rows) + 0.75), dpi=240, sharey=True)
    y = np.arange(len(rows))
    for ax, key, base_key, title in ((axes[0], "mov", "base_mov", "Moving points"), (axes[1], "sta", "base_sta", "Static points")):
        m = np.array([r[key][0] for r in rows])
        s = np.array([r[key][1] for r in rows])
        ax.barh(y, m, 0.62, color=[COLOR[r["design"]] for r in rows], xerr=s, error_kw=dict(lw=0.8, capsize=1.8, ecolor="#5F6670"))
        base = rows[0][base_key][0]
        ax.axvline(base, color="#5F6670", lw=0.9, ls=(0, (3, 2.5)))
        ax.text(base, len(rows) - 0.45, " nothing moves", ha="left", va="bottom", fontsize=6.6, color="#5F6670")
        for yi, mi, si in zip(y, m, s):
            ax.text(mi + si + 0.02, yi, f"{mi:.2f}", va="center", ha="left", fontsize=6.8, color=INK)
        ax.set_title(title, fontsize=8, fontweight="medium", loc="left", pad=8)
        ax.set_xlim(0, max(base, (m + s).max()) * 1.18)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(length=0, labelsize=7)
        ax.grid(axis="x", color="#EDEFF2", lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("3D error on held-out combinations", fontsize=7.2)
    axes[0].set_yticks(y, [SHORT[r["design"]] for r in rows], fontsize=7.4)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def summary_md(rows, path):
    L = ["# Sparse time-camera supervision in a small D4RT-style model", "",
         "Mean 3D end-point error (scene units; spheres have radius 0.55 to 0.8) on held-out test scenes, for queries whose "
         "t_src, t_tgt and t_cam are all different. For the sparse designs these combinations were never supervised; "
         f"the all-{N2} model saw every combination. Mean ± sd over seeds; acc = share of queries within 0.25 units.", "",
         "| Supervision | Combinations per t_src | Seeds | Moving points | Static points | Moving, acc | Static, acc | Tracks in camera 0 (moving) | Seen combinations (moving / static) |",
         "|---|---|---|---|---|---|---|---|---|"]
    f = lambda t: "-" if t[0] is None else f"{t[0]:.3f} ± {t[1]:.3f}"
    p = lambda t: "-" if t[0] is None else f"{100 * t[0]:.0f}%"
    for r in rows:
        L.append(f"| {SHORT[r['design']]} | {r['cells']} | {r['seeds']} | {f(r['mov'])} | {f(r['sta'])} | {p(r['mov_acc'])} | {p(r['sta_acc'])} | "
                 f"{f(r['track'])} | {r['seen_mov'][0]:.3f} / {r['seen_sta'][0]:.3f} |")
    if rows:
        L += ["", f"Baseline that assumes nothing moves (the point where it was, in its own camera): moving {rows[0]['base_mov'][0]:.3f}, "
              f"static {rows[0]['base_sta'][0]:.3f}."]
    Path(path).write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--name", default="end_to_end", help="prefix for the output files")
    ap.add_argument("--out", default="../results/sparse_queries")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = load(a.runs)
    rows = table(res)
    json.dump(rows, open(out / f"{a.name}.json", "w"), indent=1)
    summary_md(rows, out / f"{a.name}.md")
    heatmaps(res, out / f"{a.name}_error_maps.png", rows, "moving")
    heatmaps(res, out / f"{a.name}_error_maps_static.png", rows, "static")
    bars(rows, out / f"{a.name}_heldout_error.png")
    print((out / f"{a.name}.md").read_text())
