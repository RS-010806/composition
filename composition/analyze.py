"""Summarize the composition experiments and draw the figure.

Inputs (results/):
  scratch_cnn.json               small CNN trained from scratch (posttrain.py)
  posttrain_probe_runs.csv       linear heads on frozen Gemini image embeddings (posttrain_probe.js)
  icl_records.csv                in-context trials (icl_runner.js)
  icl_to_posttrain_queries.csv   teacher answers in the in-context-labels pilot (icl_to_posttrain.js)
  icl_to_posttrain_summary.json  student scores in that pilot
Outputs:
  results/composition_summary.md, results/composition_figure.png
"""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MODEL_NAMES = {"gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite", "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite",
               "gemma-4-26b-a4b-it": "Gemma 4 26B-A4B"}


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mean_sd(vals):
    m = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1))
    return m, sd


def load_scratch():
    data = json.loads((RES / "scratch_cnn.json").read_text())
    groups = defaultdict(list)
    for run in data["runs"]:
        for design, res in run["results"].items():
            groups[(run["p_home"], design)].append(res)
    out = {}
    for key, rs in groups.items():
        m, sd = mean_sd([r["held_out"]["compound_acc"] for r in rs])
        out[key] = {"seeds": len(rs), "held": m, "sd": sd,
                    "color": sum(r["held_out"]["color_acc"] for r in rs) / len(rs),
                    "shape": sum(r["held_out"]["shape_acc"] for r in rs) / len(rs),
                    "shortcut": sum(r["held_out"]["shortcut_rate"] for r in rs) / len(rs),
                    "seen": sum(r["seen"]["compound_acc"] for r in rs) / len(rs)}
    return out


def load_probe():
    groups = defaultdict(list)
    for r in csv.DictReader((RES / "posttrain_probe_runs.csv").open()):
        groups[(r["design"], r["head"])].append(r)
    out = {}
    for key, rs in groups.items():
        m, sd = mean_sd([float(r["held_compound"]) for r in rs])
        avg = lambda col: sum(float(r[col]) for r in rs) / len(rs)
        out[key] = {"seeds": len(rs), "n": int(rs[0]["n"]), "compounds": int(rs[0]["compounds"]), "held": m, "sd": sd,
                    "color": avg("held_color"), "shape": avg("held_shape"), "seen": avg("seen_compound"),
                    "perfect": sum(float(r["held_compound"]) == 1.0 for r in rs)}
    return out


def load_icl():
    cells = defaultdict(lambda: {"n": 0, "exact": 0, "first": 0, "second": 0, "distractor": 0})
    for r in csv.DictReader((RES / "icl_records.csv").open()):
        if r["ok"] != "true":
            continue
        c = cells[(r["model"], r["design"], r["order"], r["binding"] == "true")]
        c["n"] += 1
        c["exact"] += r["exact"] == "true"
        c["first"] += r["reading"] == "first_word_is_color"
        c["second"] += r["reading"] == "second_word_is_color"
        c["distractor"] += r["distractorWord"] == "true"
    return cells


def load_pilot():
    path = RES / "icl_to_posttrain_summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def pooled(cells, design, order=None, binding=False, model=None):
    k = n = 0
    for (m, d, o, b), c in cells.items():
        if d == design and b == binding and (order is None or o == order) and (model is None or m == model):
            k += c["exact"]
            n += c["n"]
    return k, n


def summary_md(scratch, probe, icl, pilot):
    L = ["# Composition results", "",
         "Grid: 6 colors x 6 shapes (36 combinations). Ten combinations are held out of every training set, "
         "example set and unlabeled pool. *cross* = one full row plus one full column of the grid (11 combinations). "
         "*confounded* = each color always shown with the same shape.", ""]

    L += ["## 1. Small CNN trained from scratch (JAX, 32 x 32 images)", "",
          "| Training mix (12,000 images) | Post-training (220 images) | Seeds | Held-out combinations right | Color right | Shape right | Seen combinations right |",
          "|---|---|---|---|---|---|---|"]
    for (p, d), v in sorted(scratch.items(), key=lambda kv: (-kv[0][0], ["none", "correlated", "cross", "all26"].index(kv[0][1]))):
        mix = "balanced over 26 combinations" if p < 0 else f"correlated ({p:.0%} typical pairs)"
        L.append(f"| {mix} | {d} | {v['seeds']} | {v['held']:.0%} ± {v['sd']:.0%} | {v['color']:.0%} | {v['shape']:.0%} | {v['seen']:.0%} |")

    L += ["", "## 2. Post-training on a frozen pretrained model (Gemini image embeddings, ridge heads)", "",
          "| Post-training set | Head | Images | Combinations | Held-out combinations right (20 seeds) | Seeds at 100% | Color right | Shape right | Seen combinations right |",
          "|---|---|---|---|---|---|---|---|---|"]
    order = [("cross", "factorized"), ("random11", "factorized"), ("confounded", "factorized"), ("cross", "joint"),
             ("all26", "factorized"), ("all26_full", "factorized"), ("all26_full", "joint")]
    for key in order:
        v = probe[key]
        L.append(f"| {key[0]} | {'one label per attribute' if key[1] == 'factorized' else 'one label per combination'} | {v['n']} | "
                 f"{v['compounds']} | {v['held']:.0%} ± {v['sd']:.0%} | {v['perfect']}/{v['seeds']} | {v['color']:.0%} | {v['shape']:.0%} | {v['seen']:.0%} |")

    L += ["", "## 3. In context (11 labeled examples in made-up words, one new picture)", "",
          "| Model | Example set | Word order | Test picture | Exact name | 95% CI |",
          "|---|---|---|---|---|---|"]
    for (m, d, o, b), c in sorted(icl.items(), key=lambda kv: (kv[0][0] != "gemini-3.5-flash-lite", kv[0][0], ["cross", "random", "confounded"].index(kv[0][1]), kv[0][3], kv[0][2])):
        lo, hi = wilson(c["exact"], c["n"])
        L.append(f"| {MODEL_NAMES.get(m, m)} | {d} | {'color word first' if o == 'adj' else 'shape word first'} | "
                 f"{'two objects' if b else 'one object'} | {c['exact']}/{c['n']} | {lo:.2f}-{hi:.2f} |")
    first = sum(c["first"] for (m, d, o, b), c in icl.items() if d == "confounded")
    second = sum(c["second"] for (m, d, o, b), c in icl.items() if d == "confounded")
    total = sum(c["n"] for (m, d, o, b), c in icl.items() if d == "confounded")
    dist = sum(c["distractor"] for (m, d, o, b), c in icl.items() if b)
    nb = sum(c["n"] for (m, d, o, b), c in icl.items() if b)
    L += ["", f"Confounded examples, all three models: every answer ({first + second}/{total}) matches one of the two readings the "
          f"examples allow. {first}/{total} read the first word as the color and {second}/{total} read the second word as the color, "
          "in both word orders.", f"Two-object pictures: {dist}/{nb} answers used a word belonging to the other object."]

    if pilot:
        L += ["", "## 4. Pilot: in-context labels as post-training data", "",
              "Labeled set: the 6 confounded combinations (108 images). Unlabeled pool: one image of each of the other 20 shown "
              "combinations. Teacher: Gemini 3.5 Flash-Lite with 11 labeled confounded examples in context. Student: ridge heads on "
              "Gemini image embeddings. Scored on 160 images of the 10 held-out combinations.", "",
              "| Post-training data | Teacher labels right | Held-out combinations right | Color and shape swapped |", "|---|---|---|---|"]
        L.append(f"| confounded only | - | {pilot['labeled_only']['compound']:.0%} | {pilot['labeled_only']['swapped']:.0%} |")
        for o, label in (("adj", "color word first"), ("noun", "shape word first")):
            if o in pilot:
                t, s = pilot[o]["teacher"], pilot[o]["student"]
                L.append(f"| + teacher labels, {label} | {t['right']}/{t['n']} | {s['compound']:.0%} | {s['swapped']:.0%} |")
        o = "adj" if "adj" in pilot else "noun"
        L.append(f"| + true labels for the same images | {pilot[o]['teacher']['n']}/{pilot[o]['teacher']['n']} | "
                 f"{pilot[o]['oracle_same_images']['compound']:.0%} | {pilot[o]['oracle_same_images']['swapped']:.0%} |")
    return "\n".join(L) + "\n"


GREEN, LIGHT, RED, GREY, DARK, INK, MUTED, GRID = "#1a7f37", "#7cc48f", "#cf222e", "#afb8c1", "#57606a", "#1f2328", "#57606a", "#eaeef2"


def style(ax):
    ax.set_ylim(0, 1.15)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d0d7de")
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))


FS = {"value": 8.8, "tick": 7.9, "title": 9.6, "ylabel": 8.6}


def bars(ax, items, title):
    for x, (label, val, err, color, note) in enumerate(items):
        ax.bar(x, val, width=0.66, color=color, yerr=err, error_kw={"lw": 0.9, "ecolor": DARK, "capsize": 2.5})
        ax.text(x, min(val + (err or 0), 1.0) + 0.03, note or f"{val:.0%}", ha="center", va="bottom",
                fontsize=FS["value"], fontweight="bold", color=INK)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([i[0] for i in items], fontsize=FS["tick"], color=INK)
    ax.set_xlim(-0.6, len(items) - 0.4)
    ax.set_title(title, fontsize=FS["title"], color=INK, loc="left", pad=8, fontweight="bold")


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    try:
        from PIL import Image
        Image.open(path).convert("RGB").save(path)
    except ImportError:
        pass


def figure(scratch, probe, icl, pilot, path, icl_model="gemini-3.5-flash-lite"):
    """Full 2 x 2 figure for the README (every condition)."""
    FS.update(value=8.8, tick=7.9, title=9.6, ylabel=8.6)
    fig = plt.figure(figsize=(11.6, 8.2), dpi=220)
    outer = fig.add_gridspec(2, 1, hspace=0.55)
    top = outer[0].subgridspec(1, 2, width_ratios=[2.1, 5], wspace=0.22)
    bot = outer[1].subgridspec(1, 2, width_ratios=[5, 4], wspace=0.22)
    axa, axb = fig.add_subplot(top[0]), fig.add_subplot(top[1])
    axc, axd = fig.add_subplot(bot[0]), fig.add_subplot(bot[1])
    for ax in (axa, axb, axc, axd):
        style(ax)

    a0, a1 = scratch[(-1.0, "none")], scratch[(-1.0, "cross")]
    bars(axa, [("trained on\n26 combinations", a0["held"], a0["sd"], GREY, None),
               ("+ row + column\npost-training", a1["held"], a1["sd"], GREEN, None)],
         "a  Small CNN trained from scratch")
    axa.set_ylabel("Never-seen combinations named correctly", fontsize=FS["ylabel"], color=INK)

    p = lambda d, h="factorized": (probe[(d, h)]["held"], probe[(d, h)]["sd"])
    bars(axb, [("row + column\n(11 combinations)", *p("cross"), GREEN, None),
               ("random\n(11 combinations)", *p("random11"), GREY, None),
               ("confounded\n(6 combinations)", *p("confounded"), RED, None),
               ("row + column, one\nlabel per combination", *p("cross", "joint"), DARK, None),
               ("all 26 shown", *p("all26"), LIGHT, None)],
         "b  Post-training on a frozen pretrained model (Gemini image embeddings, linear heads)")

    def c(label, color, design, order=None, binding=False):
        k, n = pooled(icl, design, order, binding, icl_model)
        return (label, (k / n if n else 0.0), None, color, f"{k}/{n}")
    bars(axc, [c("row + column", GREEN, "cross"), c("random", GREY, "random"),
               c("confounded,\ncolor word first", LIGHT, "confounded", "adj"),
               c("confounded,\nshape word first", RED, "confounded", "noun"),
               c("row + column,\ntwo objects", GREEN, "cross", None, True)],
         f"c  In context ({MODEL_NAMES[icl_model]}, 11 labeled examples)")
    axc.set_ylabel("Never-seen combinations named correctly", fontsize=FS["ylabel"], color=INK)

    t_adj, t_noun = pilot["adj"], pilot["noun"]
    bars(axd, [("confounded\nonly", pilot["labeled_only"]["compound"], None, RED, None),
               ("+ labels from\ncontext, color\nword first", t_adj["student"]["compound"], None, GREEN, None),
               ("+ labels from\ncontext, shape\nword first", t_noun["student"]["compound"], None, RED, None),
               ("+ true\nlabels", t_adj["oracle_same_images"]["compound"], None, LIGHT, None)],
         "d  In-context labels as post-training data (pilot)")
    fig.text(0.07, 0.0, "6 colors x 6 shapes. Every score is on combinations absent from that condition's training or example set "
             "(the 10 held-out combinations in a, b and d). Row + column = one full row plus one full column of the grid;\nconfounded = each "
             "color always with the same shape. Error bars: sd over seeds (5 in a, 20 in b). In c, labels give exact names out of trials. "
             f"In d, the trained heads are scored on 160 images.\nIn d, the in-context labels were right for {t_adj['teacher']['right']}/{t_adj['teacher']['n']} "
             f"images (color word first) and {t_noun['teacher']['right']}/{t_noun['teacher']['n']} (shape word first, "
             f"{t_noun['teacher']['swapped']} with color and shape swapped).", fontsize=7.3, color=MUTED, va="top")
    save(fig, path)


def report_figure(scratch, probe, icl, pilot, path):
    """Focused one-row figure for the two-page report."""
    FS.update(value=8.8, tick=7.6, title=9.2, ylabel=8.2)
    fig = plt.figure(figsize=(16.0, 4.0), dpi=230)
    row = fig.add_gridspec(1, 4, width_ratios=[2.2, 4, 4, 4], wspace=0.14)
    axa, axb, axc, axd = (fig.add_subplot(row[i]) for i in range(4))
    for ax in (axa, axb, axc, axd):
        style(ax)
    for ax in (axb, axc, axd):
        ax.tick_params(axis="y", labelleft=False)

    a0, a1 = scratch[(-1.0, "none")], scratch[(-1.0, "cross")]
    bars(axa, [("26 shown", a0["held"], a0["sd"], GREY, None), ("+ row +\ncolumn", a1["held"], a1["sd"], GREEN, None)],
         "a  CNN trained from scratch")
    axa.set_ylabel("Never-seen combinations named correctly", fontsize=FS["ylabel"], color=INK)

    p = lambda d: (probe[(d, "factorized")]["held"], probe[(d, "factorized")]["sd"])
    bars(axb, [("row +\ncolumn", *p("cross"), GREEN, None), ("random", *p("random11"), GREY, None),
               ("confounded", *p("confounded"), RED, None), ("all 26\nshown", *p("all26"), LIGHT, None)],
         "b  Post-training (frozen Gemini embeddings)")

    def c(label, color, design, order=None):
        k, n = pooled(icl, design, order, False)
        return (label, (k / n if n else 0.0), None, color, f"{k}/{n}")
    bars(axc, [c("row +\ncolumn", GREEN, "cross"), c("random", GREY, "random"),
               c("confounded,\ncolor word\nfirst", LIGHT, "confounded", "adj"),
               c("confounded,\nshape word\nfirst", RED, "confounded", "noun")],
         "c  In context (three models)")

    t_adj, t_noun = pilot["adj"], pilot["noun"]
    bars(axd, [("confounded\nonly", pilot["labeled_only"]["compound"], None, RED, None),
               ("+ in-context\nlabels, color\nword first", t_adj["student"]["compound"], None, GREEN, None),
               ("+ in-context\nlabels, shape\nword first", t_noun["student"]["compound"], None, RED, None),
               ("+ true\nlabels", t_adj["oracle_same_images"]["compound"], None, LIGHT, None)],
         "d  In-context labels for post-training")
    save(fig, path)


SHAPE_NAMES = ["circle", "square", "triangle", "diamond", "star", "plus"]
COLOR_HEX = ["#d72828", "#f08714", "#e1cd1e", "#28a53c", "#285adc", "#8c32b4"]
HELD_OUT = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 1), (4, 5), (5, 1), (5, 2)]
DESIGNS = [
    ("Row + column: 11 combinations", "every color and shape, in mixed pairs",
     sorted({(c, 0) for c in range(6)} | {(0, s) for s in range(6)})),
    ("Random: 11 combinations", "here, yellow and triangle never appear",
     [(5, 5), (0, 1), (3, 1), (3, 0), (1, 5), (0, 3), (0, 4), (5, 4), (4, 4), (0, 5), (1, 4)]),  # seed 0 of the probe
    ("Confounded: 6 combinations", "each color always with the same shape",
     [(c, c) for c in range(6)]),
]


def glyph(ax, shape, x, y, r, color):
    import numpy as np
    from matplotlib import patches
    if shape == "circle":
        ax.add_patch(patches.Circle((x, y), r, color=color, lw=0))
        return
    if shape == "square":
        a = r * 0.82
        pts = [(-a, -a), (a, -a), (a, a), (-a, a)]
    elif shape == "triangle":
        pts = [(r * np.cos(t), r * np.sin(t)) for t in (np.pi / 2, np.pi / 2 + 2 * np.pi / 3, np.pi / 2 + 4 * np.pi / 3)]
        pts = [(px, py - r * 0.12) for px, py in pts]
    elif shape == "diamond":
        pts = [(0, r), (r * 0.7, 0), (0, -r), (-r * 0.7, 0)]
    elif shape == "star":
        pts = [((r if k % 2 == 0 else r * 0.45) * np.cos(np.pi / 2 + k * np.pi / 5),
                (r if k % 2 == 0 else r * 0.45) * np.sin(np.pi / 2 + k * np.pi / 5)) for k in range(10)]
    else:  # plus
        a, b = r, r * 0.34
        pts = [(-b, -a), (b, -a), (b, -b), (a, -b), (a, b), (b, b), (b, a), (-b, a), (-b, b), (-a, b), (-a, -b), (-b, -b)]
    ax.add_patch(patches.Polygon([(x + px, y + py) for px, py in pts], closed=True, color=color, lw=0))


def designs_figure(path):
    """The grid, the three example sets and the held-out combinations, drawn as pictures."""
    from matplotlib import patches
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.75), dpi=230)
    for k, (ax, (title, sub, shown)) in enumerate(zip(axes, DESIGNS)):
        ax.set_xlim(-1.15, 6.05)
        ax.set_ylim(-0.25, 7.2)
        ax.set_aspect("equal")
        ax.axis("off")
        shown = set(shown)
        for s in range(6):   # column headers: grey shape glyphs
            glyph(ax, SHAPE_NAMES[s], s + 0.5, 6.55, 0.24, "#8c959f")
        for c in range(6):   # row headers: color swatches
            y = 5 - c + 0.5
            ax.add_patch(patches.FancyBboxPatch((-0.92, y - 0.2), 0.42, 0.4, boxstyle="round,pad=0.02", color=COLOR_HEX[c], lw=0))
            for s in range(6):
                x0, y0 = s, 5 - c
                held = (c, s) in HELD_OUT
                face = "#ffffff" if (c, s) in shown else ("#eef1f4" if held else "#fafbfc")
                ax.add_patch(patches.Rectangle((x0 + 0.04, y0 + 0.04), 0.92, 0.92, facecolor=face, edgecolor="#d0d7de", lw=0.7))
                if (c, s) in shown:
                    glyph(ax, SHAPE_NAMES[s], x0 + 0.5, y0 + 0.5, 0.33, COLOR_HEX[c])
                elif held:
                    ax.text(x0 + 0.5, y0 + 0.47, "?", ha="center", va="center", fontsize=11, color="#8c959f", fontweight="bold")
        ax.text(-1.15, -0.05, title, fontsize=10.2, fontweight="bold", color=INK, va="top")
        ax.text(-1.15, -0.62, sub, fontsize=8.9, color=MUTED, va="top")
    fig.subplots_adjust(wspace=0.08)
    save(fig, path)


def main():
    scratch, probe, icl, pilot = load_scratch(), load_probe(), load_icl(), load_pilot()
    (RES / "composition_summary.md").write_text(summary_md(scratch, probe, icl, pilot))
    figure(scratch, probe, icl, pilot, RES / "composition_figure.png")
    report_figure(scratch, probe, icl, pilot, RES / "composition_figure_report.png")
    designs_figure(RES / "designs.png")
    print((RES / "composition_summary.md").read_text())


if __name__ == "__main__":
    main()
