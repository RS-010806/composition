"""Aggregate per-response verdicts into pass rates, intervals, tests and the figure.

Input : pair/data/verdicts.csv  (one row per graded response; final verdict after
        two-grader agreement or manual adjudication)
Output: pair/results/summary.csv, pair/results/stats.md, pair/results/figure.png
"""
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent  # pair/

MODELS = [  # (model id, display name, group)
    ("gemini-3.1-pro-preview", "Gemini 3.1 Pro†", "Larger"),
    ("gemini-3-flash-preview", "Gemini 3 Flash", "Larger"),
    ("gemini-3.8-flash", "Gemini 3.8 Flash", "Larger"),
    ("gemini-4-argon", "Gemini 4 Argon", "Larger"),
    ("gemini-3.1-flash-lite", "Gemini 3.1 Flash-Lite", "Small"),
    ("gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite", "Small"),
    ("gemma-4-26b-a4b-it", "Gemma 4 26B-A4B", "Small"),
]
CONDS = [  # (condition, column title)
    ("B", "Textbook rule\n(aloo matar)"),
    ("K", "States the\nstricter rule\nwhen asked"),
    ("A0", "Applies it\nat dinner"),
    ("A_recite", "+ recite\nrules first"),
    ("A_demo", "+ composition\nexample"),
    ("A_rule", "+ rule in\ncontext"),
    ("A0_high", "+ high\nthinking"),
]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fisher_two_sided(a, b, c, d):
    """Fisher exact test for the 2x2 table [[a, b], [c, d]]."""
    n1, n2, m = a + b, c + d, a + c
    total = n1 + n2

    def p_of(x):
        return math.comb(n1, x) * math.comb(n2, m - x) / math.comb(total, m)

    p_obs = p_of(a)
    lo, hi = max(0, m - n2), min(n1, m)
    return min(1.0, sum(p_of(x) for x in range(lo, hi + 1) if p_of(x) <= p_obs * (1 + 1e-9)))


def load():
    rows = list(csv.DictReader((ROOT / "data" / "verdicts.csv").open()))
    cells = defaultdict(lambda: [0, 0])
    stated = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["final_pass"] == "":
            continue
        key = (r["model"], r["cond"])
        cells[key][1] += 1
        cells[key][0] += int(r["final_pass"] == "1")
        if r["cond"] != "B" and r.get("states_all_green", "") != "":
            stated[key][1] += 1
            stated[key][0] += int(r["states_all_green"] == "1")
    return rows, cells, stated


def write_summary(cells, stated):
    out = ROOT / "results" / "summary.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "condition", "n", "passes", "pass_rate", "ci95_low", "ci95_high",
                    "answers_stating_all_green_rule"])
        for m, _, _ in MODELS:
            for c, _ in CONDS:
                k, n = cells.get((m, c), (0, 0))
                if n == 0:
                    continue
                lo, hi = wilson(k, n)
                sk, sn = stated.get((m, c), (0, 0))
                w.writerow([m, c, n, k, f"{k / n:.2f}", f"{lo:.2f}", f"{hi:.2f}",
                            f"{sk}/{sn}" if sn else ""])
    return out


def write_stats(cells):
    lines = ["# Key contrasts (two-sided Fisher exact test)", ""]

    def contrast(label, m1, c1, m2, c2):
        k1, n1 = cells.get((m1, c1), (0, 0))
        k2, n2 = cells.get((m2, c2), (0, 0))
        if n1 and n2:
            p = fisher_two_sided(k1, n1 - k1, k2, n2 - k2)
            lines.append(f"- {label}: {k1}/{n1} vs {k2}/{n2}, p = {p:.2g}")

    for m, name, grp in MODELS:
        if grp == "Small":
            contrast(f"{name}: knows (K) vs applies (A0)", m, "K", m, "A0")
            for c in ("A_recite", "A_demo", "A_rule", "A0_high"):
                contrast(f"{name}: A0 vs {c}", m, "A0", m, c)
    for cond, label in (("A0", "Applies at dinner (A0)"), ("K", "Gives the stricter rule (K)")):
        big = [cells.get((m, cond), (0, 0)) for m, _, g in MODELS if g == "Larger"]
        small = [cells.get((m, cond), (0, 0)) for m, _, g in MODELS if g == "Small"]
        k1, n1 = sum(k for k, _ in big), sum(n for _, n in big)
        k2, n2 = sum(k for k, _ in small), sum(n for _, n in small)
        p = fisher_two_sided(k1, n1 - k1, k2, n2 - k2)
        lines.append(f"- {label}, larger vs small models pooled: {k1}/{n1} vs {k2}/{n2}, p = {p:.2g}")
    contrast("Applies (A0): 3.1 Pro vs 3.1 Flash-Lite", "gemini-3.1-pro-preview", "A0",
             "gemini-3.1-flash-lite", "A0")
    contrast("Applies (A0): 3.8 Flash vs 3.5 Flash-Lite", "gemini-3.8-flash", "A0",
             "gemini-3.5-flash-lite", "A0")
    contrast("Applies (A0): Gemini 4 Argon vs 3.5 Flash-Lite", "gemini-4-argon", "A0",
             "gemini-3.5-flash-lite", "A0")
    (ROOT / "results" / "stats.md").write_text("\n".join(lines) + "\n")


def figure(cells, path):
    cmap = LinearSegmentedColormap.from_list("pass", ["#fbeaea", "#f4f6f8", "#d9efe1", "#1a7f37"])
    fig, ax = plt.subplots(figsize=(10.0, 0.75 * len(MODELS) + 0.9), dpi=220)
    nrow, ncol = len(MODELS), len(CONDS)
    for r, (m, name, grp) in enumerate(MODELS):
        for c, (cond, _) in enumerate(CONDS):
            k, n = cells.get((m, cond), (0, 0))
            y = nrow - 1 - r
            if n == 0:
                ax.add_patch(plt.Rectangle((c, y), 1, 1, facecolor="white", edgecolor="#d0d7de", lw=0.8))
                ax.text(c + 0.5, y + 0.5, "·", ha="center", va="center", fontsize=14, color="#8c959f")
                continue
            rate = k / n
            rgba = cmap(rate)
            ax.add_patch(plt.Rectangle((c, y), 1, 1, facecolor=rgba, edgecolor="white", lw=2))
            lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            color = "white" if lum < 0.5 else "#1f2328"
            ax.text(c + 0.5, y + 0.56, f"{k}/{n}", ha="center", va="center", fontsize=15,
                    fontweight="bold", color=color)
    ax.set_xlim(0, ncol)
    ax.set_ylim(0, nrow)
    ax.set_xticks([c + 0.5 for c in range(ncol)])
    ax.set_xticklabels([t for _, t in CONDS], fontsize=10)
    ax.xaxis.tick_top()
    ax.set_yticks([nrow - 1 - r + 0.5 for r in range(nrow)])
    ax.set_yticklabels([name for _, name, _ in MODELS], fontsize=11)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    n_small = sum(g == "Small" for _, _, g in MODELS)
    ax.axhline(n_small, color="#57606a", lw=1.2)
    ax.axvline(1, color="#57606a", lw=0.8, ls=":")
    ax.axvline(3, color="#57606a", lw=1.2)
    ytr = ax.get_yaxis_transform()
    for y0, y1, label in ((n_small, nrow, "larger models"), (0, n_small, "small models")):
        ax.plot([-0.255, -0.255], [y0 + 0.12, y1 - 0.12], color="#8c959f", lw=1, transform=ytr, clip_on=False)
        ax.text(-0.268, (y0 + y1) / 2, label, rotation=90, ha="right", va="center", fontsize=9.5,
                color="#57606a", transform=ytr)
    ax.text(5.0, -0.34, "in-context interventions (no retraining)", ha="center", va="center",
            fontsize=9.5, color="#57606a")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    try:
        from PIL import Image
        Image.open(path).convert("RGB").save(path)
    except ImportError:
        pass


def main():
    (ROOT / "results").mkdir(exist_ok=True)
    rows, cells, stated = load()
    write_summary(cells, stated)
    write_stats(cells)
    figure(cells, ROOT / "results" / "figure.png")
    print(f"{len(rows)} graded responses -> pair/results/summary.csv, pair/results/stats.md, pair/results/figure.png")


if __name__ == "__main__":
    main()
