"""Statistics for the sparse-query experiment.

End to end (3 seeds per label set): every model is scored on the same 100 test scenes, so each label set is compared
with the all-36 model scene by scene. Two sources of variation matter: which scenes are tested, and which training
seed produced the model. The headline interval resamples both (two-level bootstrap, 10,000 resamples: test scenes
with replacement, and for each label set its training seeds with replacement). Alongside: Welch's t-test on the seed
means (the replication test), and, conditional on the trained models, a scene-level bootstrap interval and a two-sided
Wilcoxon signed-rank test on per-scene errors averaged over seeds.
Decoder only (3 seeds per label set): seed means ± sd and Welch's t-test against the all-36 decoder.

python stats.py --out ../results/sparse_queries
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as st

ORDER = ["full", "rowcol", "random", "tiedrow", "split", "tied", "row"]
KEY = {"full": "general_all"}


def load(pattern_dir, glob):
    out = defaultdict(dict)
    for f in sorted(Path(pattern_dir).glob(glob)):
        if f.name.endswith("_train.json"):
            continue
        r = json.load(open(f))
        out[r["design"]][r["seed"]] = r
    return out


def seed_values(runs, d, which):
    key = KEY.get(d, "general_heldout")
    return np.array([runs[d][s][key][which]["epe"] for s in sorted(runs[d])])


def boot_ci(x, n=10000, seed=0, fn=np.mean):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    b = fn(x[idx], axis=1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def scene_matrix(runs, d, which):
    """(seeds, scenes) per-scene errors; requires per-scene results for every seed."""
    rows = [runs[d][s].get(f"per_scene_{which}") for s in sorted(runs[d])]
    if any(r is None for r in rows):
        return None
    return np.array([[np.nan if v is None else v for v in r] for r in rows], float)


def two_level(A, B, n=10000, seed=2):
    """95% intervals for mean(A) - mean(B) and mean(A) / mean(B), resampling test scenes (shared, since both are
    scored on the same scenes) and each side's training seeds. A, B: (seeds, scenes)."""
    ok = np.isfinite(A).all(0) & np.isfinite(B).all(0)
    A, B = A[:, ok], B[:, ok]
    rng = np.random.default_rng(seed)
    n_sc = A.shape[1]
    sc = rng.integers(0, n_sc, size=(n, n_sc))
    sa = rng.integers(0, len(A), size=(n, len(A)))
    sb = rng.integers(0, len(B), size=(n, len(B)))
    a = np.array([A[sa[i]][:, sc[i]].mean() for i in range(n)])
    b = np.array([B[sb[i]][:, sc[i]].mean() for i in range(n)])
    pct = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
    return pct(a - b), pct(a / b)


def compare(runs, d, which, ref="full"):
    out = {}
    a, b = seed_values(runs, d, which), seed_values(runs, ref, which)
    out["seeds"] = len(a)
    out["mean"], out["sd"] = float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0
    out["seed_values"] = a.round(4).tolist()
    if d != ref and len(a) > 1 and len(b) > 1:
        out["welch_p_seeds"] = float(st.ttest_ind(a, b, equal_var=False).pvalue)
    A, B = scene_matrix(runs, d, which), scene_matrix(runs, ref, which)
    if d != ref and A is not None and B is not None:
        ea, eb = np.nanmean(A, 0), np.nanmean(B, 0)
        ok = np.isfinite(ea) & np.isfinite(eb)
        diff = ea[ok] - eb[ok]
        out["scenes"] = int(ok.sum())
        out["diff"] = float(diff.mean())
        out["diff_ci"] = boot_ci(diff)
        out["ratio"] = float(ea[ok].mean() / eb[ok].mean())
        rng = np.random.default_rng(1)
        idx = rng.integers(0, ok.sum(), size=(10000, ok.sum()))
        r = ea[ok][idx].mean(1) / eb[ok][idx].mean(1)
        out["ratio_ci"] = (float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)))
        out["wilcoxon_p"] = float(st.wilcoxon(diff).pvalue)
        out["scenes_worse"] = int((diff > 0).sum())
        out["diff_ci_2level"], out["ratio_ci_2level"] = two_level(A, B)
        out["every_seed_worse"] = bool(a.min() > b.max())          # worst seed of the reference beats the best seed of d
        out["every_seed_better"] = bool(a.max() < b.min())
    return out


def holm(ps):
    """Holm-Bonferroni adjusted p-values for a dict {key: p}."""
    keys = sorted(ps, key=ps.get)
    m, out, run = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * ps[k]))
        out[k] = run
    return out


def fmt_p(p):
    return "<1e-15" if p < 1e-15 else (f"{p:.1e}" if p < 1e-3 else f"{p:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/sparse_queries")
    a = ap.parse_args()
    out = Path(a.out)
    e2e = load("runs", "*_eval.json")
    dec = load("runs_dec", "enc-*_dec-*.json")
    res = {"end_to_end": {}, "decoder_only": {}}
    for name, runs in (("end_to_end", e2e), ("decoder_only", dec)):
        for d in ORDER:
            if d in runs:
                res[name][d] = {w: compare(runs, d, w) for w in ("static", "moving")}
        # Holm correction across the comparisons with the all-36 model in this table
        for test in ("wilcoxon_p", "welch_p_seeds"):
            ps = {(d, w): c[test] for d, ws in res[name].items() for w, c in ws.items() if test in c}
            for (d, w), p in holm(ps).items():
                res[name][d][w][test + "_holm"] = p
    # does it matter whether the two kinds of label come from the same clips? (one planned comparison per point type)
    if "split" in e2e and "tiedrow" in e2e:
        res["split_vs_tiedrow"] = {w: compare(e2e, "split", w, ref="tiedrow") for w in ("static", "moving")}
        for test in ("wilcoxon_p", "welch_p_seeds"):
            for w, p in holm({w: c[test] for w, c in res["split_vs_tiedrow"].items()}).items():
                res["split_vs_tiedrow"][w][test + "_holm"] = p
    json.dump(res, open(out / "stats.json", "w"), indent=1)

    L = ["# Statistics", "",
         "3D error on held-out combinations (t_src, t_tgt, t_cam all different), 100 test scenes. Mean ± sd over seeds.",
         "",
         "- Difference and ratio to all 36, seeds and scenes: 95% interval from a two-level bootstrap that resamples the",
         "  test scenes and each label set's training seeds. This is the headline interval.",
         "- Welch p: t-test on the seed means (does the difference replicate across training runs?).",
         "- Scene-level columns condition on the trained models: per scene, errors averaged over seeds, then a 95% bootstrap",
         "  interval over scenes and a two-sided Wilcoxon signed-rank test.",
         "- Both p columns are Holm-corrected across all comparisons with the all-36 model in the same table.", ""]
    for name in ("end_to_end", "decoder_only"):
        L += [f"## {name.replace('_', ' ')}", ""]
        if name == "end_to_end":
            L += ["| Label set | Points | Seeds | Error (mean ± sd) | Difference, seeds and scenes [95% CI] | Ratio, seeds and scenes [95% CI] | Welch p | "
                  "Difference, scenes only [95% CI] | Scenes worse | Wilcoxon p |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        else:
            L += ["| Label set | Points | Seeds | Error (mean ± sd) | Difference to all 36 | Welch p |", "|---|---|---|---|---|---|"]
        ref = res[name]["full"]
        for d, ws in res[name].items():
            for w, c in ws.items():
                tp = fmt_p(c["welch_p_seeds_holm"]) if "welch_p_seeds_holm" in c else "-"
                if name == "end_to_end":
                    d2 = f"{c['diff']:+.3f} [{c['diff_ci_2level'][0]:+.3f}, {c['diff_ci_2level'][1]:+.3f}]" if "diff" in c else "-"
                    r2 = f"{c['ratio']:.2f} [{c['ratio_ci_2level'][0]:.2f}, {c['ratio_ci_2level'][1]:.2f}]" if "ratio" in c else "-"
                    d1 = f"{c['diff']:+.3f} [{c['diff_ci'][0]:+.3f}, {c['diff_ci'][1]:+.3f}]" if "diff" in c else "-"
                    worse = f"{c['scenes_worse']}/{c['scenes']}" if "scenes" in c else "-"
                    wp = fmt_p(c["wilcoxon_p_holm"]) if "wilcoxon_p_holm" in c else "-"
                    L.append(f"| {d} | {w} | {c['seeds']} | {c['mean']:.3f} ± {c['sd']:.3f} | {d2} | {r2} | {tp} | {d1} | {worse} | {wp} |")
                else:
                    df = "reference" if d == "full" else f"{c['mean'] - ref[w]['mean']:+.3f}"
                    L.append(f"| {d} | {w} | {c['seeds']} | {c['mean']:.3f} ± {c['sd']:.3f} | {df} | {tp} |")
        L.append("")
    if "split_vs_tiedrow" in res:
        L += ["## Same clips or different clips", "",
              "Camera-frame tracks and posed depth on different clips (split) against both on every clip (tiedrow), end to end.", "",
              "| Points | Split | Same clips | Difference, seeds and scenes [95% CI] | Ratio [95% CI] | Welch p | Wilcoxon p |",
              "|---|---|---|---|---|---|---|"]
        for w, c in res["split_vs_tiedrow"].items():
            r = res["end_to_end"]["tiedrow"][w]
            L.append(f"| {w} | {c['mean']:.3f} ± {c['sd']:.3f} | {r['mean']:.3f} ± {r['sd']:.3f} | "
                     f"{c['diff']:+.3f} [{c['diff_ci_2level'][0]:+.3f}, {c['diff_ci_2level'][1]:+.3f}] | "
                     f"{c['ratio']:.2f} [{c['ratio_ci_2level'][0]:.2f}, {c['ratio_ci_2level'][1]:.2f}] | "
                     f"{fmt_p(c['welch_p_seeds_holm'])} | {fmt_p(c['wilcoxon_p_holm'])} |")
        L.append("")
    (out / "stats.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
