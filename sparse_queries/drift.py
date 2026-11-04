"""How much does a model move a static point when only the target time changes?

For a static point the right answer does not depend on t_tgt. For every static test point and camera t_cam, this
measures the spread of the model's answers over the 6 target times: the mean distance of each answer from their
average. A model that keeps time and camera apart scores near 0.

python drift.py --runs runs --out ../results/sparse_queries/drift.json
python drift.py --runs runs --decoders runs_dec --out ../results/sparse_queries/drift_decoder_only.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

import d4rt_small as D
import scenes as S


def drift(params, test, model, fwd, rgb_pad):
    data, ts, py, px = test
    T = S.T
    n_clips, M = ts.shape
    TT = np.repeat(np.arange(T), T)[None].repeat(M, 0)          # cell order: t_tgt major, t_cam minor
    TC = np.tile(np.arange(T), T)[None].repeat(M, 0)
    out = []
    for c in range(n_clips):
        clip = np.full((M, T * T), c)
        TS = np.repeat(ts[c][:, None], T * T, 1)
        PY, PX = np.repeat(py[c][:, None], T * T, 1), np.repeat(px[c][:, None], T * T, 1)
        uv = np.stack([(PX + 0.5) / S.W, (PY + 0.5) / S.H], -1).astype(np.float32)
        pred = np.asarray(fwd(params, jnp.asarray(data["rgb"][c:c + 1] / 255.0, jnp.float32), uv.reshape(1, -1, 2),
                              TS.reshape(1, -1), TT.reshape(1, -1), TC.reshape(1, -1),
                              D.patches(rgb_pad, clip, TS, PY, PX).reshape(1, M * T * T, -1))).reshape(M, T, T, 3) * D.SCALE
        _, mov = S.ground_truth(data, clip, TS, PY, PX, TT, TC)
        static = ~mov[:, 0]
        p = pred[static]                                          # (points, t_tgt, t_cam, 3)
        spread = np.linalg.norm(p - p.mean(1, keepdims=True), axis=-1).mean(1)   # (points, t_cam)
        out.append(spread.mean())
    return float(np.mean(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="../results/sparse_queries/drift.json")
    ap.add_argument("--decoders", default=None, help="folder of decoder-only runs; uses the encoder of <runs>/full_s0")
    a = ap.parse_args()
    test = D.test_set(100)
    model = D.D4RTSmall()
    rgb_pad = np.pad(test[0]["rgb"], ((0, 0), (0, 0), (D.PAD, D.PAD), (D.PAD, D.PAD), (0, 0)), mode="edge")
    z = np.zeros((1, 4), np.int64)
    template = model.init(jax.random.PRNGKey(0), test[0]["rgb"][:1] / 255.0, np.zeros((1, 4, 2), np.float32), z, z, z,
                          np.zeros((1, 4, 3 * (2 * D.PAD + 1) ** 2), np.float32))
    fwd = jax.jit(lambda p, v, *q: model.apply(p, v, *q))
    res = defaultdict(dict)
    if a.decoders:                                                # fresh decoders on the frozen all-36 encoder
        enc = serialization.from_bytes(template, open(f"{a.runs}/full_s0_params.msgpack", "rb").read())["params"]["encoder"]
        for f in sorted(Path(a.decoders).glob("*_decoder.msgpack")):
            design, seed = f.name[:-len("_decoder.msgpack")].split("_dec-")[1].rsplit("_s", 1)
            dec = serialization.from_bytes(template["params"]["decoder"], f.read_bytes())
            res[design][int(seed)] = drift({"params": {"encoder": enc, "decoder": dec}}, test, model, fwd, rgb_pad)
            print("decoder", design, seed, round(res[design][int(seed)], 3), flush=True)
    else:
        for f in sorted(Path(a.runs).glob("*_params.msgpack")):
            design, seed = f.name[:-len("_params.msgpack")].rsplit("_s", 1)
            params = serialization.from_bytes(template, f.read_bytes())
            res[design][int(seed)] = drift(params, test, model, fwd, rgb_pad)
            print(design, seed, round(res[design][int(seed)], 3), flush=True)
    summary = {d: {"per_seed": v, "mean": float(np.mean(list(v.values()))),
                   "sd": float(np.std(list(v.values()), ddof=1)) if len(v) > 1 else 0.0} for d, v in res.items()}
    json.dump(summary, open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))
