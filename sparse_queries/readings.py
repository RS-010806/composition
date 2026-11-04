"""How does a model trained only on t_tgt = t_cam answer a query with t_tgt != t_cam?

For held-out queries (t_src, t_tgt, t_cam all different), compare the prediction with three answers:
the truth P(t_tgt, t_cam), and the two answers the tied labels cannot tell apart from it,
P(t_tgt, t_tgt) (camera read as the target time) and P(t_cam, t_cam) (target time read as the camera).

python readings.py --run runs/tied_s0 --out ../results/sparse_queries/readings_tied.json
"""
import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

import d4rt_small as D
import scenes as S


def main(run, out, n_test=100):
    data, ts, py, px = D.test_set(n_test)
    model = D.D4RTSmall()
    rgb_pad = np.pad(data["rgb"], ((0, 0), (0, 0), (D.PAD, D.PAD), (D.PAD, D.PAD), (0, 0)), mode="edge")
    T = S.T
    cells = [(g, c) for g in range(T) for c in range(T)]
    M = ts.shape[1]
    z = np.zeros((1, 4), np.int64)
    template = model.init(jax.random.PRNGKey(0), data["rgb"][:1] / 255.0, np.zeros((1, 4, 2), np.float32), z, z, z,
                          np.zeros((1, 4, 3 * (2 * D.PAD + 1) ** 2), np.float32))
    params = serialization.from_bytes(template, open(f"{run}_params.msgpack", "rb").read())
    fwd = jax.jit(lambda p, v, *q: model.apply(p, v, *q))
    TT = np.array([g for g, _ in cells])[None].repeat(M, 0)
    TC = np.array([c for _, c in cells])[None].repeat(M, 0)
    e_true, e_tgt, e_cam, mov_all = [], [], [], []
    for c in range(n_test):
        clip = np.full((M, len(cells)), c)
        TS = np.repeat(ts[c][:, None], len(cells), 1)
        PY, PX = np.repeat(py[c][:, None], len(cells), 1), np.repeat(px[c][:, None], len(cells), 1)
        uv = np.stack([(PX + 0.5) / S.W, (PY + 0.5) / S.H], -1).astype(np.float32)
        pred = np.asarray(fwd(params, jnp.asarray(data["rgb"][c:c + 1] / 255.0, jnp.float32), uv.reshape(1, -1, 2),
                              TS.reshape(1, -1), TT.reshape(1, -1), TC.reshape(1, -1),
                              D.patches(rgb_pad, clip, TS, PY, PX).reshape(1, M * len(cells), -1))).reshape(M, len(cells), 3) * D.SCALE
        P, mov = S.ground_truth(data, clip, TS, PY, PX, TT, TC)
        P_tgt, _ = S.ground_truth(data, clip, TS, PY, PX, TT, TT)          # camera index read as the target time
        P_cam, _ = S.ground_truth(data, clip, TS, PY, PX, TC, TC)          # target time read as the camera index
        general = (TS != TT) & (TS != TC) & (TT != TC)
        e_true.append(np.linalg.norm(pred - P, axis=-1)[general])
        e_tgt.append(np.linalg.norm(pred - P_tgt, axis=-1)[general])
        e_cam.append(np.linalg.norm(pred - P_cam, axis=-1)[general])
        mov_all.append(mov[general])
    e_true, e_tgt, e_cam, mov = (np.concatenate(x) for x in (e_true, e_tgt, e_cam, mov_all))
    res = {}
    for name, m in (("all", np.ones_like(mov, bool)), ("moving", mov), ("static", ~mov)):
        stack = np.stack([e_true[m], e_tgt[m], e_cam[m]], 1)
        closest = np.bincount(stack.argmin(1), minlength=3) / m.sum()
        res[name] = {"n": int(m.sum()), "err_to_truth": float(e_true[m].mean()), "err_to_tgt_reading": float(e_tgt[m].mean()),
                     "err_to_cam_reading": float(e_cam[m].mean()),
                     "closest_share": {"truth": float(closest[0]), "both_as_t_tgt": float(closest[1]), "both_as_t_cam": float(closest[2])}}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main(a.run, a.out)
