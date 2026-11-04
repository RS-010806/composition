"""A small D4RT-style model trained with sparse time-camera supervision.

D4RT (Zhang et al., CVPR 2026) encodes a video once and answers independent
queries q = (u, v, t_src, t_tgt, t_cam): the 3D position, at time t_tgt and in
the coordinates of camera t_cam, of the point seen at (u, v) in frame t_src.
Here the same interface is built at small scale:

  encoder  3x3 conv over each frame and its difference from the previous frame,
           8x8 patches, transformer over all frames -> scene tokens F
  query    Fourier(u, v) + E_src[t_src] + E_tgt[t_tgt] + E_cam[t_cam] + embed(9x9 RGB patch)
  decoder  cross-attention to F only (queries never attend to each other) -> P in R^3

D4RT samples t_src, t_tgt, t_cam uniformly and ties t_tgt = t_cam with
probability 0.4, so every time-camera combination is supervised. The designs
below restrict supervision to a subset of the (t_tgt, t_cam) grid for each
t_src, with the same number of training queries, and test the rest.

Usage: python d4rt_small.py --design rowcol --seed 0 --steps 5000 --out runs/
"""
import argparse
import json
import time
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

import scenes as S

T = S.T
DESIGNS = ("full", "rowcol", "random", "tied", "tiedrow", "split", "row")


def allowed_cells(design, rng):
    """Boolean mask A[t_src, t_tgt, t_cam] of supervised combinations."""
    s, g, c = np.meshgrid(np.arange(T), np.arange(T), np.arange(T), indexing="ij")
    if design == "full":
        return np.ones((T, T, T), bool)
    if design == "rowcol":       # one row and one column of each t_src's grid: t_tgt = t_src or t_cam = t_src
        return (g == s) | (c == s)
    if design == "tied":         # camera-frame tracks: the point at time t in camera t
        return g == c
    if design == "tiedrow":      # camera-frame tracks plus posed depth at the source time
        return (g == c) | (g == s)
    if design == "split":        # the same cells as tiedrow, but each clip has only one kind of label (see sample_queries)
        return (g == c) | (g == s)
    if design == "row":          # posed depth only: the point at its own time, in every camera
        return g == s
    if design == "random":       # the same number of cells as rowcol, drawn at random for each t_src
        A = np.zeros((T, T, T), bool)
        for ts in range(T):
            idx = rng.choice(T * T, size=2 * T - 1, replace=False)
            A[ts].flat[idx] = True
        return A
    raise ValueError(design)


# ---------------------------------------------------------------- model

class Block(nn.Module):
    d: int
    heads: int

    @nn.compact
    def __call__(self, x, ctx=None):
        h = nn.LayerNorm()(x)
        kv = h if ctx is None else ctx
        x = x + nn.MultiHeadDotProductAttention(num_heads=self.heads, qkv_features=self.d)(h, kv)
        h = nn.LayerNorm()(x)
        return x + nn.Dense(self.d)(nn.gelu(nn.Dense(4 * self.d)(h)))


class Encoder(nn.Module):
    d: int = 96
    heads: int = 4
    layers: int = 3
    patch: int = 8

    @nn.compact
    def __call__(self, video):                      # (B, T, H, W, 3) in [0, 1]
        B, Tn, Hh, Ww, _ = video.shape
        p = self.patch
        # each frame also sees its difference from the previous frame (D4RT's tubelets span 2 frames),
        # and a 3x3 convolution keeps sub-patch detail before 8x8 patching
        prev = jnp.concatenate([video[:, :1], video[:, :-1]], axis=1)
        x = jnp.concatenate([video - 0.5, video - prev], axis=-1)
        x = nn.gelu(nn.Conv(32, (3, 3))(x.reshape(B * Tn, Hh, Ww, -1)))
        C = x.shape[-1]
        x = x.reshape(B, Tn, Hh // p, p, Ww // p, p, C).transpose(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(B, Tn, (Hh // p) * (Ww // p), p * p * C)
        x = nn.Dense(self.d)(x)
        x = x + self.param("pos", nn.initializers.normal(0.02), (1, 1, x.shape[2], self.d))
        x = x + self.param("time", nn.initializers.normal(0.02), (1, Tn, 1, self.d))
        x = x.reshape(B, -1, self.d)
        for _ in range(self.layers):
            x = Block(self.d, self.heads)(x)
        return nn.LayerNorm()(x)


def fourier(uv, n=5):
    f = 2.0 ** jnp.arange(n) * jnp.pi
    a = uv[..., None] * f
    return jnp.concatenate([jnp.sin(a), jnp.cos(a)], -1).reshape(*uv.shape[:-1], -1)


class Decoder(nn.Module):
    d: int = 96
    heads: int = 4
    layers: int = 3

    @nn.compact
    def __call__(self, F, uv, ts, tt, tc, patch):   # F (B, N, d); per query (B, Q, ...)
        q = nn.Dense(self.d)(fourier(uv))
        q = q + nn.Embed(T, self.d, name="E_src")(ts) + nn.Embed(T, self.d, name="E_tgt")(tt) + nn.Embed(T, self.d, name="E_cam")(tc)
        q = q + nn.Dense(self.d)(patch - 0.5)
        for _ in range(self.layers):
            q = Block(self.d, self.heads)(q, ctx=F)  # cross-attention only: queries are independent
        return nn.Dense(3)(nn.LayerNorm()(q))


class D4RTSmall(nn.Module):
    def setup(self):
        self.encoder, self.decoder = Encoder(), Decoder()

    def __call__(self, video, uv, ts, tt, tc, patch):
        return self.decoder(self.encoder(video), uv, ts, tt, tc, patch)

    def encode(self, video):
        return self.encoder(video)

    def decode(self, F, uv, ts, tt, tc, patch):
        return self.decoder(F, uv, ts, tt, tc, patch)


# ---------------------------------------------------------------- queries

PAD = 4


def patches(rgb_pad, clip, ts, py, px):
    """9x9 RGB patches around (py, px) of frame ts; rgb_pad is padded by PAD pixels."""
    oy, ox = np.mgrid[0:2 * PAD + 1, 0:2 * PAD + 1]
    P = rgb_pad[clip[..., None, None], ts[..., None, None], py[..., None, None] + oy, px[..., None, None] + ox]
    return P.reshape(*py.shape, -1).astype(np.float32) / 255.0


def pixel_tables(ids):
    """For every (clip, frame): pixel indices on moving objects and on any surface, listed first, with counts."""
    flat = ids.reshape(ids.shape[0], T, -1)
    out = []
    for mask in (S.is_moving(flat), flat != S.SKY):
        out.append((np.argsort(~mask, axis=-1, kind="stable").astype(np.int32), mask.sum(-1)))
    return out


def sample_queries(data, rgb_pad, clips, A, Q, rng, design, tables=None):
    """Q training queries per clip, restricted to the allowed (t_src, t_tgt, t_cam) cells."""
    B = len(clips)
    clip = np.repeat(clips[:, None], Q, 1)
    ts = rng.integers(0, T, size=(B, Q))
    if design == "full":                            # D4RT's rule: uniform, with t_tgt = t_cam forced w.p. 0.4
        tt = rng.integers(0, T, size=(B, Q))
        tc = np.where(rng.random((B, Q)) < 0.4, tt, rng.integers(0, T, size=(B, Q)))
    elif design == "split":                         # even-numbered clips: camera-frame tracks only (t_tgt = t_cam);
        tc = rng.integers(0, T, size=(B, Q))        # odd-numbered clips: posed depth only (t_tgt = t_src)
        tt = np.where((np.asarray(clips) % 2 == 0)[:, None], tc, ts)
    else:
        flat = A.reshape(T, T * T)
        counts = flat.sum(1)
        order = np.argsort(~flat, axis=1, kind="stable")              # allowed cells first, per t_src
        pick = order[ts, (rng.random((B, Q)) * counts[ts]).astype(np.int64)]
        tt, tc = pick // T, pick % T
    # pixels: half on moving objects (where t_tgt matters), half anywhere on the scene
    (mov_ord, mov_cnt), (val_ord, val_cnt) = tables if tables is not None else pixel_tables(data["ids"])
    want_mov = (np.arange(Q)[None, :] % 2 == 0) & (mov_cnt[clip, ts] > 0)
    cnt = np.where(want_mov, mov_cnt[clip, ts], val_cnt[clip, ts])
    k = (rng.random((B, Q)) * cnt).astype(np.int64)
    pix = np.where(want_mov, mov_ord[clip, ts, k], val_ord[clip, ts, k])
    py, px = pix // S.W, pix % S.W
    P, mov = S.ground_truth(data, clip, ts, py, px, tt, tc)
    uv = np.stack([(px + 0.5) / S.W, (py + 0.5) / S.H], -1).astype(np.float32)
    return dict(uv=uv, ts=ts, tt=tt, tc=tc, patch=patches(rgb_pad, clip, ts, py, px), P=P, mov=mov)


SCALE = 5.0                                         # typical depth; targets are divided by it


def squash(x):                                      # D4RT's sign(x) log(1 + |x|), which damps far points
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def loss_fn(params, model, video, q):
    pred = model.apply(params, video, q["uv"], q["ts"], q["tt"], q["tc"], q["patch"])
    return jnp.mean(jnp.abs(squash(pred) - squash(q["P"] / SCALE)))


def train(design, seed, steps, n_train, out_dir, batch=8, Q=128, lr=1e-3, log_every=250):
    rng = np.random.default_rng(1000 + seed)
    data = S.make_dataset(n_train, seed=12345)      # the same training scenes for every design and seed
    rgb_pad = np.pad(data["rgb"], ((0, 0), (0, 0), (PAD, PAD), (PAD, PAD), (0, 0)), mode="edge")
    A = allowed_cells(design, np.random.default_rng(777 + seed))
    tables = pixel_tables(data["ids"])
    model = D4RTSmall()
    q0 = sample_queries(data, rgb_pad, np.arange(2), A, 4, rng, design, tables)
    params = model.init(jax.random.PRNGKey(seed), data["rgb"][:2] / 255.0, q0["uv"], q0["ts"], q0["tt"], q0["tc"], q0["patch"])
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, min(200, steps // 4), steps, lr * 0.02)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-4))
    state = opt.init(params)

    @jax.jit
    def step(params, state, video, q):
        l, g = jax.value_and_grad(loss_fn)(params, model, video, q)
        upd, state = opt.update(g, state, params)
        return optax.apply_updates(params, upd), state, l

    t0, hist = time.time(), []
    for it in range(1, steps + 1):
        clips = rng.integers(0, n_train, size=batch)
        q = sample_queries(data, rgb_pad, clips, A, Q, rng, design, tables)
        qj = {k: jnp.asarray(v) for k, v in q.items() if k != "mov"}
        params, state, l = step(params, state, jnp.asarray(data["rgb"][clips] / 255.0, jnp.float32), qj)
        if it % log_every == 0 or it == 1:
            hist.append((it, float(l)))
            print(f"[{design} s{seed}] step {it} loss {float(l):.4f} ({time.time() - t0:.0f}s)", flush=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{design}_s{seed}_A.npy", A)
    with open(out / f"{design}_s{seed}_params.msgpack", "wb") as f:
        from flax import serialization
        f.write(serialization.to_bytes(params))
    json.dump({"design": design, "seed": seed, "steps": steps, "n_train": n_train, "batch": batch, "Q": Q,
               "lr": lr, "loss": hist, "train_seconds": time.time() - t0}, open(out / f"{design}_s{seed}_train.json", "w"))
    return params, A


# ---------------------------------------------------------------- evaluation

N_PTS = 12                                          # source points per (clip, t_src): 6 on moving objects, 6 static


def test_set(n_clips=100, seed=999):
    """Held-out scenes and, for each frame, 6 moving and 6 static source pixels. Every point is
    queried at all 64 (t_tgt, t_cam) combinations."""
    data = S.make_dataset(n_clips, seed=seed)
    rng = np.random.default_rng(seed)
    ts, py, px = (np.zeros((n_clips, T * N_PTS), np.int64) for _ in range(3))
    for c in range(n_clips):
        for t in range(T):
            ids = data["ids"][c, t]
            mov = np.argwhere(S.is_moving(ids))
            sta = np.argwhere((ids != S.SKY) & ~S.is_moving(ids))
            pts = []
            for pool, k in ((mov, N_PTS // 2), (sta, N_PTS // 2)):
                pool = pool if len(pool) else sta
                pts.append(pool[rng.choice(len(pool), size=k, replace=len(pool) < k)])
            pts = np.concatenate(pts)
            sl = slice(t * N_PTS, (t + 1) * N_PTS)
            ts[c, sl], py[c, sl], px[c, sl] = t, pts[:, 0], pts[:, 1]
    return data, ts, py, px


def evaluate(params, A, test):
    data, ts, py, px = test
    n_clips, M = ts.shape
    model = D4RTSmall()
    tt_cell, tc_cell = np.meshgrid(np.arange(T), np.arange(T), indexing="ij")
    tt_cell, tc_cell = tt_cell.ravel(), tc_cell.ravel()                 # 64 cells
    rgb_pad = np.pad(data["rgb"], ((0, 0), (0, 0), (PAD, PAD), (PAD, PAD), (0, 0)), mode="edge")
    enc = jax.jit(lambda p, v: model.apply(p, v, method=D4RTSmall.encode))
    dec = jax.jit(lambda p, F, *q: model.apply(p, F, *q, method=D4RTSmall.decode))
    err = np.zeros((n_clips, M, T * T), np.float32)
    base = np.zeros((n_clips, M, T * T), np.float32)
    moving = np.zeros((n_clips, M), bool)
    for c in range(n_clips):
        clip = np.full((M, T * T), c)
        TS = np.repeat(ts[c][:, None], T * T, 1)
        PY, PX = np.repeat(py[c][:, None], T * T, 1), np.repeat(px[c][:, None], T * T, 1)
        TT, TC = np.broadcast_to(tt_cell, (M, T * T)), np.broadcast_to(tc_cell, (M, T * T))
        P, mov = S.ground_truth(data, clip, TS, PY, PX, TT, TC)
        P0, _ = S.ground_truth(data, clip, TS, PY, PX, TS, TS)          # "nothing changes" baseline
        uv = np.stack([(PX + 0.5) / S.W, (PY + 0.5) / S.H], -1).astype(np.float32)
        F = enc(params, jnp.asarray(data["rgb"][c:c + 1] / 255.0, jnp.float32))
        pred = dec(params, F, jnp.asarray(uv.reshape(1, -1, 2)), jnp.asarray(TS.reshape(1, -1)),
                   jnp.asarray(TT.reshape(1, -1)), jnp.asarray(TC.reshape(1, -1)),
                   jnp.asarray(patches(rgb_pad, clip, TS, PY, PX).reshape(1, M * T * T, -1)))
        pred = np.asarray(pred).reshape(M, T * T, 3) * SCALE
        err[c] = np.linalg.norm(pred - P, axis=-1)
        base[c] = np.linalg.norm(P0 - P, axis=-1)
        moving[c] = mov[:, 0]
    return summarize(err, base, moving, ts, A)


def summarize(err, base, moving, ts, A):
    n_clips, M, _ = err.shape
    TT, TC = np.meshgrid(np.arange(T), np.arange(T), indexing="ij")
    TT, TC = TT.ravel()[None, None], TC.ravel()[None, None]
    TS = ts[:, :, None]
    seen = A[TS, TT, TC]
    general = (TS != TT) & (TS != TC) & (TT != TC)
    mov = np.broadcast_to(moving[:, :, None], err.shape)
    track = general & (TC == 0)                                         # 3D tracks in camera 0's frame
    def stats(mask, e=err):
        e = e[mask]
        return {"n": int(e.size), "epe": float(e.mean()) if e.size else None,
                "acc25": float((e < 0.25).mean()) if e.size else None, "acc50": float((e < 0.5).mean()) if e.size else None}
    held = general & ~seen
    out = {
        "seen": stats(seen), "unseen": stats(~seen),
        "general_heldout": {"all": stats(held), "moving": stats(held & mov), "static": stats(held & ~mov)},
        "general_all": {"all": stats(general), "moving": stats(general & mov), "static": stats(general & ~mov)},
        "track_cam0_heldout_moving": stats(track & ~seen & mov),
        "baseline_general": {"moving": stats(general & mov, base), "static": stats(general & ~mov, base)},
        "seen_moving": stats(seen & mov), "seen_static": stats(seen & ~mov),
    }
    # per-cell error maps for t_src = 0 (rows t_tgt, columns t_cam)
    for name, m in (("map_moving", moving), ("map_static", ~moving)):
        sel = (ts == 0) & m
        out[name] = err[sel].mean(0).reshape(T, T).tolist() if sel.any() else None
    # per test scene: mean error on general-position combinations that were not supervised
    # (all of them for the sparse label sets; for the all-36 model they were all supervised)
    target = held if (~seen & general).any() else general
    for name, m in (("per_scene_static", target & ~mov), ("per_scene_moving", target & mov),
                    ("per_scene_baseline_static", general & ~mov)):
        e = base if "baseline" in name else err
        out[name] = [float(e[c][m[c]].mean()) if m[c].any() else None for c in range(n_clips)]
    out["supervised_cells_per_tsrc"] = int(A[0].sum())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", choices=DESIGNS, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_test", type=int, default=100)
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()
    params, A = train(a.design, a.seed, a.steps, a.n_train, a.out)
    t0 = time.time()
    res = evaluate(params, A, test_set(a.n_test))
    res.update(design=a.design, seed=a.seed, steps=a.steps, eval_seconds=time.time() - t0)
    json.dump(res, open(Path(a.out) / f"{a.design}_s{a.seed}_eval.json", "w"))
    g = res["general_heldout"]
    print(f"[{a.design} s{a.seed}] held-out general EPE moving {g['moving']['epe']} static {g['static']['epe']} "
          f"| seen EPE {res['seen']['epe']:.3f} | baseline moving {res['baseline_general']['moving']['epe']:.3f}", flush=True)
