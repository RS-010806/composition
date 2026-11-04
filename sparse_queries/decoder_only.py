"""Train a fresh query decoder on a frozen encoder, once per supervision design and seed.

The encoder comes from a model trained on every time-camera combination. Each
decoder sees only the combinations its design allows (same number of training
queries for every design) and is scored on held-out scenes and combinations.
This is the post-training version of the question: a new decoder on a frozen
video encoder.

python decoder_only.py --encoder runs/full_s0 --designs full rowcol random tiedrow tied row --seeds 0 1 2 --out runs_dec
"""
import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

import d4rt_small as D
import scenes as S


def main(encoder, designs, seeds, steps, n_train, n_test, out, batch=8, Q=128, lr=1e-3):
    data = S.make_dataset(n_train, seed=12345)                          # the same training scenes as the encoder
    rgb_pad = np.pad(data["rgb"], ((0, 0), (0, 0), (D.PAD, D.PAD), (D.PAD, D.PAD), (0, 0)), mode="edge")
    tables = D.pixel_tables(data["ids"])
    test = D.test_set(n_test)
    model = D.D4RTSmall()
    rng0 = np.random.default_rng(0)
    q0 = D.sample_queries(data, rgb_pad, np.arange(2), D.allowed_cells("full", rng0), 4, rng0, "full", tables)
    init = lambda k: model.init(jax.random.PRNGKey(k), data["rgb"][:2] / 255.0, q0["uv"], q0["ts"], q0["tt"], q0["tc"], q0["patch"])
    trained = serialization.from_bytes(init(0), open(f"{encoder}_params.msgpack", "rb").read())
    enc_p = trained["params"]["encoder"]
    enc = jax.jit(lambda v: model.apply({"params": {"encoder": enc_p, "decoder": trained["params"]["decoder"]}}, v, method=D.D4RTSmall.encode))
    F_all = np.concatenate([np.asarray(enc(jnp.asarray(data["rgb"][i:i + 50] / 255.0, jnp.float32))) for i in range(0, n_train, 50)])
    print(f"encoded {n_train} clips with frozen encoder {encoder}", flush=True)

    def loss_fn(dec_p, F, q):
        pred = model.apply({"params": {"encoder": enc_p, "decoder": dec_p}}, F, q["uv"], q["ts"], q["tt"], q["tc"], q["patch"],
                           method=D.D4RTSmall.decode)
        return jnp.mean(jnp.abs(D.squash(pred) - D.squash(q["P"] / D.SCALE)))

    sched = optax.warmup_cosine_decay_schedule(0.0, lr, min(200, steps // 4), steps, lr * 0.02)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-4))

    @jax.jit
    def step(dec_p, state, F, q):
        l, g = jax.value_and_grad(loss_fn)(dec_p, F, q)
        upd, state = opt.update(g, state, dec_p)
        return optax.apply_updates(dec_p, upd), state, l

    Path(out).mkdir(parents=True, exist_ok=True)
    tag = Path(encoder).name
    for seed in seeds:
        for design in designs:
            name = Path(out) / f"enc-{tag}_dec-{design}_s{seed}"
            if Path(f"{name}.json").exists():
                continue
            rng = np.random.default_rng(5000 + seed)
            A = D.allowed_cells(design, np.random.default_rng(777 + seed))
            dec_p = init(100 + seed)["params"]["decoder"]                  # fresh decoder
            state = opt.init(dec_p)
            t0 = time.time()
            for it in range(1, steps + 1):
                clips = rng.integers(0, n_train, size=batch)
                q = D.sample_queries(data, rgb_pad, clips, A, Q, rng, design, tables)
                dec_p, state, l = step(dec_p, state, jnp.asarray(F_all[clips]), {k: jnp.asarray(v) for k, v in q.items() if k != "mov"})
            res = D.evaluate({"params": {"encoder": enc_p, "decoder": dec_p}}, A, test)
            res.update(encoder=tag, design=design, seed=seed, steps=steps, final_loss=float(l), seconds=time.time() - t0)
            json.dump(res, open(f"{name}.json", "w"))
            np.save(f"{name}_A.npy", A)
            open(f"{name}_decoder.msgpack", "wb").write(serialization.to_bytes(dec_p))
            g = res["general_heldout"] if design != "full" else res["general_all"]
            print(f"[{design} s{seed}] held-out moving {g['moving']['epe']:.3f} static {g['static']['epe']:.3f} "
                  f"(seen moving {res['seen_moving']['epe']:.3f}; baseline {res['baseline_general']['moving']['epe']:.3f}) {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, help="prefix of a trained run, e.g. runs/full_s0")
    ap.add_argument("--designs", nargs="+", default=list(D.DESIGNS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_test", type=int, default=100)
    ap.add_argument("--out", default="runs_dec")
    a = ap.parse_args()
    main(a.encoder, a.designs, a.seeds, a.steps, a.n_train, a.n_test, a.out)
