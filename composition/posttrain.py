"""Small CNN trained from scratch: does composition come for free, and can post-training add it?

1. Train a small CNN with two heads (color, shape) on 12,000 images of the 26 shown compounds,
   either balanced (p_home < 0) or correlated: each color appears with its "home" shape with
   probability p_home, the way web images pair attributes with typical objects. Ten compounds
   are never shown at all.
2. Post-train (fine-tune all weights) on 220 images chosen by one of five designs:
     none        no post-training
     correlated  220 more images from the pretraining mix
     random11    20 images each of 11 compounds drawn at random from the 26 shown compounds
     cross       20 images each of the 11 cross compounds (one full row plus one full column)
     all26       220 images spread over all 26 shown compounds
3. Score on fresh images of the 10 never-shown compounds. A compound is right only if both
   heads are right. A "shortcut" error predicts the color's home shape.

Usage (from the repo root):
    python composition/posttrain.py --seeds 5 --p_home -1 0.95 --designs none correlated cross all26 --out results/scratch_cnn.json
"""
import argparse
import json
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

import shapes as sh

TRAIN_PER, EVAL_PER = 160, 80  # renders per compound reserved for training and for evaluation


class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        for width in (32, 64, 64):
            x = nn.relu(nn.Conv(width, (3, 3))(x))
            x = nn.max_pool(x, (2, 2), (2, 2))
        h = nn.relu(nn.Dense(128)(x.reshape((x.shape[0], -1))))
        return nn.Dense(sh.C, name="color")(h), nn.Dense(sh.S, name="shape")(h)


model = CNN()


def loss_fn(params, x, yc, ys):
    lc, ls = model.apply({"params": params}, x)
    return (optax.softmax_cross_entropy_with_integer_labels(lc, yc).mean()
            + optax.softmax_cross_entropy_with_integer_labels(ls, ys).mean())


@partial(jax.jit, static_argnums=(5,))
def step(params, opt_state, x, yc, ys, tx):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, yc, ys)
    updates, opt_state = tx.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss


@jax.jit
def predict(params, x):
    lc, ls = model.apply({"params": params}, x)
    return lc.argmax(-1), ls.argmax(-1)


def fit(params, X, yc, ys, rng, epochs, lr, batch):
    tx = optax.adam(lr)
    opt_state = tx.init(params)
    n = len(X)
    for _ in range(epochs):
        order = rng.permutation(n)
        for i in range(0, n - batch + 1, batch):
            idx = order[i:i + batch]
            params, opt_state, _ = step(params, opt_state, X[idx], yc[idx], ys[idx], tx)
    return params


def sample_correlated(n, p_home, rng):
    """Compounds from the correlated pretraining mix (never a held-out compound).
    p_home < 0 means uniform over the 26 shown compounds (no correlation)."""
    if p_home < 0:
        return [sh.SEEN[j] for j in rng.integers(len(sh.SEEN), size=n)]
    out = []
    for _ in range(n):
        c = int(rng.integers(sh.C))
        if rng.random() < p_home:
            out.append((c, sh.HOME[c]))
        else:
            options = [s for s in range(sh.S) if s != sh.HOME[c] and (c, s) not in sh.HELD_OUT]
            out.append((c, int(rng.choice(options))))
    return out


def gather(bank, compounds, rng, lo, hi):
    idx = rng.integers(lo, hi, len(compounds))
    X = np.stack([bank[c, s, i] for (c, s), i in zip(compounds, idx)])
    yc = np.array([c for c, _ in compounds], dtype=np.int32)
    ys = np.array([s for _, s in compounds], dtype=np.int32)
    return X, yc, ys


def evaluate(params, bank, compounds):
    X = np.concatenate([bank[c, s, TRAIN_PER:] for c, s in compounds])
    yc = np.repeat([c for c, _ in compounds], EVAL_PER)
    ys = np.repeat([s for _, s in compounds], EVAL_PER)
    pc, ps = (np.asarray(a) for a in predict(params, X))
    home_of_true_color = np.array([sh.HOME[c] for c in yc])
    return {
        "compound_acc": float(np.mean((pc == yc) & (ps == ys))),
        "color_acc": float(np.mean(pc == yc)),
        "shape_acc": float(np.mean(ps == ys)),
        "shortcut_rate": float(np.mean(ps == home_of_true_color)),
    }


def design_compounds(name, rng, p_home, budget=220):
    per = budget // 11
    if name == "correlated":
        return sample_correlated(budget, p_home, rng)
    if name == "cross":
        return [k for k in sh.CROSS for _ in range(per)]
    if name == "random11":
        pick = rng.choice(len(sh.SEEN), 11, replace=False)
        return [sh.SEEN[j] for j in pick for _ in range(per)]
    if name == "all26":
        return [sh.SEEN[j % len(sh.SEEN)] for j in rng.permutation(budget)]
    raise ValueError(name)


def run(seed, p_home, bank, designs, n_pretrain, pre_epochs, post_epochs, budget=220):
    rng = np.random.default_rng(1000 + seed)
    params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1, 32, 32, 3)))["params"]
    X, yc, ys = gather(bank, sample_correlated(n_pretrain, p_home, rng), rng, 0, TRAIN_PER)
    params = fit(params, X, yc, ys, rng, pre_epochs, 2e-3, 128)
    out = {}
    for d in designs:
        drng = np.random.default_rng(hash((seed, p_home, d)) % 2**32)
        if d == "none":
            p = params
        else:
            comps = design_compounds(d, drng, p_home, budget)
            Xp, ycp, ysp = gather(bank, comps, drng, 0, TRAIN_PER)
            p = fit(params, Xp, ycp, ysp, drng, max(1, post_epochs * 220 // budget), 5e-4, 32)
        out[d] = {"held_out": evaluate(p, bank, sh.HELD_OUT), "seen": evaluate(p, bank, sh.SEEN)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/posttrain.json")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--p_home", type=float, nargs="+", default=[0.8, 0.9, 0.95, 0.99])
    ap.add_argument("--designs", nargs="+", default=["none", "correlated", "random11", "cross", "all26"])
    ap.add_argument("--n_pretrain", type=int, default=12000)
    ap.add_argument("--pre_epochs", type=int, default=5)
    ap.add_argument("--post_epochs", type=int, default=30, help="epochs at a 220-image budget; scaled to keep steps fixed")
    ap.add_argument("--budget", type=int, nargs="+", default=[220])
    args = ap.parse_args()

    t0 = time.time()
    bank = sh.bank(TRAIN_PER + EVAL_PER, seed=7)
    print(f"rendered {bank.shape[0] * bank.shape[1] * bank.shape[2]} images in {time.time() - t0:.0f}s", flush=True)
    results = {"config": vars(args), "held_out": sh.HELD_OUT, "cross": sh.CROSS, "runs": []}
    for p_home in args.p_home:
        for budget in args.budget:
            for seed in range(args.seeds):
                t1 = time.time()
                res = run(seed, p_home, bank, args.designs, args.n_pretrain, args.pre_epochs, args.post_epochs, budget)
                results["runs"].append({"p_home": p_home, "budget": budget, "seed": seed, "results": res})
                summary = "  ".join(f"{d}={res[d]['held_out']['compound_acc']:.2f}/sh{res[d]['held_out']['shape_acc']:.2f}" for d in args.designs)
                print(f"p_home={p_home} budget={budget} seed={seed} [{time.time() - t1:.0f}s] held-out: {summary}", flush=True)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
