"""Re-run the evaluation of saved end-to-end models (adds per-scene errors to their result files).

python reeval.py --runs runs
"""
import argparse
import json
from pathlib import Path

import jax
import numpy as np
from flax import serialization

import d4rt_small as D

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    a = ap.parse_args()
    test = D.test_set(100)
    model = D.D4RTSmall()
    z = np.zeros((1, 4), np.int64)
    template = model.init(jax.random.PRNGKey(0), test[0]["rgb"][:1] / 255.0, np.zeros((1, 4, 2), np.float32), z, z, z,
                          np.zeros((1, 4, 3 * (2 * D.PAD + 1) ** 2), np.float32))
    for f in sorted(Path(a.runs).glob("*_params.msgpack")):
        stem = str(f)[:-len("_params.msgpack")]
        old = json.load(open(stem + "_eval.json"))
        params = serialization.from_bytes(template, f.read_bytes())
        res = D.evaluate(params, np.load(stem + "_A.npy"), test)
        res.update({k: old[k] for k in ("design", "seed", "steps") if k in old})
        assert abs(res["general_all"]["static"]["epe"] - old["general_all"]["static"]["epe"]) < 1e-4, "evaluation changed"
        json.dump(res, open(stem + "_eval.json", "w"))
        print("re-evaluated", stem, flush=True)
