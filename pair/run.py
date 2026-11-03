"""Collect model responses for every (model, condition, sample).

Mirrors the runner used for the released data: one independent REST call per
sample to the Gemini API `generateContent` endpoint, model-default generation
settings unless the condition sets its own (A0_high), visible text only
(thought parts are excluded).

Usage:
    export GEMINI_API_KEY=...
    python pair/run.py --models gemini-3.5-flash-lite gemma-4-26b-a4b-it \
        --conditions B K A0 A_recite A_demo A_rule --n 10 --out pair/data/responses.jsonl
"""
import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
HERE = Path(__file__).resolve().parent
PROMPTS = json.loads((HERE / "prompts.json").read_text())


def request_body(cond):
    spec = PROMPTS[cond]
    contents = PROMPTS[spec["contents_from"]]["contents"] if "contents_from" in spec else spec["contents"]
    body = {"contents": contents}
    if "generationConfig" in spec:
        body["generationConfig"] = spec["generationConfig"]
    return body


def call(model, body, key, max_attempts=12):
    last = None
    for attempt in range(1, max_attempts + 1):
        t0 = time.time()
        r = requests.post(API.format(model=model), json=body,
                          headers={"x-goog-api-key": key}, timeout=600)
        j = r.json()
        if r.ok and j.get("candidates"):
            c = j["candidates"][0]
            parts = (c.get("content") or {}).get("parts", [])
            return {
                "ok": True,
                "text": "".join(p.get("text", "") for p in parts if not p.get("thought")),
                "finishReason": c.get("finishReason"),
                "modelVersion": j.get("modelVersion"),
                "usage": j.get("usageMetadata"),
                "attempts": attempt,
                "latencyMs": int((time.time() - t0) * 1000),
            }
        last = {"status": r.status_code, "error": (j.get("error") or {}).get("message", "")[:300]}
        if r.status_code in (400, 402, 403, 404):
            break
        wait = min(90, 2 * 2 ** attempt) + random.random() * 1.5
        for d in (j.get("error") or {}).get("details", []):
            if "RetryInfo" in d.get("@type", "") and d.get("retryDelay"):
                wait = max(wait, float(d["retryDelay"].rstrip("s")) + 0.5)
        time.sleep(wait)
    return {"ok": False, "error": last, "attempts": attempt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--conditions", nargs="+", default=["B", "K", "A0"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default=str(HERE / "data" / "responses.jsonl"))
    args = ap.parse_args()
    key = os.environ["GEMINI_API_KEY"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in out.read_text().splitlines() if l.strip()}

    with out.open("a") as f:
        for cond in args.conditions:
            body = request_body(cond)
            for model in args.models:
                for i in range(args.n):
                    rid = f"{model}|{cond}|{i}"
                    if rid in done:
                        continue
                    res = call(model, body, key)
                    rec = {"id": rid, "model": model, "cond": cond, "i": i, "source": "api",
                           "ts": datetime.now(timezone.utc).isoformat(), **res}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    print(rid, "ok" if res["ok"] else res["error"])


if __name__ == "__main__":
    main()
