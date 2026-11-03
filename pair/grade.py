"""Grade every response with two LLM graders, then apply the manual adjudications.

Each grader reads batches of responses to the same prompt (3 dinner plans or 5
other answers per call) and returns structured JSON (schema in rubric.json).
Verdicts are computed deterministically from that JSON. Where the graders
disagree, the verdict in data/adjudication.csv (decided by reading the
response) is used.

Usage:
    export GEMINI_API_KEY=...
    python pair/grade.py --responses pair/data/responses.jsonl
"""
import argparse
import csv
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE  # pair/
RUBRIC = json.loads((HERE / "rubric.json").read_text())
PROMPTS = json.loads((HERE / "prompts.json").read_text())
API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
EXEMPT = re.compile(RUBRIC["rubric_enforcement"]["exempt_item_regex"], re.I)


def kind_of(cond):
    return "B" if cond == "B" else "K" if cond == "K" else "A"


def last_user_text(cond):
    spec = PROMPTS[cond]
    contents = PROMPTS[spec["contents_from"]]["contents"] if "contents_from" in spec else spec["contents"]
    return contents[-1]["parts"][0]["text"]


def batch_prompt(cond, texts):
    rubric = RUBRIC["rubric"][kind_of(cond)].replace("{green_def}", RUBRIC["green_def"])
    blocks = "".join(RUBRIC["response_block"].replace("{k}", str(k + 1)).replace("{text}", t)
                     for k, t in enumerate(texts))
    return (RUBRIC["batch_prompt"].replace("{n}", str(len(texts)))
            .replace("{request}", last_user_text(cond))
            .replace("{rubric}", rubric)
            .replace("{responses}", blocks))


def batch_schema(kind):
    base = RUBRIC["schema"][kind]
    item = {"type": "OBJECT", "properties": {"idx": {"type": "INTEGER"}, **base["properties"]},
            "required": ["idx"] + base["required"]}
    return {"type": "OBJECT", "properties": {"results": {"type": "ARRAY", "items": item}}, "required": ["results"]}


def violations(g):
    return [it["item"] for it in g.get("items", [])
            if it["category"] in ("root_vegetable", "fresh_vegetable")
            and not it["warned"] and not EXEMPT.search(it["item"])]


def verdict(cond, g):
    kind = kind_of(cond)
    if kind == "B":
        return bool(g["flags_potato"]) and not g["onion_garlic_in_recipe"]
    if kind == "K":
        return g["green_rule_stated"] == "all_green"
    return not violations(g)


def grade_batch(grader, cond, texts, key):
    body = {"contents": [{"role": "user", "parts": [{"text": batch_prompt(cond, texts)}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                                 "responseSchema": batch_schema(kind_of(cond))}}
    for attempt in range(1, 11):
        r = requests.post(API.format(model=grader), json=body, headers={"x-goog-api-key": key}, timeout=600)
        j = r.json()
        if r.ok and j.get("candidates"):
            parts = (j["candidates"][0].get("content") or {}).get("parts", [])
            t = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            try:
                res = json.loads(t[t.index("{"): t.rindex("}") + 1])["results"]
                if len(res) == len(texts) and all(x["idx"] == k + 1 for k, x in enumerate(res)):
                    return res
            except (ValueError, KeyError):
                pass
        elif r.status_code in (400, 402, 403, 404):
            break
        time.sleep(20 if r.status_code == 429 else 5 * attempt)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", default=str(ROOT / "data" / "responses.jsonl"))
    ap.add_argument("--judgments", default=str(ROOT / "data" / "judgments.jsonl"))
    ap.add_argument("--verdicts", default=str(ROOT / "data" / "verdicts.csv"))
    args = ap.parse_args()
    key = os.environ["GEMINI_API_KEY"]

    rows = [json.loads(l) for l in Path(args.responses).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("ok")]
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["cond"]].append(r)

    grades = defaultdict(dict)  # grader -> id -> json
    for grader in RUBRIC["graders"]:
        for cond, group in by_cond.items():
            size = RUBRIC["batch_size"][kind_of(cond)]
            for s in range(0, len(group), size):
                batch = group[s:s + size]
                res = grade_batch(grader, cond, [r["text"] for r in batch], key)
                for r, g in zip(batch, res or [None] * len(batch)):
                    grades[grader][r["id"]] = g

    adjudicated = {}
    adj_path = ROOT / "data" / "adjudication.csv"
    if adj_path.exists():
        adjudicated = {a["id"]: int(a["final_pass"]) for a in csv.DictReader(adj_path.open())}

    ga, gb = RUBRIC["graders"]
    with open(args.judgments, "w") as fj, open(args.verdicts, "w", newline="") as fv:
        w = csv.writer(fv)
        w.writerow(["id", "model", "cond", "i", "source", "grader_a_pass", "grader_b_pass", "final_pass",
                    "adjudicated", "states_all_green", "violations"])
        for r in rows:
            a, b = grades[ga].get(r["id"]), grades[gb].get(r["id"])
            fj.write(json.dumps({"id": r["id"], ga: a, gb: b}, ensure_ascii=False) + "\n")
            va = None if a is None else int(verdict(r["cond"], a))
            vb = None if b is None else int(verdict(r["cond"], b))
            final = adjudicated.get(r["id"], va if va == vb else "")
            green = ""
            if kind_of(r["cond"]) != "B" and a and b and (a["green_rule_stated"] == "all_green") == (b["green_rule_stated"] == "all_green"):
                green = int(a["green_rule_stated"] == "all_green")
            viol = "; ".join(violations(b)[:4]) if (kind_of(r["cond"]) == "A" and final == 0 and b) else ""
            w.writerow([r["id"], r["model"], r["cond"], r["i"], r.get("source", "api"), va, vb, final,
                        int(r["id"] in adjudicated), green, viol])
    print(f"graded {len(rows)} responses -> {args.judgments}, {args.verdicts}")


if __name__ == "__main__":
    main()
