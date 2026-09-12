#!/usr/bin/env python3
"""Symmetric truncation diagnostic: how often does a model hit the frozen
max_new_tokens=512 budget and emit an incomplete JSON structure?

Applied identically to every model. Truncation signature = parse failed AND the raw output
does not end in '}' or ']' (i.e. it stops mid-structure rather than emitting malformed JSON).
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from v3_metrics import read_jsonl, is_valid

def diag(rows):
    cells = defaultdict(lambda: {"n":0,"invalid":0,"truncated":0,"malformed":0,"schema_bad":0})
    for r in rows:
        c = cells[f"{r.get('mode')}_{r.get('task')}"]; c["n"] += 1
        if is_valid(r): continue
        c["invalid"] += 1
        raw = (r.get("raw_output") or "").rstrip()
        if r.get("json_parse_success"): c["schema_bad"] += 1
        elif raw and not raw.endswith(("}", "]")): c["truncated"] += 1
        else: c["malformed"] += 1
    for c in cells.values():
        c["truncation_pct_of_all"] = round(100.0*c["truncated"]/c["n"], 2) if c["n"] else None
        c["truncation_pct_of_invalid"] = round(100.0*c["truncated"]/c["invalid"], 1) if c["invalid"] else None
    return dict(cells)

ap = argparse.ArgumentParser(); ap.add_argument("--spec", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True); a = ap.parse_args()
spec = json.loads(a.spec.read_text())
res = {m: diag(read_jsonl(Path(p))) for m, p in spec["models"].items()}
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
print(f"{'model':30} {'cell':16} {'n':>5} {'invalid':>8} {'trunc':>6} {'trunc%':>7} {'malformed':>10} {'schemaBad':>10}")
for m in sorted(res):
    for k in sorted(res[m]):
        d = res[m][k]
        print(f"{m:30} {k:16} {d['n']:5} {d['invalid']:8} {d['truncated']:6} "
              f"{str(d['truncation_pct_of_all']):>7} {d['malformed']:10} {d['schema_bad']:10}")
