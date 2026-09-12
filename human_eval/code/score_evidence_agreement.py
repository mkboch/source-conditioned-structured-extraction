#!/usr/bin/env python3
"""H2 inter-annotator scoring for the evidence-attribution task.
Raw agreement, Cohen's kappa, confusion matrix, per-category agreement, uncertain rate.
Run ONLY after both first passes are validated. Adjudicated proportions are computed by
score_adjudicated_results.py, not here.
"""
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
HE = Path(__file__).resolve().parents[1]
CATS = ["both", "transcript_only", "image_only", "neither", "uncertain"]

def load(annot):
    ann = json.loads((HE/f"completed/annotator_{annot}/annotations_annotator_{annot}.json").read_text())
    smap = json.loads((HE/f"sealed_mapping/annotator_{annot}_mapping.json").read_text())
    meta = {e["evidence_id"]: e for e in smap["evidence_items"]}
    out = {}
    for eid, rec in ann.get("evidence", {}).items():
        if eid in meta and rec.get("label"):
            m = meta[eid]
            out[(m["model"], m["task"], m["lecture"], m["slide_id"], m["normalized_key"])] = rec["label"]
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path,
        default=HE/"results_evidence_agreement.json"); a = ap.parse_args()
    A, B = load("A"), load("B")
    keys = sorted(set(A) & set(B))
    n = len(keys)
    conf = defaultdict(int)
    for k in keys: conf[(A[k], B[k])] += 1
    agree = sum(v for (x, y), v in conf.items() if x == y)
    po = agree/n if n else None
    cA, cB = Counter(A[k] for k in keys), Counter(B[k] for k in keys)
    pe = sum((cA[c]/n)*(cB[c]/n) for c in CATS) if n else None
    kappa = (po-pe)/(1-pe) if (po is not None and pe is not None and pe != 1) else None
    per_cat = {}
    for c in CATS:
        a_c = cA[c]; b_c = cB[c]; both_c = conf[(c, c)]
        per_cat[c] = {"A_count": a_c, "B_count": b_c, "both_assigned": both_c,
                      "A_rate": a_c/n if n else None, "B_rate": b_c/n if n else None,
                      "jaccard_of_assignments": both_c/(a_c+b_c-both_c) if (a_c+b_c-both_c) else None}
    by = defaultdict(lambda: {"n": 0, "agree": 0})
    for k in keys:
        g = by[f"{k[0]}|{k[1]}"]; g["n"] += 1; g["agree"] += (A[k] == B[k])
    out = {"n_items_both_annotators": n,
           "n_items_A_only": len(set(A)-set(B)), "n_items_B_only": len(set(B)-set(A)),
           "raw_agreement": po, "cohens_kappa": kappa,
           "confusion_matrix_A_rows_B_cols": {f"A={x}|B={y}": v for (x, y), v in sorted(conf.items())},
           "per_category": per_cat,
           "uncertain_rate": {"A": cA["uncertain"]/n if n else None,
                              "B": cB["uncertain"]/n if n else None},
           "per_model_task_agreement": {k: {"n": v["n"], "raw_agreement": v["agree"]/v["n"]}
                                        for k, v in sorted(by.items())},
           "note": "Kappa alone is not sufficient when categories are unbalanced; read it "
                   "with the confusion matrix, per-category rates and the uncertain rate."}
    a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("wrote", a.out)

if __name__ == "__main__":
    raise SystemExit(main())
