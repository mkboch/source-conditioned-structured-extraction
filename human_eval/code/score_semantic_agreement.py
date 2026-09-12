#!/usr/bin/env python3
"""H1 scoring: human-semantic Jaccard per model/task/slide/source-pair, plus
inter-annotator agreement on cross-source equivalence decisions.

Deblinds via the sealed mapping. Run ONLY after both first passes are validated.

Metric policy mirrors the automated study:
  - a semantic cluster is present in a source if >=1 item from that source is in it;
  - set Jaccard over cluster IDs;
  - co-empty comparisons are EXCLUDED, never scored 1.0 (Hard Rule 12);
  - sources whose output was invalid are excluded, never treated as empty (Hard Rule 11).
"""
from __future__ import annotations
import argparse, json, random, statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path
HE = Path(__file__).resolve().parents[1]
PAIRS = (("MM","TX"),("MM","IMG"),("TX","IMG"))

def load(annot):
    f = HE / f"completed/annotator_{annot}/annotations_annotator_{annot}.json"
    ann = json.loads(f.read_text())
    smap = json.loads((HE / f"sealed_mapping/annotator_{annot}_mapping.json").read_text())
    meta = {i["item_id"]: i for i in smap["items"]}
    # (model,task,lecture,slide) -> source -> set(global cluster ids)
    cells = defaultdict(lambda: defaultdict(set))
    itemcl = {}
    for pid, rec in ann.get("panels", {}).items():
        for iid, lab in (rec.get("clusters") or {}).items():
            lab = (lab or "").strip()
            if not lab or iid not in meta: continue
            m = meta[iid]
            gid = f"{pid}::{lab}"                      # cluster ids are panel-local
            cells[(m["model"], m["task"], m["lecture"], m["slide_id"])][m["source"]].add(gid)
            itemcl[iid] = gid
    return ann, meta, cells, itemcl

def jac(a, b):
    u = a | b
    return None if not u else len(a & b) / len(u)

def cluster_boot(vals, seed=20260901, n=10000):
    """Lecture-cluster bootstrap. Returns mean and 95% CI, plus a stability flag."""
    if not vals: return {"mean": None, "ci_low": None, "ci_high": None, "n": 0, "n_clusters": 0}
    by = defaultdict(list)
    for lec, v in vals: by[lec].append(v)
    lecs = sorted(by); rng = random.Random(seed); ms = []
    for _ in range(n):
        pool = []
        for _ in range(len(lecs)): pool.extend(by[lecs[rng.randrange(len(lecs))]])
        if pool: ms.append(sum(pool)/len(pool))
    ms.sort()
    q = lambda p: ms[min(int(p*(len(ms)-1)), len(ms)-1)] if ms else None
    flat = [v for _, v in vals]
    out = {"mean": statistics.mean(flat), "ci_low": q(.025), "ci_high": q(.975),
           "n": len(flat), "n_clusters": len(lecs)}
    if len(lecs) < 10 or len(flat) < 15:
        out["stability_warning"] = (f"only {len(lecs)} lecture clusters / {len(flat)} slides; "
            "the clustered interval is unstable and is reported as-is rather than silently "
            "replaced by a slide-level bootstrap")
    return out

def equiv_decisions(cells_src, itemcl, meta):
    """Cross-source binary equivalence decisions: for each cross-source item pair within a
    panel, did the annotator place them in the same cluster?"""
    bypanel = defaultdict(list)
    for iid, gid in itemcl.items():
        m = meta[iid]; bypanel[(m["model"], m["task"], m["lecture"], m["slide_id"])].append((iid, m["source"], gid))
    dec = {}
    for cell, items in bypanel.items():
        for (i1, s1, g1), (i2, s2, g2) in combinations(sorted(items), 2):
            if s1 == s2: continue
            dec[(i1, i2)] = 1 if g1 == g2 else 0
    return dec

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path,
        default=HE/"results_semantic_agreement.json"); a = ap.parse_args()
    A = load("A"); B = load("B")
    out = {"per_annotator": {}, "inter_annotator": {}}

    for tag, (ann, meta, cells, itemcl) in (("A", A), ("B", B)):
        rows = defaultdict(list)
        for (model, task, lec, slide), src in cells.items():
            for l, r in PAIRS:
                if l not in src or r not in src: continue     # invalid source excluded
                j = jac(src[l], src[r])
                if j is None: continue                        # co-empty excluded, never 1.0
                rows[(model, task, f"{l}-{r}")].append((lec, j))
        out["per_annotator"][tag] = {
            f"{m}|{t}|{p}": cluster_boot(v) for (m, t, p), v in sorted(rows.items())}

    dA = equiv_decisions(None, A[3], A[1]); dB = equiv_decisions(None, B[3], B[1])
    # map B's item ids onto A's by (model,task,lecture,slide,source,normalized_key)
    keyA = {i["item_id"]: (i["model"],i["task"],i["lecture"],i["slide_id"],i["source"],i["normalized_key"]) for i in
            json.loads((HE/"sealed_mapping/annotator_A_mapping.json").read_text())["items"]}
    keyB = {i["item_id"]: (i["model"],i["task"],i["lecture"],i["slide_id"],i["source"],i["normalized_key"]) for i in
            json.loads((HE/"sealed_mapping/annotator_B_mapping.json").read_text())["items"]}
    b_by_key = {v: k for k, v in keyB.items()}
    both = {}
    for (i1, i2), va in dA.items():
        j1, j2 = b_by_key.get(keyA.get(i1)), b_by_key.get(keyA.get(i2))
        if j1 is None or j2 is None: continue
        vb = dB.get((j1, j2), dB.get((j2, j1)))
        if vb is None: continue
        both[(i1, i2)] = (va, vb)
    n = len(both)
    if n:
        agree = sum(1 for v in both.values() if v[0] == v[1])
        tp = sum(1 for v in both.values() if v == (1,1)); fp = sum(1 for v in both.values() if v == (0,1))
        fn = sum(1 for v in both.values() if v == (1,0)); tn = sum(1 for v in both.values() if v == (0,0))
        po = agree/n
        pA1 = (tp+fn)/n; pB1 = (tp+fp)/n
        pe = pA1*pB1 + (1-pA1)*(1-pB1)
        kappa = (po-pe)/(1-pe) if pe != 1 else None
        prec = tp/(tp+fp) if tp+fp else None
        rec  = tp/(tp+fn) if tp+fn else None
        f1 = 2*prec*rec/(prec+rec) if prec and rec else None
        out["inter_annotator"] = {
            "n_comparable_cross_source_pairs": n,
            "raw_agreement": po, "cohens_kappa": kappa,
            "positive_equivalence_decisions_A": tp+fn, "positive_equivalence_decisions_B": tp+fp,
            "positive_class_precision_B_vs_A": prec, "positive_class_recall_B_vs_A": rec,
            "positive_class_f1": f1,
            "confusion": {"both_equivalent": tp, "A_only": fn, "B_only": fp, "both_not": tn},
            "note": ("Kappa is unstable when equivalent pairs are sparse; raw agreement, the "
                     "positive-class counts and F1 must be read alongside it, never kappa alone.")}
    else:
        out["inter_annotator"] = {"n_comparable_cross_source_pairs": 0,
            "note": "no comparable cross-source item pairs could be matched between annotators"}
    a.out.write_text(json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")
    print("wrote", a.out)

if __name__ == "__main__":
    raise SystemExit(main())
