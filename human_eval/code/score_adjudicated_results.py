#!/usr/bin/env python3
"""Final scoring after adjudication.

(a) evidence-category proportions overall and by model/task, with Wilson 95% CIs;
(b) human-semantic Jaccard vs EXACT STRUCTURAL Jaccard per model x task x source pair,
    with lecture-cluster bootstrap CIs;
Reporting rules enforced in the output text:
  * 1 - Jaccard is NEVER called a "semantic error rate";
  * a low exact structural Jaccard is NEVER equated with semantic disagreement.
"""
from __future__ import annotations
import argparse, csv, json, math, random, statistics, sys
from collections import defaultdict
from pathlib import Path
HE = Path(__file__).resolve().parents[1]
ROOT = HE.parent
sys.path.insert(0, str(ROOT/"code"))
from v3_metrics import read_jsonl, is_valid, item_set, _jaccard
CATS = ["both","transcript_only","image_only","neither","uncertain"]
PAIRS = (("MM","TX"),("MM","IMG"),("TX","IMG"))
MODELS = ["InternVL3-14B","Gemma-4-31B","Qwen3-VL-32B-Instruct"]

def wilson(k, n, z=1.959963985):
    if not n: return (None, None)
    p = k/n; d = 1+z*z/n
    c = (p+z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (max(0.0, c-h), min(1.0, c+h))

def cluster_boot(vals, seed=20260901, n=10000):
    if not vals: return {"mean":None,"ci_low":None,"ci_high":None,"n":0,"n_clusters":0}
    by = defaultdict(list)
    for lec, v in vals: by[lec].append(v)
    lecs = sorted(by); rng = random.Random(seed); ms = []
    for _ in range(n):
        pool=[]
        for _ in range(len(lecs)): pool.extend(by[lecs[rng.randrange(len(lecs))]])
        if pool: ms.append(sum(pool)/len(pool))
    ms.sort(); q=lambda p: ms[min(int(p*(len(ms)-1)),len(ms)-1)] if ms else None
    flat=[v for _,v in vals]
    out={"mean":statistics.mean(flat),"ci_low":q(.025),"ci_high":q(.975),
         "n":len(flat),"n_clusters":len(lecs)}
    if len(lecs)<10 or len(flat)<15:
        out["stability_warning"]=(f"only {len(lecs)} lecture clusters / {len(flat)} slides; "
          "clustered interval unstable — reported as-is, not silently replaced")
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--semantic-consensus", type=Path, default=HE/"adjudication/semantic_disagreements.csv")
    ap.add_argument("--evidence-consensus", type=Path, default=HE/"adjudication/evidence_disagreements.csv")
    ap.add_argument("--out", type=Path, default=HE/"results_adjudicated.json")
    a=ap.parse_args()
    for f in (a.semantic_consensus, a.evidence_consensus):
        if not f.exists(): sys.exit(f"MISSING adjudication file: {f}\nRun create_disagreement_packet.py and complete the consensus columns first.")

    # ---- (a) adjudicated evidence proportions ------------------------------------
    annA=json.loads((HE/"completed/annotator_A/annotations_annotator_A.json").read_text())
    smapA=json.loads((HE/"sealed_mapping/annotator_A_mapping.json").read_text())
    metaA={e["evidence_id"]:e for e in smapA["evidence_items"]}
    final={}
    for eid,rec in annA.get("evidence",{}).items():
        if eid in metaA and rec.get("label"):
            m=metaA[eid]; final[(m["model"],m["task"],m["lecture"],m["slide_id"],m["normalized_key"])]=rec["label"]
    n_adj=0
    for r in csv.DictReader(open(a.evidence_consensus)):
        lab=(r.get("consensus_label__FILL") or "").strip()
        if not lab: continue
        if lab not in CATS: sys.exit(f"invalid consensus label {lab!r} in {r.get('case_id')}")
        for k in list(final):
            if k[1]==r["task"] and k[2]==r["lecture"] and k[3]==r["slide_id"]:
                final[k]=lab; n_adj+=1; break
    def props(sel):
        c={x:0 for x in CATS}
        for k,v in final.items():
            if sel(k): c[v]+=1
        n=sum(c.values())
        return {"n":n, **{x:{"count":c[x], "proportion":(c[x]/n if n else None),
                             "wilson_95ci":wilson(c[x],n)} for x in CATS}}
    ev_out={"n_adjudicated_cases_applied":n_adj,
            "overall":props(lambda k:True),
            "by_model":{m:props(lambda k,m=m:k[0]==m) for m in MODELS},
            "by_task":{t:props(lambda k,t=t:k[1]==t) for t in ("concepts","triples")},
            "by_model_task":{f"{m}|{t}":props(lambda k,m=m,t=t:k[0]==m and k[1]==t)
                             for m in MODELS for t in ("concepts","triples")}}

    # ---- (b) human-semantic vs exact structural Jaccard ---------------------------
    cohort=[(r["lecture"],r["slide_id"]) for r in csv.DictReader(open(HE/"frozen_30_slide_manifest.csv"))]
    ck=set(cohort)
    struct=defaultdict(list)
    for m in MODELS:
        recs=defaultdict(dict)
        for r in read_jsonl(ROOT/f"results/final_task_specific/merged/{m}.jsonl"):
            k=(r["lecture"],r["slide_id"])
            if k in ck: recs[(r["task"],k)][r["mode"]]=r
        for (t,k),by in recs.items():
            for l,rr in PAIRS:
                x,y=by.get(l),by.get(rr)
                if not(x and y and is_valid(x) and is_valid(y)): continue
                j=_jaccard(item_set(t,x.get("parsed_output")), item_set(t,y.get("parsed_output")))
                if j is not None: struct[(m,t,f"{l}-{rr}")].append((k[0],j))
    sem=json.loads((HE/"results_semantic_agreement.json").read_text())["per_annotator"] \
        if (HE/"results_semantic_agreement.json").exists() else {}
    comp={}
    for key,vals in sorted(struct.items()):
        m,t,p=key; sk=f"{m}|{t}|{p}"
        s=cluster_boot(vals)
        h=(sem.get("A",{}) or {}).get(sk)
        row={"exact_structural_jaccard":s,"human_semantic_jaccard_annotatorA":h}
        if h and h.get("mean") is not None and s.get("mean") is not None:
            row["absolute_difference_human_minus_exact"]=h["mean"]-s["mean"]
            row["relative_difference"]=((h["mean"]-s["mean"])/s["mean"]) if s["mean"]>0 else \
                "undefined (exact structural Jaccard is 0; relative difference is not meaningful)"
        comp[sk]=row

    out={"protocol":{"note":"Human-semantic Jaccard is computed over annotator-assigned "
                     "semantic clusters; exact structural Jaccard uses the frozen V2 item-set "
                     "definition. Co-empty comparisons are excluded from both, never scored 1.0.",
         "REPORTING_RULES":["1 - Jaccard must NOT be described as a 'semantic error rate'.",
                            "A low exact structural Jaccard must NOT be equated with semantic disagreement.",
                            "Relative difference is undefined where the exact structural Jaccard is 0."]},
         "evidence_adjudicated":ev_out,
         "exact_vs_human_semantic":comp}
    a.out.write_text(json.dumps(out,indent=2,sort_keys=True,default=str)+"\n")
    print("wrote",a.out)

if __name__=="__main__":
    raise SystemExit(main())
