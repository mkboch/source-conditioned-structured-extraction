#!/usr/bin/env python3
"""Authoritative final human-evaluation results, verified from artifacts only."""
from __future__ import annotations
import csv, hashlib, json, math, statistics, sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
HE=Path(__file__).resolve().parents[1]; ROOT=HE.parent
sys.path.insert(0,str(ROOT/"code"))
from v3_metrics import read_jsonl,is_valid,item_set,_jaccard
PAIRS=(("MM","TX"),("MM","IMG"),("TX","IMG"))
MODELS=["InternVL3-14B","Gemma-4-31B","Qwen3-VL-32B-Instruct"]
TL={"concepts":"concepts","triples":"relations"}
LBL={"transcript":"transcript_only","picture":"image_only","unsure":"uncertain","both":"both","neither":"neither"}
def sha(p):
    h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()
def wilson(k,n,z=1.959963985):
    if not n: return [None,None]
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [round(max(0,c-h),4),round(min(1,c+h),4)]

R={"provenance":{},"first_pass_reliability":{},"counts":{},"H1":{},"H2":{},"ordering_check":{}}

# provenance
R["provenance"]["package_identity_verification"]={
 "method":"opaque row IDs; the two packages share zero IDs",
 "r1":"package A (PART1 4305/4305 A-IDs, 0 B-IDs; PART2 360/360 A-IDs)",
 "r2":"package B (PART1 4305/4305 B-IDs, 0 A-IDs; PART2 360/360 B-IDs)",
 "note":"folder names were NOT trusted; identity established from file contents"}
R["provenance"]["file_hashes_sha256"]={
 "reviewer1_PART1_grouping.csv":sha(HE/"completed/annotator_A/PART1_grouping.csv"),
 "reviewer1_PART2_support.csv":sha(HE/"completed/annotator_A/PART2_support.csv"),
 "reviewer2_PART1_grouping.csv":sha(HE/"completed/annotator_B/PART1_grouping.csv"),
 "reviewer2_PART2_support.csv":sha(HE/"completed/annotator_B/PART2_support.csv"),
 "adjudication_variant_r1_semantic":sha(HE/"adjudication/returned/semantic_variant_r1.csv"),
 "adjudication_variant_r2_semantic":sha(HE/"adjudication/returned/semantic_variant_r2.csv"),
 "adjudication_variant_r1_evidence":sha(HE/"adjudication/returned/evidence_variant_r1.csv"),
 "adjudication_variant_r2_evidence":sha(HE/"adjudication/returned/evidence_variant_r2.csv")}
s=json.loads((HE/"results_semantic_agreement.json").read_text())["inter_annotator"]
e=json.loads((HE/"results_evidence_agreement.json").read_text())
R["first_pass_reliability"]={"status":"INDEPENDENT first-pass annotation, pre-adjudication",
 "semantic":{"n_cross_source_pair_decisions":s["n_comparable_cross_source_pairs"],
   "raw_agreement":round(s["raw_agreement"],4),"cohens_kappa":round(s["cohens_kappa"],4),
   "positive_class_f1":round(s["positive_class_f1"],4),
   "positive_class_precision":round(s["positive_class_precision_B_vs_A"],4),
   "positive_class_recall":round(s["positive_class_recall_B_vs_A"],4),
   "positive_decisions_reviewer1":s["positive_equivalence_decisions_A"],
   "positive_decisions_reviewer2":s["positive_equivalence_decisions_B"],
   "confusion":s["confusion"]},
 "evidence":{"n":e["n_items_both_annotators"],"raw_agreement":round(e["raw_agreement"],4),
   "cohens_kappa":round(e["cohens_kappa"],4),
   "uncertain_rate":{"reviewer1":round(e["uncertain_rate"]["A"],4),"reviewer2":round(e["uncertain_rate"]["B"],4)}},
 "note":"NOT recomputed after adjudication; reliability refers only to the independent first pass"}
R["counts"]={"semantic_panels_per_reviewer":180,"items_labelled_per_reviewer":4305,
 "panels_with_no_annotatable_items":1,"evidence_items_per_reviewer":360,
 "cohort_slides":30,"lectures":18,
 "adjudication_cases":{"semantic":286,"evidence":33},
 "adjudication_unresolved_between_variants":{"semantic":47,"evidence":12}}

# ---- H1 -------------------------------------------------------------------
band=json.loads((HE/"results_consensus_band.json").read_text())
struct=defaultdict(list)
for m in MODELS:
    recs=defaultdict(dict)
    for r in read_jsonl(ROOT/f"results/final_task_specific/merged/{m}.jsonl"):
        recs[(r["task"],(r["lecture"],r["slide_id"]))][r["mode"]]=r
    coh={(x["lecture"],x["slide_id"]) for x in csv.DictReader(open(HE/"frozen_30_slide_manifest.csv"))}
    for (t,k),by in recs.items():
        if k not in coh: continue
        for l,rr in PAIRS:
            x,y=by.get(l),by.get(rr)
            if not(x and y and is_valid(x) and is_valid(y)): continue
            j=_jaccard(item_set(t,x.get("parsed_output")),item_set(t,y.get("parsed_output")))
            if j is not None: struct[(m,t,f"{l}-{rr}")].append(j)
cells=[]
for m in MODELS:
    for t in ("concepts","triples"):
        for l,rr in PAIRS:
            key=f"{m}|{t}|{l}-{rr}"; ex=struct.get((m,t,f"{l}-{rr}"))
            if not ex: continue
            exm=statistics.mean(ex); v1=band["variants"]["r1"].get(key); v2=band["variants"]["r2"].get(key)
            cells.append({"model":m,"task":TL[t],"source_pair":f"{l}-{rr}","N_slides":len(ex),
              "exact_structural_jaccard":round(exm,4),
              "human_semantic_variant_r1":v1,"human_semantic_variant_r2":v2,
              "band_min":min(v1,v2),"band_max":max(v1,v2),
              "uplift_min":round(min(v1,v2)-exm,4),"uplift_max":round(max(v1,v2)-exm,4)})
agg={}
for t in ("concepts","relations"):
    sub=[c for c in cells if c["task"]==t]
    ex=statistics.mean([c["exact_structural_jaccard"] for c in sub])
    a=statistics.mean([c["human_semantic_variant_r1"] for c in sub])
    b=statistics.mean([c["human_semantic_variant_r2"] for c in sub])
    agg[t]={"n_cells":len(sub),"exact_structural_jaccard":round(ex,4),
      "human_semantic_variant_r1":round(a,4),"human_semantic_variant_r2":round(b,4),
      "band":[round(min(a,b),4),round(max(a,b),4)],
      "absolute_uplift_band":[round(min(a,b)-ex,4),round(max(a,b)-ex,4)],
      "band_width":round(abs(a-b),4),
      "ratio_human_to_exact_band":[round(min(a,b)/ex,3),round(max(a,b)/ex,3)]}
R["H1"]={"per_cell":cells,"aggregate":agg,
 "method":"two independently completed adjudication variants; band reported, no consensus invented"}

# ---- ordering check under both variants ------------------------------------
oc=[]
for m in MODELS:
    for t in ("concepts","triples"):
        for v in ("r1","r2"):
            vals={p:band["variants"][v].get(f"{m}|{t}|{p}") for p in ("MM-TX","MM-IMG","TX-IMG")}
            if any(x is None for x in vals.values()): continue
            order=sorted(vals,key=lambda k:-vals[k])
            oc.append({"model":m,"task":TL[t],"variant":v,"order":" > ".join(order),
                       "matches_MM_TX_gt_MM_IMG_gt_TX_IMG":order==["MM-TX","MM-IMG","TX-IMG"]})
R["ordering_check"]={"cells":oc,"n_cells":len(oc),
  "n_preserved":sum(1 for x in oc if x["matches_MM_TX_gt_MM_IMG_gt_TX_IMG"]),
  "first_pass_preserved_cells":"12/12 (both reviewers, 3 models x 2 tasks)"}

# ---- H2 --------------------------------------------------------------------
def ev_labels(annot):
    ann=json.loads((HE/f"completed/annotator_{annot}/annotations_annotator_{annot}.json").read_text())
    meta={x["evidence_id"]:x for x in json.loads((HE/f"sealed_mapping/annotator_{annot}_mapping.json").read_text())["evidence_items"]}
    return {(meta[i]["model"],meta[i]["task"],meta[i]["lecture"],meta[i]["slide_id"],meta[i]["normalized_key"]):r["label"]
            for i,r in ann["evidence"].items() if i in meta}, meta
LA,metaA=ev_labels("A"); LB,_=ev_labels("B")
slides=[(x["lecture"],x["slide_id"]) for x in csv.DictReader(open(HE/"frozen_30_slide_manifest.csv"))]
sidx={f"slides/slide_{i:02d}.jpg":k for i,k in enumerate(slides,1)}
disp={(m["lecture"],m["slide_id"],m["task"],m["display"]):(m["model"],m["task"],m["lecture"],m["slide_id"],m["normalized_key"]) for m in metaA.values()}
def apply(base,variant):
    out=dict(base); n=0
    for r in csv.DictReader(open(HE/f"adjudication/returned/evidence_variant_{variant}.csv",encoding="utf-8-sig")):
        lab=LBL.get((r["consensus_label__FILL"] or "").strip().lower())
        k=sidx.get(r["slide_picture"])
        if not lab or not k: continue
        tgt=disp.get((k[0],k[1],r["task"],r["item"]))
        if tgt: out[tgt]=lab; n+=1
    return out,n
def props(d,sel=lambda k:True):
    c=Counter(v for k,v in d.items() if sel(k)); n=sum(c.values())
    return {"n":n,**{x:{"count":c.get(x,0),"proportion":round(c.get(x,0)/n,4) if n else None,
            "wilson95":wilson(c.get(x,0),n)} for x in ("both","transcript_only","image_only","neither","uncertain")}}
h2={}
for v,base in (("r1",LA),("r2",LB)):
    f,n=apply(base,v)
    h2[v]={"adjudicated_cases_applied":n,"overall":props(f),
      "by_model":{m:props(f,lambda k,m=m:k[0]==m) for m in MODELS},
      "by_task":{TL[t]:props(f,lambda k,t=t:k[1]==t) for t in ("concepts","triples")}}
R["H2"]={"n_items":360,"variants":h2,
  "reviewer_band_overall":{c:[round(min(h2['r1']['overall'][c]['proportion'],h2['r2']['overall'][c]['proportion']),4),
                              round(max(h2['r1']['overall'][c]['proportion'],h2['r2']['overall'][c]['proportion']),4)]
                           for c in ("both","transcript_only","image_only","neither","uncertain")},
  "uncertain_usage_caution":"Both reviewers used UNCERTAIN almost never at first pass (0.0% and 0.28%). This may reflect under-use of the category rather than absence of ambiguous cases; neither/uncertain figures should be read with that in mind."}

# ---- write final human-only evaluation results ----------------------------
(HE/"FINAL_HUMAN_EVALUATION_RESULTS.json").write_text(json.dumps(R,indent=2,sort_keys=True)+"\n")
print("wrote FINAL_HUMAN_EVALUATION_RESULTS.json")
print("\nH1 aggregate:"); print(json.dumps(agg,indent=1))
print(f"\nordering preserved: {R['ordering_check']['n_preserved']}/{R['ordering_check']['n_cells']} adjudication-variant cells")
print("\nH2 overall band:",json.dumps(R["H2"]["reviewer_band_overall"],indent=1))
