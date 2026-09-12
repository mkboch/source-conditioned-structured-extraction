#!/usr/bin/env python3
"""Final scoring when adjudication returned TWO independent variants rather than one consensus.

Rather than inventing a consensus, this computes every quantity under BOTH returned variants
and reports the band. All 47 unresolved semantic cases split the same way (r1 = not-equivalent,
r2 = equivalent), so r1 is the CONSERVATIVE bound (lower semantic uplift, biased AGAINST the
study's own hypothesis) and r2 is the LIBERAL bound.

Consensus clustering is built per (model, task, slide) as connected components over the union
of: (a) item pairs BOTH first-pass annotators called equivalent, and (b) the disputed pairs the
returned variant marks equivalent. Transitive closure can merge chains the annotators did not
each endorse pairwise; the size of that effect is measured and reported rather than assumed away.
"""
from __future__ import annotations
import csv, json, statistics, sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
HE = Path(__file__).resolve().parents[1]; ROOT = HE.parent
sys.path.insert(0, str(ROOT/"code"))
from v3_metrics import read_jsonl, is_valid, item_set, _jaccard
PAIRS=(("MM","TX"),("MM","IMG"),("TX","IMG"))
MODELS=["InternVL3-14B","Gemma-4-31B","Qwen3-VL-32B-Instruct"]
TL={"concepts":"concepts","triples":"relations"}

def first_pass(annot):
    ann=json.loads((HE/f"completed/annotator_{annot}/annotations_annotator_{annot}.json").read_text())
    meta={i["item_id"]:i for i in json.loads((HE/f"sealed_mapping/annotator_{annot}_mapping.json").read_text())["items"]}
    cl={}
    for pid,rec in ann["panels"].items():
        for iid,lab in (rec.get("clusters") or {}).items():
            lab=(lab or "").strip()
            if lab and iid in meta: cl[iid]=f"{pid}::{lab}"
    return cl, meta

clA,metaA=first_pass("A"); clB,metaB=first_pass("B")
# canonical item key shared across annotators
kA={i:(m["model"],m["task"],m["lecture"],m["slide_id"],m["source"],m["normalized_key"]) for i,m in metaA.items()}
kB={i:(m["model"],m["task"],m["lecture"],m["slide_id"],m["source"],m["normalized_key"]) for i,m in metaB.items()}

def decisions(cl,keymap):
    by=defaultdict(list)
    for iid,gid in cl.items():
        k=keymap[iid]; by[(k[0],k[1],k[2],k[3])].append((k[5],k[4],gid))
    d={}
    for cell,items in by.items():
        for x,y in combinations(sorted(items),2):
            if x[1]==y[1]: continue
            d[(cell,x[0],y[0])]=1 if x[2]==y[2] else 0
    return d
dA,dB=decisions(clA,kA),decisions(clB,kB)
agreed_yes={k for k in dA if dB.get(k)==1 and dA[k]==1}
disputed=[k for k in dA if k in dB and dA[k]!=dB[k]]

def load_variant(v):
    rows=list(csv.DictReader(open(HE/f"adjudication/returned/semantic_variant_{v}.csv",encoding="utf-8-sig")))
    return {r["case_id"]:(r["consensus_equivalent__FILL_1_or_0"] or "").strip() for r in rows}, rows

# case_id order matches create_disagreement_packet's sorted(set(dA)&set(dB)) traversal
ordered=[k for k in sorted(set(dA)&set(dB),key=str) if dA[k]!=dB[k]]

def clusters_for(variant):
    cons,rows=load_variant(variant)
    cid={r["case_id"]:i for i,r in enumerate(rows)}
    yes=set(agreed_yes)
    for i,k in enumerate(ordered,1):
        if cons.get(f"S{i:05d}")=="1": yes.add(k)
    # connected components per cell
    nodes=defaultdict(set); adj=defaultdict(lambda: defaultdict(set))
    for iid,m in metaA.items():
        cell=(m["model"],m["task"],m["lecture"],m["slide_id"]); nodes[cell].add((m["normalized_key"],m["source"]))
    for (cell,x,y) in yes:
        for sx in [s for (nk,s) in nodes[cell] if nk==x]:
            for sy in [s for (nk,s) in nodes[cell] if nk==y]:
                adj[cell][(x,sx)].add((y,sy)); adj[cell][(y,sy)].add((x,sx))
    out={}
    for cell,ns in nodes.items():
        seen={}; cnum=0
        for n in sorted(ns):
            if n in seen: continue
            cnum+=1; stack=[n]
            while stack:
                u=stack.pop()
                if u in seen: continue
                seen[u]=cnum
                for v in adj[cell].get(u,()): 
                    if v not in seen: stack.append(v)
        out[cell]=seen
    return out

struct=defaultdict(list)
COHORT={(x["lecture"],x["slide_id"]) for x in csv.DictReader(open(HE/"frozen_30_slide_manifest.csv"))}
for m in MODELS:
    recs=defaultdict(dict)
    for r in read_jsonl(ROOT/f"results/final_task_specific/merged/{m}.jsonl"):
        recs[(r["task"],(r["lecture"],r["slide_id"]))][r["mode"]]=r
    for (t,k),by in recs.items():
        if k not in COHORT: continue   # human comparison must use the SAME 30 slides
        for l,rr in PAIRS:
            x,y=by.get(l),by.get(rr)
            if not(x and y and is_valid(x) and is_valid(y)): continue
            j=_jaccard(item_set(t,x.get("parsed_output")),item_set(t,y.get("parsed_output")))
            if j is not None: struct[(m,t,f"{l}-{rr}")].append((k[0],j))

res={"method":"two independent adjudication variants; band reported, no consensus invented",
     "unresolved_semantic":len(ordered),"variants":{}}
for v in ("r1","r2"):
    cc=clusters_for(v)
    rows=defaultdict(list)
    for cell,seen in cc.items():
        model,task,lec,slide=cell
        bysrc=defaultdict(set)
        for (nk,src),c in seen.items(): bysrc[src].add(c)
        for l,rr in PAIRS:
            if l not in bysrc or rr not in bysrc: continue
            j=_jaccard(bysrc[l],bysrc[rr])
            if j is not None: rows[(model,task,f"{l}-{rr}")].append((lec,j))
    res["variants"][v]={f"{m}|{t}|{p}":round(statistics.mean([x for _,x in vv]),4) for (m,t,p),vv in sorted(rows.items())}
res["exact_structural"]={f"{m}|{t}|{p}":round(statistics.mean([x for _,x in v]),4) for (m,t,p),v in sorted(struct.items())}
(HE/"results_consensus_band.json").write_text(json.dumps(res,indent=2,sort_keys=True)+"\n")

print(f"{'model':24} {'task':10} {'pair':8} {'exact':>7} {'r1':>7} {'r2':>7} {'band uplift':>18}")
agg=defaultdict(lambda: defaultdict(list))
for key in res["exact_structural"]:
    m,t,p=key.split("|"); e=res["exact_structural"][key]
    a=res["variants"]["r1"].get(key); b=res["variants"]["r2"].get(key)
    if a is None or b is None: continue
    print(f"{m:24} {TL[t]:10} {p:8} {e:7.4f} {a:7.4f} {b:7.4f}   +{a-e:.4f} .. +{b-e:.4f}")
    agg[t]["e"].append(e); agg[t]["a"].append(a); agg[t]["b"].append(b)
print()
for t in ("concepts","triples"):
    e=statistics.mean(agg[t]["e"]); a=statistics.mean(agg[t]["a"]); b=statistics.mean(agg[t]["b"])
    print(f"{TL[t].upper():10} exact={e:.4f}  consensus band = {a:.4f} .. {b:.4f}   uplift +{a-e:.4f} .. +{b-e:.4f}")
