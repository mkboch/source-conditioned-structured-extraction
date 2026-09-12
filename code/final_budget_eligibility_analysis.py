#!/usr/bin/env python3
"""STAGE H — final relation token-budget and model-eligibility analysis.

Extends the Pilot-100 diagnostic (D-018) to FULL COHORT for the two incumbent models, which
now have relations at BOTH budgets over all 1,062 slides. Qwen is reported at Pilot-100 for
the 512-vs-1024 contrast (it has no full1062 run at 512) and at full1062 for 1024 health.
"""
from __future__ import annotations
import json, statistics, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from v3_metrics import read_jsonl, is_valid, item_set, output_count, read_manifest_keys, _jaccard

TASK, MODES = "triples", ("MM","TX","IMG")
V2 = Path("external_artifacts/experiment_v2"); FT = ROOT/"results/final_task_specific"

FULL = {
 "InternVL3-14B": (V2/"results_full1062_internvl3_14b/full1062/records.jsonl",
                   FT/"stageB_internvl3_14b_P0_relations1024/records.jsonl"),
 "Gemma-4-31B":   (V2/"results_full1062_gemma4_31b/full1062/records.jsonl",
                   FT/"stageB_gemma4_31b_P0_relations1024/records.jsonl"),
}
PILOT = {"Qwen3-VL-32B-Instruct": (ROOT/"results/stage7_pilot100/qwen3vl_32b_instruct/records.jsonl",
                                   ROOT/"results/token_budget_1024/qwen3vl_32b_instruct/records.jsonl")}

def idx(p, mode, uni):
    return {k: r for r in read_jsonl(p) if r.get("task")==TASK and r.get("mode")==mode
            for k in [(r.get("lecture"), r.get("slide_id"))] if k in uni}

def truncated(r):
    raw=(r.get("raw_output") or "").rstrip()
    return (not is_valid(r)) and raw and not r.get("json_parse_success") and not raw.endswith(("}","]"))

def cell(a, b, uni):
    ia=[r for r in a.values() if not is_valid(r)]; ib=[r for r in b.values() if not is_valid(r)]
    ta=[r for r in ia if truncated(r)]; tb=[r for r in ib if truncated(r)]
    both=[k for k in a if k in b and is_valid(a[k]) and is_valid(b[k])]
    eq=adds=rem=0
    for k in both:
        sa,sb=item_set(TASK,a[k].get("parsed_output")),item_set(TASK,b[k].get("parsed_output"))
        if sa==sb: eq+=1
        if sb-sa: adds+=1
        if sa-sb: rem+=1
    inv=[k for k in a if not is_valid(a[k]) and k in b]
    rec=[k for k in inv if is_valid(b[k])]
    rc=[output_count(TASK,b[k].get("parsed_output")) for k in rec]
    allv=[output_count(TASK,r.get("parsed_output")) for r in b.values() if is_valid(r)]
    la=sorted(len(r.get("raw_output") or "") for r in a.values())
    lb=sorted(len(r.get("raw_output") or "") for r in b.values())
    q=lambda L,p: L[min(int(p*(len(L)-1)),len(L)-1)] if L else None
    return {
      "N": len(a),
      "invalid_512": len(ia), "invalid_rate_512_pct": round(100*len(ia)/len(a),2),
      "invalid_1024": len(ib), "invalid_rate_1024_pct": round(100*len(ib)/len(b),2),
      "truncation_assoc_512": len(ta),
      "truncation_frac_of_invalid_512_pct": round(100*len(ta)/len(ia),1) if ia else None,
      "truncation_assoc_1024": len(tb),
      "truncation_frac_of_invalid_1024_pct": round(100*len(tb)/len(ib),1) if ib else None,
      "N_valid_both": len(both),
      "exact_set_equality": eq,
      "exact_set_equality_pct": round(100*eq/len(both),2) if both else None,
      "adds_at_least_one": adds, "removes_or_changes_at_least_one": rem,
      "N_invalid_512": len(inv), "N_recovered_1024": len(rec),
      "recovery_rate_pct": round(100*len(rec)/len(inv),2) if inv else None,
      "recovered_relation_counts": {"n":len(rc),"median":statistics.median(rc) if rc else None,
                                    "mean":round(statistics.mean(rc),2) if rc else None,
                                    "min":min(rc) if rc else None,"max":max(rc) if rc else None},
      "all_valid_1024_relation_counts": {"median":statistics.median(allv) if allv else None,
                                         "mean":round(statistics.mean(allv),2) if allv else None},
      "output_chars_512": {"median":q(la,.5),"p95":q(la,.95),"max":la[-1] if la else None},
      "output_chars_1024": {"median":q(lb,.5),"p95":q(lb,.95),"max":lb[-1] if lb else None},
    }

out={"scope":{}, "full1062":{}, "pilot100":{}}
uni_full=set(read_manifest_keys(ROOT/"manifests/milu1062_manifest.csv"))
uni_pil=set(read_manifest_keys(ROOT/"manifests/pilot100_manifest.csv"))
for name,(p5,p10) in FULL.items():
    out["full1062"][name]={mo: cell(idx(p5,mo,uni_full), idx(p10,mo,uni_full), uni_full) for mo in MODES}
for name,(p5,p10) in PILOT.items():
    out["pilot100"][name]={mo: cell(idx(p5,mo,uni_pil), idx(p10,mo,uni_pil), uni_pil) for mo in MODES}
out["scope"]={"full1062_models":list(FULL),"pilot100_models":list(PILOT),
  "note":"Qwen has no full1062 run at 512 (its 512 Pilot-100 gate failed and was never scaled), so its 512-vs-1024 contrast is reported at Pilot-100."}
(ROOT/"analysis/final_relation_budget_and_eligibility_analysis.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
print("wrote analysis/final_relation_budget_and_eligibility_analysis.json\n")
for scope in ("full1062","pilot100"):
    for m,cells in out[scope].items():
        print(f"=== {m} ({scope})")
        for mo,c in cells.items():
            print(f"  {mo:3} N={c['N']:4} inv512={c['invalid_rate_512_pct']:5.2f}% inv1024={c['invalid_rate_1024_pct']:5.2f}% "
                  f"truncfrac512={c['truncation_frac_of_invalid_512_pct']}% | bothvalid={c['N_valid_both']:4} "
                  f"exact-eq={c['exact_set_equality_pct']}% adds={c['adds_at_least_one']} rem={c['removes_or_changes_at_least_one']} | "
                  f"recov={c['N_recovered_1024']}/{c['N_invalid_512']} recov_med_rel={c['recovered_relation_counts']['median']} "
                  f"vs all_med={c['all_valid_1024_relation_counts']['median']}")
