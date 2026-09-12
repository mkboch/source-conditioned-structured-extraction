#!/usr/bin/env python3
"""Build adjudication packets containing ONLY the cases where the two annotators disagree.

Model and source identity are NOT revealed. Consensus fields are left empty; nothing is
resolved automatically. Run after both first passes are validated.
"""
from __future__ import annotations
import argparse, csv, json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
HE = Path(__file__).resolve().parents[1]

def maps(annot):
    s = json.loads((HE/f"sealed_mapping/annotator_{annot}_mapping.json").read_text())
    return ({i["item_id"]: i for i in s["items"]}, {e["evidence_id"]: e for e in s["evidence_items"]})

def slide_index_map():
    """(lecture, slide_id) -> the slide_NN name the annotators already have."""
    rows = list(csv.DictReader(open(HE/"frozen_30_slide_manifest.csv")))
    return {(r["lecture"], r["slide_id"]): i for i, r in enumerate(rows, 1)}


def main():
    ap = argparse.ArgumentParser(); a = ap.parse_args()
    out = HE/"adjudication"; out.mkdir(exist_ok=True)
    SIDX = slide_index_map()
    mA, eA = maps("A"); mB, eB = maps("B")
    annA = json.loads((HE/"completed/annotator_A/annotations_annotator_A.json").read_text())
    annB = json.loads((HE/"completed/annotator_B/annotations_annotator_B.json").read_text())

    # ---- semantic: cross-source pairs the annotators grouped differently ----------
    def decisions(ann, meta):
        cl = {}
        for pid, rec in ann.get("panels", {}).items():
            for iid, lab in (rec.get("clusters") or {}).items():
                lab = (lab or "").strip()
                if lab and iid in meta: cl[iid] = f"{pid}::{lab}"
        by = defaultdict(list)
        for iid, gid in cl.items():
            m = meta[iid]
            by[(m["model"], m["task"], m["lecture"], m["slide_id"])].append((m["normalized_key"], m["source"], gid, m["display"]))
        d = {}
        for cell, items in by.items():
            for x, y in combinations(sorted(items), 2):
                if x[1] == y[1]: continue
                d[(cell, x[0], y[0])] = (1 if x[2] == y[2] else 0, x[3], y[3])
        # cell = (model, task, lecture, slide_id); slide reference added by the caller
        return d
    dA, dB = decisions(annA, mA), decisions(annB, mB)
    rows = []
    for k in sorted(set(dA) & set(dB), key=str):
        va, vb = dA[k][0], dB[k][0]
        if va == vb: continue
        si = SIDX.get((k[0][2], k[0][3]))
        rows.append({"case_id": f"S{len(rows)+1:05d}",
                     "type": "CONCEPT" if k[0][1] == "concepts" else "RELATION",
                     "slide_picture": f"slides/slide_{si:02d}.jpg" if si else "",
                     "transcript_file": f"transcripts/slide_{si:02d}.txt" if si else "",
                     "task": k[0][1],
                     "item_1": dA[k][1], "item_2": dA[k][2],
                     "annotator_A_says_equivalent": va, "annotator_B_says_equivalent": vb,
                     "consensus_equivalent__FILL_1_or_0": "", "consensus_note": ""})
    with (out/"semantic_disagreements.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else
            ["case_id","task","item_1","item_2","annotator_A_says_equivalent",
             "annotator_B_says_equivalent","consensus_equivalent__FILL_1_or_0","consensus_note"])
        w.writeheader(); w.writerows(rows)
    (out/"semantic_disagreements.json").write_text(json.dumps(rows, indent=1)+"\n")

    # ---- evidence: differing category labels ---------------------------------------
    kA = {(e["model"],e["task"],e["lecture"],e["slide_id"],e["normalized_key"]): (i, e)
          for i, e in eA.items()}
    kB = {(e["model"],e["task"],e["lecture"],e["slide_id"],e["normalized_key"]): (i, e)
          for i, e in eB.items()}
    erows = []
    for k in sorted(set(kA) & set(kB), key=str):
        la = (annA.get("evidence", {}).get(kA[k][0]) or {}).get("label")
        lb = (annB.get("evidence", {}).get(kB[k][0]) or {}).get("label")
        if not la or not lb or la == lb: continue
        si = SIDX.get((k[2], k[3]))
        erows.append({"case_id": f"E{len(erows)+1:05d}",
                      "type": "CONCEPT" if k[1] == "concepts" else "RELATION",
                      "slide_picture": f"slides/slide_{si:02d}.jpg" if si else "",
                      "transcript_file": f"transcripts/slide_{si:02d}.txt" if si else "",
                      "task": k[1], "item": kA[k][1]["display"],
                      "annotator_A_label": la, "annotator_B_label": lb,
                      "consensus_label__FILL": "", "consensus_note": ""})
    with (out/"evidence_disagreements.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(erows[0].keys()) if erows else
            ["case_id","task","item","lecture","slide_id","annotator_A_label",
             "annotator_B_label","consensus_label__FILL","consensus_note"])
        w.writeheader(); w.writerows(erows)
    (out/"evidence_disagreements.json").write_text(json.dumps(erows, indent=1)+"\n")
    print(f"semantic disagreements: {len(rows)}   evidence disagreements: {len(erows)}")
    print("Model and source identity are not included. Consensus fields are empty by design.")

if __name__ == "__main__":
    raise SystemExit(main())
