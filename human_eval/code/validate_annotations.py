#!/usr/bin/env python3
"""Validate a completed first-pass annotation export before any scoring.

Checks: package identity, all panels present, all items covered, valid cluster labels,
valid evidence categories, no duplicate/missing/unknown item IDs, mapping integrity.
Run this on each annotator's export before scoring anything.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
HE = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotator", required=True, choices=["A", "B"])
    ap.add_argument("--file", type=Path)
    a = ap.parse_args()
    f = a.file or HE / f"completed/annotator_{a.annotator}/annotations_annotator_{a.annotator}.json"
    if not f.exists():
        sys.exit(f"MISSING annotation file: {f}")
    ann = json.loads(f.read_text())
    panels = json.loads((HE / f"package_data/annotator_{a.annotator}/panels.json").read_text())["panels"]
    ev = json.loads((HE / f"package_data/annotator_{a.annotator}/evidence.json").read_text())["items"]
    smap = json.loads((HE / f"sealed_mapping/annotator_{a.annotator}_mapping.json").read_text())

    errs, warns = [], []
    if ann.get("annotator") != a.annotator:
        errs.append(f"package identity mismatch: file says {ann.get('annotator')!r}, expected {a.annotator!r}")

    known = {i["item_id"] for i in smap["items"]}
    seen = set()
    for p in panels:
        rec = ann.get("panels", {}).get(p["panel_id"])
        ids = [it["item_id"] for s in p["sets"] for it in s["items"]]
        if p["no_annotatable_items"]:
            continue
        if not rec:
            errs.append(f"panel {p['panel_id']}: no annotation recorded"); continue
        cl = rec.get("clusters", {})
        for iid in ids:
            v = (cl.get(iid) or "").strip()
            if not v:
                errs.append(f"panel {p['panel_id']}: item {iid} has no group label")
            if iid in seen:
                errs.append(f"duplicate item id across panels: {iid}")
            seen.add(iid)
        extra = set(cl) - set(ids)
        if extra:
            errs.append(f"panel {p['panel_id']}: {len(extra)} labels for unknown item ids")
        # a panel where every item got a distinct label is legal but worth flagging
        vals = [v for v in (cl.get(i, "").strip() for i in ids) if v]
        if vals and len(set(vals)) == len(vals) and len(vals) > 3:
            warns.append(f"panel {p['panel_id']}: no two items grouped together ({len(vals)} items)")
    unknown = seen - known
    if unknown:
        errs.append(f"{len(unknown)} annotated item ids are not in the sealed mapping")

    VALID = {"both", "transcript_only", "image_only", "neither", "uncertain"}
    eids = {v["evidence_id"] for v in ev}
    got = ann.get("evidence", {})
    for v in ev:
        r = got.get(v["evidence_id"])
        if not r or not r.get("label"):
            errs.append(f"evidence {v['evidence_id']}: no label")
        elif r["label"] not in VALID:
            errs.append(f"evidence {v['evidence_id']}: invalid label {r['label']!r}")
    if set(got) - eids:
        errs.append(f"{len(set(got)-eids)} evidence labels for unknown ids")

    print(f"annotator {a.annotator}: {len(panels)} panels, {len(ev)} evidence items")
    print(f"  errors:   {len(errs)}")
    print(f"  warnings: {len(warns)}")
    for e in errs[:40]: print("   ERROR  ", e)
    for w in warns[:20]: print("   warn   ", w)
    if len(errs) > 40: print(f"   ... {len(errs)-40} more errors")
    print("VALIDATION:", "PASS" if not errs else "FAIL")
    return 0 if not errs else 1

if __name__ == "__main__":
    raise SystemExit(main())
