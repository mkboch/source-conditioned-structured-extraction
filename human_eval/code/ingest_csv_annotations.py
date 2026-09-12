#!/usr/bin/env python3
"""Convert returned PART1/PART2 CSVs into the JSON format the scoring scripts consume.

Run this on each annotator's returned files, then run validate_annotations.py as before.
Answer wording in the CSV is deliberately plain; it is mapped to the frozen category names.
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path
HE = Path(__file__).resolve().parents[1]

ANSWER_MAP = {
    "both": "both",
    "transcript": "transcript_only", "transcript_only": "transcript_only",
    "picture": "image_only", "image": "image_only", "image_only": "image_only",
    "neither": "neither",
    "unsure": "uncertain", "uncertain": "uncertain",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotator", required=True, choices=["A", "B"])
    ap.add_argument("--part1", type=Path)
    ap.add_argument("--part2", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    base = HE / f"completed/annotator_{a.annotator}"
    p1 = a.part1 or base / "PART1_grouping.csv"
    p2 = a.part2 or base / "PART2_support.csv"
    out = a.out or base / f"annotations_annotator_{a.annotator}.json"
    for f in (p1, p2):
        if not f.exists():
            sys.exit(f"MISSING returned file: {f}")

    panels = defaultdict(lambda: {"clusters": {}, "uncertain": {}, "note": ""})
    bad = []
    for r in csv.DictReader(open(p1, encoding="utf-8-sig")):
        rid = (r.get("row_id") or "").strip()
        if not rid:
            continue                      # the "nothing to do" placeholder rows
        lab = (r.get("FILL_label") or "").strip()
        if not lab:
            bad.append(f"PART1 {r['panel']} row {rid}: FILL_label empty")
            continue
        pid = r["panel"]
        panels[pid]["clusters"][rid] = lab
        if (r.get("FILL_unsure") or "").strip().lower() in ("y", "yes", "1", "true"):
            panels[pid]["uncertain"][rid] = True
        n = (r.get("FILL_note") or "").strip()
        if n:
            panels[pid]["note"] = (panels[pid]["note"] + " | " + n).strip(" |")

    evidence = {}
    for r in csv.DictReader(open(p2, encoding="utf-8-sig")):
        rid = (r.get("row_id") or "").strip()
        raw = (r.get("FILL_answer") or "").strip().lower()
        if not rid:
            continue
        if not raw:
            bad.append(f"PART2 row {r.get('no')}: FILL_answer empty")
            continue
        if raw not in ANSWER_MAP:
            bad.append(f"PART2 row {r.get('no')}: unrecognised answer {raw!r} "
                       f"(expected one of: both, transcript, picture, neither, unsure)")
            continue
        evidence[rid] = {"label": ANSWER_MAP[raw], "note": (r.get("FILL_note") or "").strip()}

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"annotator": a.annotator, "format_version": 1,
                               "source": "csv_offline_package",
                               "panels": dict(panels), "evidence": evidence}, indent=1) + "\n")
    print(f"annotator {a.annotator}: {len(panels)} panels, "
          f"{sum(len(v['clusters']) for v in panels.values())} labelled rows, "
          f"{len(evidence)} answers -> {out}")
    if bad:
        print(f"\n{len(bad)} problem rows (fix and re-run, or they are excluded):")
        for b in bad[:25]:
            print("   ", b)
        if len(bad) > 25:
            print(f"    ... {len(bad)-25} more")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
