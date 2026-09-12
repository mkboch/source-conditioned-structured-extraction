#!/usr/bin/env python3
"""Build the blinded two-annotator human-evaluation package.

Blinding: model -> {M1,M2,M3} and source -> {A,B,C}, with DIFFERENT permutations per
annotator, derived from frozen seeds. Item presentation order randomised within each source
set. Opaque item IDs are HMAC-derived and encode nothing about model or source.

Deblinding maps are written ONLY to human_eval/sealed_mapping/, never into annotator folders.
"""
from __future__ import annotations
import csv, hashlib, hmac, json, random, shutil, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from v3_metrics import is_valid, item_set, normalize_text

HE = ROOT / "human_eval"
MODELS = ["InternVL3-14B", "Gemma-4-31B", "Qwen3-VL-32B-Instruct"]
REV = {"InternVL3-14B": "419aa10d2db7da6c64382ad3124f79a60cec42aa",
       "Gemma-4-31B": "145dc2508c480a64b47242f160d286cff94a2343",
       "Qwen3-VL-32B-Instruct": "0cfaf48183f594c314753d30a4c4974bc75f3ccb"}
TASKS = ["concepts", "triples"]
MODES = ["MM", "TX", "IMG"]
SEEDS = {"A": 20260901, "B": 20260902}
V2PACK = Path("external_artifacts/experiment_v2/stage5_audit30_packet")

cohort = list(csv.DictReader(open(HE / "frozen_30_slide_manifest.csv")))
CKEYS = [(r["lecture"], r["slide_id"]) for r in cohort]
CROW = {(r["lecture"], r["slide_id"]): r for r in cohort}


def raw_items(task, parsed):
    """Ordered, de-duplicated display items with their normalised key (frozen definition)."""
    out, seen = [], set()
    if not isinstance(parsed, dict):
        return out
    if task == "concepts":
        for it in parsed.get("concepts", []) or []:
            if not (isinstance(it, dict) and it.get("term")):
                continue
            key = normalize_text(f"{it.get('term','')}|{it.get('category','')}")
            if key in seen:
                continue
            seen.add(key)
            out.append({"key": key, "display": f"{it.get('term','')}  [{it.get('category','')}]",
                        "term": it.get("term",""), "category": it.get("category","")})
    else:
        for it in parsed.get("triples", []) or []:
            if not (isinstance(it, dict) and it.get("subject") and it.get("predicate") and it.get("object")):
                continue
            key = normalize_text(f"{it.get('subject','')}|{it.get('predicate','')}|{it.get('object','')}")
            if key in seen:
                continue
            seen.add(key)
            out.append({"key": key,
                        "display": f"{it.get('subject','')}  —{it.get('predicate','')}→  {it.get('object','')}",
                        "subject": it.get("subject",""), "predicate": it.get("predicate",""),
                        "object": it.get("object","")})
    return out


# ---- load final merged outputs restricted to the cohort -----------------------------
rec = defaultdict(dict)
for m in MODELS:
    for line in open(ROOT / f"results/final_task_specific/merged/{m}.jsonl"):
        r = json.loads(line)
        k = (r["lecture"], r["slide_id"])
        if k in CROW:
            rec[(m, r["task"], k)][r["mode"]] = r


def oid(annot, *parts):
    return "i" + hmac.new(f"itemid-{annot}".encode(),
                          "|".join(map(str, parts)).encode(), hashlib.sha256).hexdigest()[:12]


REF_A = {"model": None, "source": None}   # annotator A's permutation, for derangement


def _deranged(rng, keys, labels, ref):
    """Draw an alias permutation. For the second annotator, require a DERANGEMENT relative to
    annotator A: no model and no source may keep the same alias across annotators. With three
    items a plain reshuffle leaves a fixed point about 2/3 of the time, which weakens blinding
    if the annotators ever compare notes."""
    for _ in range(10000):
        cand = dict(zip(rng.sample(keys, len(keys)), labels))
        if ref is None or all(cand[k] != ref[k] for k in keys):
            return cand
    raise RuntimeError("could not find a deranged permutation")


def build(annot: str):
    rng = random.Random(SEEDS[annot])
    model_alias = _deranged(rng, MODELS, ["M1", "M2", "M3"], REF_A["model"])
    source_alias = _deranged(rng, MODES, ["A", "B", "C"], REF_A["source"])
    if annot == "A":
        REF_A["model"], REF_A["source"] = model_alias, source_alias

    pkg = HE / f"annotator_{annot}"
    (pkg / "slides").mkdir(parents=True, exist_ok=True)
    for i, k in enumerate(CKEYS, 1):
        shutil.copy(CROW[k]["image_path"], pkg / "slides" / f"slide_{i:02d}.jpg")

    panels, mapping_items = [], []
    pn = 0
    for m in MODELS:
        for t in TASKS:
            for si, k in enumerate(CKEYS, 1):
                pn += 1
                rs = rec.get((m, t, k), {})
                sets = []
                for mode in MODES:
                    r = rs.get(mode)
                    valid = bool(r) and is_valid(r)
                    items = raw_items(t, r.get("parsed_output")) if valid else []
                    disp = []
                    for idx, it in enumerate(items):
                        iid = oid(annot, m, t, k[0], k[1], mode, idx)
                        disp.append({"item_id": iid, "text": it["display"]})
                        mapping_items.append({"item_id": iid, "panel_id": f"P{pn:03d}",
                                              "model": m, "model_alias": model_alias[m],
                                              "task": t, "lecture": k[0], "slide_id": k[1],
                                              "source": mode, "source_alias": source_alias[mode],
                                              "normalized_key": it["key"], "display": it["display"]})
                    rng.shuffle(disp)           # randomise presentation order within the source set
                    sets.append({"source_alias": source_alias[mode], "valid": valid, "items": disp})
                sets.sort(key=lambda s: s["source_alias"])
                n_items = sum(len(s["items"]) for s in sets)
                panels.append({
                    "panel_id": f"P{pn:03d}",
                    "model_alias": model_alias[m],
                    "task_display": "CONCEPT" if t == "concepts" else "RELATION",
                    "slide_index": si,
                    "slide_image": f"slides/slide_{si:02d}.jpg",
                    "transcript": Path(CROW[k]["text_path"]).read_text(encoding="utf-8", errors="replace"),
                    "sets": sets,
                    "n_items": n_items,
                    "no_annotatable_items": n_items == 0,
                })

    # ---- evidence sample: deterministic, stratified across slides, seed 20260831 -----
    ev_rng = random.Random(20260831)
    evidence, ev_map = [], []
    ev_counts = {}
    for m in MODELS:
        for t in TASKS:
            by_slide = defaultdict(list)
            for k in CKEYS:
                r = rec.get((m, t, k), {}).get("MM")
                if not (r and is_valid(r)):
                    continue
                for idx, it in enumerate(raw_items(t, r.get("parsed_output"))):
                    by_slide[k].append((k, idx, it))
            for k in by_slide:
                ev_rng.shuffle(by_slide[k])
            # round-robin across slides so high-yield slides cannot dominate
            picked, slides = [], sorted(by_slide, key=lambda x: (x[0], x[1]))
            ptr = {k: 0 for k in slides}
            while len(picked) < 60:
                progressed = False
                for k in slides:
                    if ptr[k] < len(by_slide[k]):
                        picked.append(by_slide[k][ptr[k]]); ptr[k] += 1; progressed = True
                        if len(picked) == 60:
                            break
                if not progressed:
                    break
            ev_counts[f"{m}|{t}"] = len(picked)
            for (k, idx, it) in picked:
                eid = oid(annot, "EV", m, t, k[0], k[1], idx)
                si = CKEYS.index(k) + 1
                evidence.append({"evidence_id": eid, "model_alias": model_alias[m],
                                 "task_display": "CONCEPT" if t == "concepts" else "RELATION",
                                 "slide_index": si, "slide_image": f"slides/slide_{si:02d}.jpg",
                                 "transcript": Path(CROW[k]["text_path"]).read_text(encoding="utf-8", errors="replace"),
                                 "item_text": it["display"]})
                ev_map.append({"evidence_id": eid, "model": m, "model_alias": model_alias[m],
                               "task": t, "lecture": k[0], "slide_id": k[1], "source": "MM",
                               "normalized_key": it["key"], "display": it["display"]})
    ev_rng.shuffle(evidence)

    (pkg / "panels.json").write_text(json.dumps(
        {"annotator": annot, "n_panels": len(panels), "panels": panels}, indent=1, sort_keys=False))
    (pkg / "evidence.json").write_text(json.dumps(
        {"annotator": annot, "n_items": len(evidence), "items": evidence}, indent=1, sort_keys=False))

    sealed = {"annotator": annot, "seed": SEEDS[annot],
              "model_alias": model_alias, "source_alias": source_alias,
              "model_revisions": REV,
              "evidence_sample_seed": 20260831, "evidence_counts_per_cell": ev_counts,
              "items": mapping_items, "evidence_items": ev_map}
    (HE / "sealed_mapping" / f"annotator_{annot}_mapping.json").write_text(
        json.dumps(sealed, indent=1, sort_keys=True))
    return panels, evidence, model_alias, source_alias, ev_counts


summary = {}
for a in ("A", "B"):
    p, e, ma, sa, ec = build(a)
    summary[a] = {"panels": len(p), "evidence": len(e),
                  "items_total": sum(x["n_items"] for x in p),
                  "empty_panels": sum(1 for x in p if x["no_annotatable_items"]),
                  "model_alias": ma, "source_alias": sa, "evidence_per_cell": ec}
print(json.dumps(summary, indent=2))
assert summary["A"]["model_alias"] != summary["B"]["model_alias"], "alias permutations must differ"
assert summary["A"]["source_alias"] != summary["B"]["source_alias"], "alias permutations must differ"
print("\nalias permutations differ between annotators: OK")
