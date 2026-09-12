#!/usr/bin/env python3
"""Frozen technical gate. OUTPUT HEALTH ONLY - no scientific overlap is computed here.

Deliberately imports nothing from v3_metrics: the gate decision must be reachable without
ever looking at agreement between conditions (Stage 2 / Stage 7 requirement).

PASS requires ALL of:
  1. aggregate technical validity >= 95.0 %   (parsed AND schema-valid AND no generation error)
  2. zero generation errors
  3. degenerate-repetition rate <= 1.0 %      ("no systematic corruption")
  4. every (mode, task) cell individually >= 90.0 %  (catches a single collapsed condition
     that an aggregate above 95 % would hide)
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

AGG_MIN, CELL_MIN, DEGEN_MAX = 95.0, 90.0, 1.0


def read_jsonl(p: Path) -> list[dict]:
    rows = []
    if not p.exists():
        return rows
    for line in p.open(encoding="utf-8"):
        if line.strip():
            rows.append(json.loads(line))
    return rows


def latest(rows: list[dict]) -> list[dict]:
    """One record per (lecture, slide, mode, task); a retry supersedes an earlier attempt."""
    d = {}
    for r in rows:
        d[(r.get("lecture"), r.get("slide_id"), r.get("mode"), r.get("task"))] = r
    return list(d.values())


def gate(name: str, path: Path, expected: int | None) -> dict:
    rows = latest(read_jsonl(path))
    n = len(rows)
    err = [r for r in rows if r.get("generation_error")]
    parse_ok = sum(1 for r in rows if r.get("json_parse_success"))
    schema_ok = sum(1 for r in rows if r.get("schema_valid"))
    valid = sum(1 for r in rows if r.get("json_parse_success") and r.get("schema_valid")
                and not r.get("generation_error"))
    degen = sum(1 for r in rows if r.get("degenerate_repetition"))
    empty_raw = sum(1 for r in rows if not (r.get("raw_output") or "").strip())
    lens = [len((r.get("raw_output") or "")) for r in rows]

    cells = {}
    for r in rows:
        cells.setdefault((r.get("mode"), r.get("task")), []).append(r)
    cell_stats = {}
    for (m, t), rs in sorted(cells.items()):
        v = sum(1 for r in rs if r.get("json_parse_success") and r.get("schema_valid")
                and not r.get("generation_error"))
        cell_stats[f"{m}_{t}"] = {"n": len(rs), "valid": v,
                                  "validity_pct": round(100.0 * v / len(rs), 2) if rs else None,
                                  "generation_errors": sum(1 for r in rs if r.get("generation_error")),
                                  "degenerate": sum(1 for r in rs if r.get("degenerate_repetition"))}

    agg = 100.0 * valid / n if n else 0.0
    degen_pct = 100.0 * degen / n if n else 0.0
    worst = min((c["validity_pct"] for c in cell_stats.values() if c["validity_pct"] is not None),
                default=0.0)

    reasons = []
    if expected is not None and n != expected:
        reasons.append(f"record count {n} != expected {expected}")
    if agg < AGG_MIN:
        reasons.append(f"aggregate validity {agg:.2f}% < {AGG_MIN}%")
    if err:
        reasons.append(f"{len(err)} generation errors")
    if degen_pct > DEGEN_MAX:
        reasons.append(f"degenerate repetition {degen_pct:.2f}% > {DEGEN_MAX}%")
    if worst < CELL_MIN:
        reasons.append(f"worst cell validity {worst:.2f}% < {CELL_MIN}%")

    return {
        "condition": name, "records_path": str(path), "n_records": n,
        "n_expected": expected,
        "json_parse_success": parse_ok, "schema_valid": schema_ok,
        "technically_valid": valid,
        "aggregate_validity_pct": round(agg, 2),
        "generation_errors": len(err),
        "generation_error_examples": [e.get("generation_error", "")[:300] for e in err[:3]],
        "degenerate_repetition": degen, "degenerate_repetition_pct": round(degen_pct, 2),
        "empty_raw_output": empty_raw,
        "raw_output_chars_mean": round(statistics.mean(lens), 1) if lens else None,
        "raw_output_chars_median": round(statistics.median(lens), 1) if lens else None,
        "raw_output_chars_min": min(lens) if lens else None,
        "raw_output_chars_max": max(lens) if lens else None,
        "per_cell": cell_stats,
        "thresholds": {"aggregate_min_pct": AGG_MIN, "per_cell_min_pct": CELL_MIN,
                       "degenerate_max_pct": DEGEN_MAX},
        "GATE": "PASS" if not reasons else "FAIL",
        "fail_reasons": reasons,
    }


def diagnostic_subset(path: Path, k: int = 3) -> list[dict]:
    """Fixed diagnostic subset: first k records in sorted key order, inspected verbatim."""
    rows = sorted(latest(read_jsonl(path)),
                  key=lambda r: (r.get("lecture", ""), r.get("slide_id", ""),
                                 r.get("mode", ""), r.get("task", "")))
    return [{"key": f"{r.get('lecture')}/{r.get('slide_id')}/{r.get('mode')}/{r.get('task')}",
             "parse": r.get("json_parse_success"), "schema": r.get("schema_valid"),
             "raw_head": (r.get("raw_output") or "")[:400]} for r in rows[:k]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True,
                    help="JSON list of {name, path, expected}")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--diagnostics", action="store_true")
    a = ap.parse_args()
    spec = json.loads(a.spec.read_text())
    results = []
    for s in spec:
        r = gate(s["name"], Path(s["path"]), s.get("expected"))
        if a.diagnostics:
            r["diagnostic_subset"] = diagnostic_subset(Path(s["path"]))
        results.append(r)
    overall = "PASS" if all(r["GATE"] == "PASS" for r in results) else "FAIL"
    out = {"overall_gate": overall, "n_conditions": len(results), "conditions": results}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for r in results:
        print(f"{r['GATE']:4}  {r['condition']:44} n={r['n_records']:5} "
              f"valid={r['aggregate_validity_pct']:6.2f}%  err={r['generation_errors']:3} "
              f"degen={r['degenerate_repetition_pct']:5.2f}%"
              + ("  <- " + "; ".join(r["fail_reasons"]) if r["fail_reasons"] else ""))
    print("OVERALL GATE:", overall)
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
