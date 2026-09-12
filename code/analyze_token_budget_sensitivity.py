#!/usr/bin/env python3
"""Symmetric relation token-budget sensitivity diagnostic: 512 vs 1024.

Frozen config: configs/relation_token_budget_sensitivity_512_vs_1024.json
               sha256 67f6e180d4a6aa61e2785238cc84dc6f410f2c71795ad3d5575beeeb698d771a

ORDERING DISCIPLINE IS ENFORCED STRUCTURALLY: section A (output health) is computed and
written to disk by `--phase health` before `--phase full` will compute any overlap. Running
`--phase full` without a frozen health file is refused.

Applied IDENTICALLY to every model. No prompt, parser, schema, temperature, source condition
or other serving parameter differs between the two conditions - only max_new_tokens.
"""
from __future__ import annotations
import argparse, json, statistics, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from v3_metrics import (read_jsonl, is_valid, item_set, output_count,
                        cluster_bootstrap_mean, read_manifest_keys, _jaccard, fmt_ci)

TASK = "triples"          # frozen on-disk identifier for the relations task
MODES = ("MM", "TX", "IMG")
V2 = Path("external_artifacts/experiment_v2")

SRC_512 = {
    "InternVL3-14B": V2 / "results_full1062_internvl3_14b/full1062/records.jsonl",
    "Gemma-4-31B":   V2 / "results_full1062_gemma4_31b/full1062/records.jsonl",
    "Qwen3-VL-32B-Instruct": ROOT / "results/stage7_pilot100/qwen3vl_32b_instruct/records.jsonl",
}
SRC_1024 = {
    "InternVL3-14B": ROOT / "results/token_budget_1024/internvl3_14b/records.jsonl",
    "Gemma-4-31B":   ROOT / "results/token_budget_1024/gemma4_31b/records.jsonl",
    "Qwen3-VL-32B-Instruct": ROOT / "results/token_budget_1024/qwen3vl_32b_instruct/records.jsonl",
}
GATE = {"aggregate_min_pct": 95.0, "per_cell_min_pct": 90.0, "degenerate_max_pct": 1.0}


def index(path: Path, mode: str, universe: set) -> dict:
    out = {}
    for r in read_jsonl(path):
        if r.get("task") != TASK or r.get("mode") != mode:
            continue
        k = (r.get("lecture"), r.get("slide_id"))
        if k in universe:
            out[k] = r
    return out


def truncated(r: dict) -> bool:
    if is_valid(r):
        return False
    raw = (r.get("raw_output") or "").rstrip()
    return bool(raw) and not r.get("json_parse_success") and not raw.endswith(("}", "]"))


# ------------------------------------------------------------------ A. TECHNICAL HEALTH
def health(recs: dict) -> dict:
    rs = list(recs.values())
    n = len(rs)
    valid = [r for r in rs if is_valid(r)]
    inval = [r for r in rs if not is_valid(r)]
    trunc = [r for r in inval if truncated(r)]
    lens = sorted(len(r.get("raw_output") or "") for r in rs)

    def q(p):
        return lens[min(int(p * (len(lens) - 1)), len(lens) - 1)] if lens else None
    return {
        "N": n,
        "parse_valid": sum(1 for r in rs if r.get("json_parse_success")),
        "schema_valid": sum(1 for r in rs if r.get("schema_valid")),
        "technically_valid": len(valid),
        "validity_pct": round(100.0 * len(valid) / n, 2) if n else None,
        "invalid": len(inval),
        "invalid_pct": round(100.0 * len(inval) / n, 2) if n else None,
        "generation_errors": sum(1 for r in rs if r.get("generation_error")),
        "degenerate_repetition": sum(1 for r in rs if r.get("degenerate_repetition")),
        "degenerate_pct": round(100.0 * sum(1 for r in rs if r.get("degenerate_repetition")) / n, 2) if n else None,
        "truncation_associated_invalid": len(trunc),
        "truncation_associated_invalid_pct_of_all": round(100.0 * len(trunc) / n, 2) if n else None,
        "truncation_associated_invalid_pct_of_invalid": round(100.0 * len(trunc) / len(inval), 1) if inval else None,
        "output_chars": {"min": lens[0] if lens else None, "p25": q(.25), "median": q(.5),
                         "p75": q(.75), "p95": q(.95), "max": lens[-1] if lens else None,
                         "mean": round(statistics.mean(lens), 1) if lens else None},
        "mean_items_when_valid": round(statistics.mean(
            [output_count(TASK, r.get("parsed_output")) for r in valid]), 2) if valid else None,
    }


# ------------------------------- B/C. STABILITY AND RECOVERY (overlap - gated behind health)
def stability(a: dict, b: dict, seed: int, resamples: int) -> dict:
    keys = [k for k in a if k in b and is_valid(a[k]) and is_valid(b[k])]
    eq = adds = removes = 0
    jac, bnj, diffs = [], [], []
    for k in keys:
        sa, sb = item_set(TASK, a[k].get("parsed_output")), item_set(TASK, b[k].get("parsed_output"))
        if sa == sb:
            eq += 1
        if sb - sa:
            adds += 1
        if sa - sb:
            removes += 1
        j = _jaccard(sa, sb)
        if j is not None:                      # excludes valid co-empty
            jac.append((k[0], j))
            if sa and sb:
                bnj.append((k[0], j))
        diffs.append((k[0], float(len(sb) - len(sa))))
    n = len(keys)
    return {
        "N_valid_at_both": n,
        "exact_set_equality": eq,
        "exact_set_equality_rate_pct": round(100.0 * eq / n, 2) if n else None,
        "adds_at_least_one_at_1024": adds,
        "adds_at_least_one_rate_pct": round(100.0 * adds / n, 2) if n else None,
        "removes_or_changes_at_least_one": removes,
        "removes_or_changes_rate_pct": round(100.0 * removes / n, 2) if n else None,
        "N_scored_validity_aware_jaccard": len(jac),
        "validity_aware_jaccard": cluster_bootstrap_mean(jac, seed, resamples),
        "both_nonempty_jaccard": cluster_bootstrap_mean(bnj, seed, resamples),
        "item_count_difference_1024_minus_512": cluster_bootstrap_mean(diffs, seed, resamples),
    }


def recovery(a: dict, b: dict) -> dict:
    inv = [k for k in a if not is_valid(a[k]) and k in b]
    rec = [k for k in inv if is_valid(b[k])]
    counts = [output_count(TASK, b[k].get("parsed_output")) for k in rec]
    still_trunc = sum(1 for k in inv if not is_valid(b[k]) and truncated(b[k]))
    terminates = sum(1 for k in rec if (b[k].get("raw_output") or "").rstrip().endswith(("}", "]")))
    return {
        "N_invalid_at_512": len(inv),
        "N_recovered_valid_at_1024": len(rec),
        "recovery_rate_pct": round(100.0 * len(rec) / len(inv), 2) if inv else None,
        "recovered_terminating_normally": terminates,
        "recovered_terminating_normally_pct": round(100.0 * terminates / len(rec), 1) if rec else None,
        "still_truncated_at_1024": still_trunc,
        "recovered_item_counts": {
            "n": len(counts),
            "min": min(counts) if counts else None,
            "median": statistics.median(counts) if counts else None,
            "mean": round(statistics.mean(counts), 2) if counts else None,
            "max": max(counts) if counts else None},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["health", "full"], required=True)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--resamples", type=int, default=10000)
    a = ap.parse_args()

    universe = set(read_manifest_keys(ROOT / "manifests/pilot100_manifest.csv"))
    health_path = ROOT / "analysis/relation_token_budget_health_FROZEN.json"

    models = [m for m in SRC_1024 if SRC_1024[m].exists()]
    missing = [m for m in SRC_1024 if not SRC_1024[m].exists()]

    if a.phase == "health":
        H = {"config_sha256": "67f6e180d4a6aa61e2785238cc84dc6f410f2c71795ad3d5575beeeb698d771a",
             "N_slides": len(universe), "models_present": models, "models_missing": missing,
             "gate_thresholds": GATE, "health": {}}
        for m in models:
            H["health"][m] = {}
            for mode in MODES:
                H["health"][m][mode] = {
                    "512": health(index(SRC_512[m], mode, universe)),
                    "1024": health(index(SRC_1024[m], mode, universe))}
            for budget in ("512", "1024"):
                cells = [H["health"][m][mo][budget] for mo in MODES]
                tot = sum(c["N"] for c in cells)
                val = sum(c["technically_valid"] for c in cells)
                deg = sum(c["degenerate_repetition"] for c in cells)
                err = sum(c["generation_errors"] for c in cells)
                worst = min(c["validity_pct"] for c in cells)
                agg = 100.0 * val / tot if tot else 0.0
                reasons = []
                if agg < GATE["aggregate_min_pct"]: reasons.append(f"aggregate {agg:.2f}% < 95%")
                if worst < GATE["per_cell_min_pct"]: reasons.append(f"worst cell {worst:.2f}% < 90%")
                if err: reasons.append(f"{err} generation errors")
                if 100.0*deg/tot > GATE["degenerate_max_pct"]: reasons.append("degenerate > 1%")
                H["health"][m][f"GATE_{budget}"] = {
                    "N": tot, "aggregate_validity_pct": round(agg, 2),
                    "worst_cell_validity_pct": worst, "generation_errors": err,
                    "degenerate": deg,
                    "GATE": "PASS" if not reasons else "FAIL", "fail_reasons": reasons,
                    "note": "RELATIONS-ONLY gate over MM/TX/IMG. The frozen Stage-7 model-level "
                            "gate additionally included the concepts task."}
        health_path.write_text(json.dumps(H, indent=2, sort_keys=True) + "\n")
        print("FROZEN health written ->", health_path)
        for m in models:
            for b in ("512", "1024"):
                g = H["health"][m][f"GATE_{b}"]
                print(f"  {m:26} {b:>4}: valid={g['aggregate_validity_pct']:6.2f}% "
                      f"worst={g['worst_cell_validity_pct']:6.2f}% err={g['generation_errors']} "
                      f"-> {g['GATE']}" + ("  " + "; ".join(g["fail_reasons"]) if g["fail_reasons"] else ""))
        return 0

    if not health_path.exists():
        raise SystemExit("REFUSED: output-health statistics must be computed and frozen "
                         "(--phase health) before any overlap is inspected.")
    H = json.loads(health_path.read_text())

    out = {"config_sha256": H["config_sha256"], "health_frozen_first": True,
           "N_slides": H["N_slides"], "models": models, "models_missing": missing,
           "bootstrap": {"kind": "lecture-level cluster bootstrap",
                         "n_resamples": a.resamples, "seed": a.seed},
           "A_technical_health": H["health"], "B_512_valid_stability": {},
           "C_512_invalid_recovery": {}, "D_source_ordering_at_1024_DIAGNOSTIC_ONLY": {}}

    for m in models:
        out["B_512_valid_stability"][m] = {}
        out["C_512_invalid_recovery"][m] = {}
        for mode in MODES:
            A = index(SRC_512[m], mode, universe)
            B = index(SRC_1024[m], mode, universe)
            out["B_512_valid_stability"][m][mode] = stability(A, B, a.seed, a.resamples)
            out["C_512_invalid_recovery"][m][mode] = recovery(A, B)

        idx = {mo: index(SRC_1024[m], mo, universe) for mo in MODES}
        uni = sorted(universe)
        d = {}
        for l, r in (("MM", "TX"), ("MM", "IMG"), ("TX", "IMG")):
            pairs, bn = [], []
            nvalid = 0
            for k in uni:
                x, y = idx[l].get(k), idx[r].get(k)
                if not (is_valid(x) and is_valid(y)):
                    continue
                nvalid += 1
                sx, sy = item_set(TASK, x.get("parsed_output")), item_set(TASK, y.get("parsed_output"))
                j = _jaccard(sx, sy)
                if j is not None:
                    pairs.append((k[0], j))
                    if sx and sy:
                        bn.append((k[0], j))
            d[f"{l}-{r}"] = {"N_pair_valid": nvalid, "N_scored": len(pairs),
                             "validity_aware_jaccard": cluster_bootstrap_mean(pairs, a.seed, a.resamples),
                             "both_nonempty_jaccard": cluster_bootstrap_mean(bn, a.seed, a.resamples)}
        means = {k: v["validity_aware_jaccard"]["mean"] for k, v in d.items()
                 if v["validity_aware_jaccard"]["mean"] is not None}
        d["observed_descending"] = sorted(means, key=lambda k: -means[k]) if means else []
        d["matches_v2_MM_TX_gt_MM_IMG_gt_TX_IMG"] = d["observed_descending"] == ["MM-TX", "MM-IMG", "TX-IMG"]
        out["D_source_ordering_at_1024_DIAGNOSTIC_ONLY"][m] = d

    p = ROOT / "analysis/relation_token_budget_sensitivity_pilot100.json"
    p.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
