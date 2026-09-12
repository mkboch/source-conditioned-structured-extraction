#!/usr/bin/env python3
"""Evaluate the Stage 3 InternVL3-14B pilot-100 structured outputs."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


MODES = ("MM", "TX", "IMG")
TASKS = ("concepts", "triples")
MODE_PAIRS = (("MM", "TX"), ("MM", "IMG"), ("TX", "IMG"))


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def item_set(task: str, parsed: Any) -> set[str]:
    if not isinstance(parsed, dict):
        return set()
    if task == "concepts":
        values = parsed.get("concepts")
        if not isinstance(values, list):
            return set()
        return {
            normalize_text(f"{item.get('term', '')}|{item.get('category', '')}")
            for item in values
            if isinstance(item, dict) and item.get("term")
        }
    values = parsed.get("triples")
    if not isinstance(values, list):
        return set()
    return {
        normalize_text(f"{item.get('subject', '')}|{item.get('predicate', '')}|{item.get('object', '')}")
        for item in values
        if isinstance(item, dict) and item.get("subject") and item.get("predicate") and item.get("object")
    }


def output_count(task: str, parsed: Any) -> int:
    if not isinstance(parsed, dict):
        return 0
    key = "concepts" if task == "concepts" else "triples"
    values = parsed.get(key)
    return len(values) if isinstance(values, list) else 0


def jaccard(left: set[str], right: set[str], empty_empty_value: float | None = None) -> float | None:
    if not left and not right:
        return empty_empty_value
    union = left | right
    if not union:
        return None
    return len(left & right) / len(union)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else 0.0


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def bootstrap_mean_ci(values: list[float], seed: int, n_resamples: int = 10000) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "ci_low": None, "ci_high": None}
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(sample))
    return {
        "mean": statistics.mean(values),
        "ci_low": percentile(means, 0.025),
        "ci_high": percentile(means, 0.975),
    }


def lexical_occurrence(task: str, parsed: Any, transcript: str) -> dict[str, int | float]:
    items = []
    transcript_norm = normalize_text(transcript)
    if task == "concepts" and isinstance(parsed, dict):
        items = [item.get("term", "") for item in parsed.get("concepts", []) if isinstance(item, dict)]
    elif task == "triples" and isinstance(parsed, dict):
        for item in parsed.get("triples", []):
            if isinstance(item, dict):
                items.extend([item.get("subject", ""), item.get("object", "")])
    terms = [normalize_text(item) for item in items if normalize_text(item)]
    hits = sum(1 for term in terms if term and term in transcript_norm)
    return {"checked_terms": len(terms), "lexical_transcript_occurrences": hits, "rate": hits / len(terms) if terms else 0.0}


def summarize(records: list[dict[str, Any]], manifest_rows: list[dict[str, str]], bootstrap_seed: int) -> dict[str, Any]:
    by_key = {}
    for rec in records:
        key = (rec.get("lecture"), rec.get("slide_id"), rec.get("mode"), rec.get("task"))
        by_key[key] = rec

    availability = []
    volumes = []
    lexical = []
    for mode in MODES:
        for task in TASKS:
            subset = [rec for rec in records if rec.get("mode") == mode and rec.get("task") == task]
            counts = [output_count(task, rec.get("parsed_output")) for rec in subset if rec.get("schema_valid")]
            nonempty = sum(1 for count in counts if count > 0)
            empty_valid = sum(1 for count in counts if count == 0)
            failures = sum(1 for rec in subset if not rec.get("json_parse_success") or not rec.get("schema_valid"))
            availability.append(
                {
                    "mode": mode,
                    "task": task,
                    "generations": len(subset),
                    "nonempty_outputs": nonempty,
                    "empty_valid_outputs": empty_valid,
                    "parse_or_schema_failures": failures,
                    "mean_emitted": statistics.mean(counts) if counts else 0.0,
                    "median_emitted": statistics.median(counts) if counts else 0.0,
                    "distribution": {str(v): counts.count(v) for v in sorted(set(counts))},
                }
            )
            volumes.extend(
                {
                    "mode": mode,
                    "task": task,
                    "lecture": r.get("lecture"),
                    "slide_id": r.get("slide_id"),
                    "count": output_count(task, r.get("parsed_output")),
                }
                for r in subset
                if r.get("json_parse_success") and r.get("schema_valid")
            )

    transcript_by_slide = {}
    for row in manifest_rows:
        try:
            transcript_by_slide[(row["lecture"], row["slide_id"])] = Path(row["text_path"]).read_text(encoding="utf-8", errors="replace")
        except Exception:
            transcript_by_slide[(row["lecture"], row["slide_id"])] = ""

    for rec in records:
        if rec.get("mode") in ("MM", "TX") and rec.get("schema_valid"):
            transcript = transcript_by_slide.get((rec.get("lecture"), rec.get("slide_id")), "")
            lexical.append(
                {
                    "lecture": rec.get("lecture"),
                    "slide_id": rec.get("slide_id"),
                    "mode": rec.get("mode"),
                    "task": rec.get("task"),
                    **lexical_occurrence(rec.get("task"), rec.get("parsed_output"), transcript),
                }
            )

    overlap = []
    bootstrap = []
    for task in TASKS:
        for left_mode, right_mode in MODE_PAIRS:
            slide_stats = []
            for row in manifest_rows:
                left = by_key.get((row["lecture"], row["slide_id"], left_mode, task), {})
                right = by_key.get((row["lecture"], row["slide_id"], right_mode, task), {})
                left_valid = (
                    bool(left)
                    and bool(left.get("json_parse_success"))
                    and bool(left.get("schema_valid"))
                )
                right_valid = (
                    bool(right)
                    and bool(right.get("json_parse_success"))
                    and bool(right.get("schema_valid"))
                )

                if not (left_valid and right_valid):
                    slide_stats.append(
                        {
                            "pair_valid": False,
                            "left_valid": left_valid,
                            "right_valid": right_valid,
                        }
                    )
                    continue

                left_set = item_set(task, left.get("parsed_output"))
                right_set = item_set(task, right.get("parsed_output"))

                slide_stats.append(
                    {
                        "pair_valid": True,
                        "left_valid": True,
                        "right_valid": True,
                        "left_count": len(left_set),
                        "right_count": len(right_set),
                        "both_empty": not left_set and not right_set,
                        "one_empty": (not left_set and bool(right_set))
                        or (bool(left_set) and not right_set),
                        "both_nonempty": bool(left_set) and bool(right_set),
                        "legacy": jaccard(left_set, right_set, 1.0),
                        "union_nonempty": jaccard(left_set, right_set, None),
                        "both_nonempty_j": (
                            jaccard(left_set, right_set, None)
                            if left_set and right_set
                            else None
                        ),
                    }
                )

            n_total = len(slide_stats)
            valid_stats = [x for x in slide_stats if x.get("pair_valid")]
            n_valid = len(valid_stats)
            n_invalid = n_total - n_valid

            legacy_values = [
                x["legacy"] for x in valid_stats if x["legacy"] is not None
            ]
            union_values = [
                x["union_nonempty"]
                for x in valid_stats
                if x["union_nonempty"] is not None
            ]
            both_values = [
                x["both_nonempty_j"]
                for x in valid_stats
                if x["both_nonempty_j"] is not None
            ]
            overlap.append(
                {
                    "task": task,
                    "pair": f"{left_mode}_vs_{right_mode}",
                    "N": n_valid,
                    "N_total": n_total,
                    "N_valid_pair": n_valid,
                    "N_invalid_pair": n_invalid,
                    "invalid_pair_rate": pct(n_invalid, n_total),
                    "both_empty": sum(1 for x in valid_stats if x["both_empty"]),
                    "one_empty": sum(1 for x in valid_stats if x["one_empty"]),
                    "both_nonempty": sum(1 for x in valid_stats if x["both_nonempty"]),
                    "both_empty_rate": pct(
                        sum(1 for x in valid_stats if x["both_empty"]), n_valid
                    ),
                    "one_empty_rate": pct(
                        sum(1 for x in valid_stats if x["one_empty"]), n_valid
                    ),
                    "both_nonempty_rate": pct(
                        sum(1 for x in valid_stats if x["both_nonempty"]), n_valid
                    ),
                    "legacy_all_slide_jaccard": statistics.mean(legacy_values) if legacy_values else None,
                    "union_nonempty_jaccard": statistics.mean(union_values) if union_values else None,
                    "both_nonempty_jaccard": statistics.mean(both_values) if both_values else None,
                }
            )
            bootstrap.append(
                {
                    "task": task,
                    "pair": f"{left_mode}_vs_{right_mode}",
                    "quantity": "union_nonempty_jaccard",
                    **bootstrap_mean_ci(union_values, bootstrap_seed),
                }
            )
            bootstrap.append(
                {
                    "task": task,
                    "pair": f"{left_mode}_vs_{right_mode}",
                    "quantity": "both_nonempty_jaccard",
                    **bootstrap_mean_ci(both_values, bootstrap_seed),
                }
            )
            diffs = [
                x["left_count"] - x["right_count"]
                for x in valid_stats
            ]
            bootstrap.append(
                {
                    "task": task,
                    "pair": f"{left_mode}_vs_{right_mode}",
                    "quantity": "output_count_difference",
                    **bootstrap_mean_ci(diffs, bootstrap_seed),
                }
            )

    return {
        "availability": availability,
        "volumes": volumes,
        "structural_overlap": overlap,
        "lexical_transcript_occurrence": lexical,
        "bootstrap_confidence_intervals": bootstrap,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=Path("external_artifacts/experiment_v2/results/pilot100/records.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("external_artifacts/experiment_v2/results/manifests/pilot100_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("external_artifacts/experiment_v2/results/evaluation"))
    parser.add_argument("--bootstrap-seed", type=int, default=20260815)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(args.records)
    manifest_rows = read_manifest(args.manifest)
    summary = summarize(records, manifest_rows, args.bootstrap_seed)
    (args.output_dir / "pilot100_evaluation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
