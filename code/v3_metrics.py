#!/usr/bin/env python3
"""Unified structural-comparison metrics for the V3 ARR extension.

The item-set construction, text normalisation and Jaccard definitions are copied VERBATIM
from the frozen V2 evaluator `code/evaluate_pilot100.py`
(SHA256 d3c6dea0ccd0c0128afa0164f08f399a936f5b851a7b4a2f8bf0feaf677124fd) so that every
perturbation axis is measured on exactly the same scale as the published V2 source result.

What is added: a single generic "condition A vs condition B" comparator so that the source
axis, the prompt axis, the stochastic axis, the model axis and the resolution axis are all
measured by the identical procedure, plus a LECTURE-LEVEL cluster bootstrap.

Metric policy (frozen, matches PROTOCOL_FREEZE.md):
  * PRIMARY  `union_nonempty_jaccard` - mean over slides where BOTH sides are valid AND the
    union of the two item sets is non-empty. Valid co-empty pairs are EXCLUDED, never scored
    as agreement. Invalid outputs are EXCLUDED, never silently treated as an empty structure.
  * SECONDARY `both_nonempty_jaccard` - restricted further to slides where both sides emit
    at least one item.
  * SENSITIVITY `conservative_all_slide` - every slide in the cohort contributes; invalid
    pairs score 0 and valid co-empty pairs score 0.
  * Uncertainty: cluster bootstrap resampling LECTURES with replacement (22 clusters).
"""
from __future__ import annotations

import csv
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------- frozen V2 definitions
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
        normalize_text(
            f"{item.get('subject', '')}|{item.get('predicate', '')}|{item.get('object', '')}")
        for item in values
        if isinstance(item, dict) and item.get("subject") and item.get("predicate")
        and item.get("object")
    }


def output_count(task: str, parsed: Any) -> int:
    if not isinstance(parsed, dict):
        return 0
    values = parsed.get("concepts" if task == "concepts" else "triples")
    return len(values) if isinstance(values, list) else 0


def _jaccard(a: set[str], b: set[str]) -> float | None:
    union = a | b
    return len(a & b) / len(union) if union else None


def is_valid(rec: dict | None) -> bool:
    return bool(rec) and bool(rec.get("json_parse_success")) and bool(rec.get("schema_valid")) \
        and not rec.get("generation_error")


# ---------------------------------------------------------------- IO
def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not Path(path).exists():
        return rows
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def index_records(rows: Iterable[dict], mode: str | None = None,
                  task: str | None = None) -> dict[tuple[str, str], dict]:
    """Index by (lecture, slide_id). Later records win, so a resumed/retried record
    supersedes an earlier failed attempt while the failed attempt stays on disk."""
    out: dict[tuple[str, str], dict] = {}
    for r in rows:
        if mode is not None and r.get("mode") != mode:
            continue
        if task is not None and r.get("task") != task:
            continue
        out[(r.get("lecture", ""), r.get("slide_id", ""))] = r
    return out


def read_manifest_keys(path: Path) -> list[tuple[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        return [(r["lecture"], r["slide_id"]) for r in csv.DictReader(fh)]


# ---------------------------------------------------------------- cluster bootstrap
def cluster_bootstrap_mean(per_slide: list[tuple[str, float]], seed: int,
                           n_resamples: int = 10000) -> dict[str, float | None]:
    """Resample LECTURES (clusters) with replacement; recompute the slide-level mean each time.

    Slides within a lecture are not independent, so a slide-level bootstrap understates
    uncertainty. Clusters are the 22 lectures.
    """
    if not per_slide:
        return {"mean": None, "ci_low": None, "ci_high": None, "n_slides": 0, "n_clusters": 0}
    by_lecture: dict[str, list[float]] = defaultdict(list)
    for lec, val in per_slide:
        by_lecture[lec].append(val)
    lectures = sorted(by_lecture)
    rng = random.Random(seed)
    k = len(lectures)
    means = []
    for _ in range(n_resamples):
        pool: list[float] = []
        for _ in range(k):
            pool.extend(by_lecture[lectures[rng.randrange(k)]])
        if pool:
            means.append(sum(pool) / len(pool))
    means.sort()

    def q(p: float) -> float | None:
        if not means:
            return None
        idx = (len(means) - 1) * p
        lo = int(idx)
        hi = min(lo + 1, len(means) - 1)
        return means[lo] * (1 - (idx - lo)) + means[hi] * (idx - lo)

    vals = [v for _, v in per_slide]
    return {"mean": statistics.mean(vals), "ci_low": q(0.025), "ci_high": q(0.975),
            "n_slides": len(vals), "n_clusters": k}


# ---------------------------------------------------------------- the comparator
def compare_conditions(left: dict[tuple[str, str], dict], right: dict[tuple[str, str], dict],
                       task: str, universe: list[tuple[str, str]], seed: int,
                       n_resamples: int = 10000, label: str = "") -> dict[str, Any]:
    """Compare two conditions over a fixed slide universe. See module docstring for policy."""
    n_total = len(universe)
    left_valid = right_valid = both_valid = 0
    left_missing = right_missing = 0
    union_pairs: list[tuple[str, float]] = []      # PRIMARY
    both_pairs: list[tuple[str, float]] = []       # SECONDARY
    cons_pairs: list[tuple[str, float]] = []       # SENSITIVITY (all slides)
    co_empty = one_empty = both_nonempty = 0
    lcounts: list[int] = []
    rcounts: list[int] = []
    diffs: list[tuple[str, float]] = []
    absdiffs: list[tuple[str, float]] = []

    for key in universe:
        lec = key[0]
        lrec, rrec = left.get(key), right.get(key)
        if lrec is None:
            left_missing += 1
        if rrec is None:
            right_missing += 1
        lv, rv = is_valid(lrec), is_valid(rrec)
        left_valid += lv
        right_valid += rv
        if not (lv and rv):
            cons_pairs.append((lec, 0.0))          # invalid -> 0, never "empty structure"
            continue
        both_valid += 1
        ls = item_set(task, lrec.get("parsed_output"))
        rs = item_set(task, rrec.get("parsed_output"))
        lcounts.append(len(ls))
        rcounts.append(len(rs))
        diffs.append((lec, float(len(ls) - len(rs))))
        absdiffs.append((lec, float(abs(len(ls) - len(rs)))))
        j = _jaccard(ls, rs)
        if j is None:                               # valid co-empty
            co_empty += 1
            cons_pairs.append((lec, 0.0))           # never counted as agreement
            continue
        union_pairs.append((lec, j))
        cons_pairs.append((lec, j))
        if ls and rs:
            both_nonempty += 1
            both_pairs.append((lec, j))
        else:
            one_empty += 1

    def pct(n: int, d: int) -> float | None:
        return 100.0 * n / d if d else None

    return {
        "label": label,
        "task": task,
        "N_universe": n_total,
        "N_left_missing": left_missing,
        "N_right_missing": right_missing,
        "N_left_valid": left_valid,
        "N_right_valid": right_valid,
        "left_invalid_rate_pct": pct(n_total - left_valid, n_total),
        "right_invalid_rate_pct": pct(n_total - right_valid, n_total),
        "N_pair_valid": both_valid,
        "N_pair_invalid": n_total - both_valid,
        "pair_invalid_rate_pct": pct(n_total - both_valid, n_total),
        "N_co_empty": co_empty,
        "co_empty_rate_pct_of_valid_pairs": pct(co_empty, both_valid),
        "N_one_empty": one_empty,
        "N_both_nonempty": both_nonempty,
        "N_primary_scored": len(union_pairs),
        "N_both_nonempty_scored": len(both_pairs),
        "N_conservative_scored": len(cons_pairs),
        "mean_items_left": statistics.mean(lcounts) if lcounts else None,
        "mean_items_right": statistics.mean(rcounts) if rcounts else None,
        "union_nonempty_jaccard": cluster_bootstrap_mean(union_pairs, seed, n_resamples),
        "both_nonempty_jaccard": cluster_bootstrap_mean(both_pairs, seed, n_resamples),
        "conservative_all_slide": cluster_bootstrap_mean(cons_pairs, seed, n_resamples),
        "item_count_difference_left_minus_right": cluster_bootstrap_mean(diffs, seed, n_resamples),
        "item_count_absolute_difference": cluster_bootstrap_mean(absdiffs, seed, n_resamples),
    }


def fmt_ci(d: dict) -> str:
    if d is None or d.get("mean") is None:
        return "n/a"
    return f"{d['mean']:.4f} [{d['ci_low']:.4f}, {d['ci_high']:.4f}]"
