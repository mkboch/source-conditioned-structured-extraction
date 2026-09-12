#!/usr/bin/env python3
"""Stage 3 — prompt-paraphrase sensitivity, and Stage 5 — stochastic sensitivity.

Both axes use the identical comparator in v3_metrics so the magnitudes are on the same
scale as each other and as the V2 source result.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from v3_metrics import (read_jsonl, index_records, read_manifest_keys,
                        compare_conditions, fmt_ci)

TASKS = ("concepts", "triples")
TASK_LABEL = {"concepts": "concepts", "triples": "relations"}

V2 = Path("external_artifacts/experiment_v2")
BASELINE_P0 = {
    "InternVL3-14B": V2 / "results_full1062_internvl3_14b/full1062/records.jsonl",
    "Gemma-4-31B":   V2 / "results_full1062_gemma4_31b/full1062/records.jsonl",
}


def load(path: Path, mode: str, task: str):
    return index_records(read_jsonl(Path(path)), mode=mode, task=task)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True,
                    help="JSON: {axis, phase, manifest, models:{name:{cond:path}}, pairs:[[a,b]]}")
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--resamples", type=int, default=10000)
    a = ap.parse_args()

    spec = json.loads(a.spec.read_text())
    universe = read_manifest_keys(Path(spec["manifest"]))
    mode = spec.get("mode", "MM")   # default MM preserves prior behaviour exactly
    tasks = tuple(spec.get("tasks", TASKS))  # default both tasks preserves prior behaviour
    results = []

    for model, conds in spec["models"].items():
        srcs = dict(conds)
        for k, v in list(srcs.items()):
            if v == "BASELINE_V2":
                srcs[k] = str(BASELINE_P0[model])
        for task in tasks:
            cache = {c: load(Path(p), mode, task) for c, p in srcs.items()}
            for left, right in spec["pairs"]:
                if left not in cache or right not in cache:
                    continue
                r = compare_conditions(cache[left], cache[right], task, universe,
                                       a.seed, a.resamples, label=f"{left}_vs_{right}")
                r.update({"model": model, "axis": spec["axis"], "phase": spec["phase"],
                          "mode": mode, "left": left, "right": right,
                          "task_label": TASK_LABEL[task]})
                results.append(r)
                print(f"{model:14} {TASK_LABEL[task]:9} {left:>22} vs {right:<22} "
                      f"primary={fmt_ci(r['union_nonempty_jaccard'])} "
                      f"N={r['N_primary_scored']}/{r['N_universe']}", flush=True)

    payload = {"axis": spec["axis"], "phase": spec["phase"],
               "manifest": spec["manifest"], "N_universe": len(universe),
               "bootstrap": {"kind": "lecture-level cluster bootstrap",
                             "n_resamples": a.resamples, "seed": a.seed},
               "metric_policy": {
                   "primary": "union_nonempty_jaccard (both sides valid AND union non-empty; "
                              "invalid excluded, valid co-empty excluded)",
                   "secondary": "both_nonempty_jaccard",
                   "sensitivity": "conservative_all_slide (invalid=0, valid co-empty=0, all slides)"},
               "comparisons": results}
    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    L = [f"# {spec['axis']} — {spec['phase']}", "",
         f"Slide universe: **{len(universe)}** (`{Path(spec['manifest']).name}`). "
         f"Uncertainty: lecture-level cluster bootstrap, {a.resamples} resamples, seed {a.seed}.",
         "",
         "Primary = validity-aware union-non-empty Jaccard: both sides valid AND union "
         "non-empty. Invalid outputs are excluded (never coerced to an empty structure); "
         "valid co-empty pairs are excluded (never scored as agreement).",
         "Conservative = all slides contribute, invalid pairs and valid co-empty pairs score 0.",
         "", "Concepts and relations are reported separately and never pooled.", ""]
    for task in tasks:
        L += [f"## {TASK_LABEL[task]}", "",
              "| model | comparison | primary Jaccard [95% CI] | N scored | both-non-empty [95% CI] | conservative all-slide [95% CI] | pair-invalid % | co-empty % of valid | mean items L / R |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in results:
            if r["task"] != task:
                continue
            L.append(
                f"| {r['model']} | {r['left']} vs {r['right']} | "
                f"{fmt_ci(r['union_nonempty_jaccard'])} | {r['N_primary_scored']}/{r['N_universe']} | "
                f"{fmt_ci(r['both_nonempty_jaccard'])} | {fmt_ci(r['conservative_all_slide'])} | "
                f"{r['pair_invalid_rate_pct']:.2f} | "
                f"{(r['co_empty_rate_pct_of_valid_pairs'] or 0):.2f} | "
                f"{(r['mean_items_left'] or 0):.2f} / {(r['mean_items_right'] or 0):.2f} |")
        L.append("")
    a.out_md.write_text("\n".join(L) + "\n")
    print("wrote", a.out_json, "and", a.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
