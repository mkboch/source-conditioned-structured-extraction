#!/usr/bin/env python3
"""Source axis (MM/TX/IMG) + cross-model agreement, using the identical comparator as every
other perturbation axis. Works for the frozen V2 baselines and for any new V3 model."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from v3_metrics import (read_jsonl, index_records, read_manifest_keys,
                        compare_conditions, fmt_ci)

TASKS = ("concepts", "triples")
TASK_LABEL = {"concepts": "concepts", "triples": "relations"}
MODE_PAIRS = (("MM", "TX"), ("MM", "IMG"), ("TX", "IMG"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True, help='JSON {"models":{name:records_path}}')
    ap.add_argument("--manifest", type=Path, default=ROOT / "manifests/milu1062_manifest.csv")
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--resamples", type=int, default=10000)
    a = ap.parse_args()

    spec = json.loads(a.spec.read_text())
    uni = read_manifest_keys(a.manifest)
    rows = {m: read_jsonl(Path(p)) for m, p in spec["models"].items()}

    source, cross, ordering = [], [], []
    for model, rs in rows.items():
        for task in TASKS:
            idx = {mo: index_records(rs, mode=mo, task=task) for mo in ("MM", "TX", "IMG")}
            means = {}
            for l, r in MODE_PAIRS:
                c = compare_conditions(idx[l], idx[r], task, uni, a.seed, a.resamples,
                                       label=f"{l}_vs_{r}")
                c.update({"model": model, "task_label": TASK_LABEL[task], "left": l, "right": r})
                source.append(c)
                means[f"{l}-{r}"] = c["union_nonempty_jaccard"]["mean"]
                print(f"{model:26} {TASK_LABEL[task]:9} {l}-{r:<4} "
                      f"{fmt_ci(c['union_nonempty_jaccard'])} N={c['N_primary_scored']}", flush=True)
            obs = sorted(means, key=lambda k: -means[k])
            ordering.append({"model": model, "task_label": TASK_LABEL[task], "means": means,
                             "observed_descending": obs,
                             "matches_v2_MM_TX_gt_MM_IMG_gt_TX_IMG":
                                 obs == ["MM-TX", "MM-IMG", "TX-IMG"]})

    names = list(rows)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            A, B = names[i], names[j]
            for task in TASKS:
                for mo in ("MM", "TX", "IMG"):
                    c = compare_conditions(index_records(rows[A], mode=mo, task=task),
                                           index_records(rows[B], mode=mo, task=task),
                                           task, uni, a.seed, a.resamples,
                                           label=f"{A}_vs_{B}")
                    c.update({"model_left": A, "model_right": B, "mode": mo,
                              "task_label": TASK_LABEL[task]})
                    cross.append(c)

    payload = {"N_universe": len(uni), "manifest": str(a.manifest),
               "bootstrap": {"kind": "lecture-level cluster bootstrap",
                             "n_resamples": a.resamples, "seed": a.seed},
               "source_axis": source, "source_ordering": ordering,
               "cross_model_fixed_source": cross}
    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    L = ["# Source axis and cross-model agreement", "",
         f"Slide universe: **{len(uni)}**. Lecture-level cluster bootstrap, "
         f"{a.resamples} resamples, seed {a.seed}.", ""]
    for task in TASKS:
        L += [f"## {TASK_LABEL[task]} — evidence-source substitution", "",
              "| model | pair | primary Jaccard [95% CI] | N scored | both-non-empty | conservative | pair-invalid % | co-empty % |",
              "|---|---|---|---|---|---|---|---|"]
        for c in source:
            if c["task"] != task: continue
            L.append(f"| {c['model']} | {c['left']}-{c['right']} | "
                     f"{fmt_ci(c['union_nonempty_jaccard'])} | {c['N_primary_scored']}/{c['N_universe']} | "
                     f"{fmt_ci(c['both_nonempty_jaccard'])} | {fmt_ci(c['conservative_all_slide'])} | "
                     f"{c['pair_invalid_rate_pct']:.2f} | {(c['co_empty_rate_pct_of_valid_pairs'] or 0):.2f} |")
        L.append("")
    L += ["## Observed source ordering per model/task", "",
          "| model | task | descending order | matches V2 (MM-TX > MM-IMG > TX-IMG)? |", "|---|---|---|---|"]
    for o in ordering:
        L.append(f"| {o['model']} | {o['task_label']} | {' > '.join(o['observed_descending'])} | "
                 f"{'yes' if o['matches_v2_MM_TX_gt_MM_IMG_gt_TX_IMG'] else '**NO**'} |")
    L += ["", "## Cross-model agreement at fixed evidence source", "",
          "| model A | model B | task | source | primary Jaccard [95% CI] | N scored |", "|---|---|---|---|---|---|"]
    for c in cross:
        L.append(f"| {c['model_left']} | {c['model_right']} | {c['task_label']} | {c['mode']} | "
                 f"{fmt_ci(c['union_nonempty_jaccard'])} | {c['N_primary_scored']}/{c['N_universe']} |")
    a.out_md.write_text("\n".join(L) + "\n")
    print("wrote", a.out_json, "and", a.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
