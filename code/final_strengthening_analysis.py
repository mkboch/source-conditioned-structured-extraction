from __future__ import annotations

import csv
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/final_strengthening_20260910"
HE = ROOT / "human_eval"

sys.path.insert(0, str(ROOT / "code"))

from v3_metrics import (
    read_jsonl,
    index_records,
    read_manifest_keys,
    item_set,
    is_valid,
    _jaccard,
    cluster_bootstrap_mean,
)

SEED = 20260821
NBOOT = 10000

MODELS = [
    "InternVL3-14B",
    "Gemma-4-31B",
    "Qwen3-VL-32B-Instruct",
]

TASKS = ["concepts", "triples"]
TL = {"concepts": "concepts", "triples": "relations"}
SOURCES = ["MM", "TX", "IMG"]
SOURCE_PAIRS = [("MM","TX"),("MM","IMG"),("TX","IMG")]

MANIFEST = ROOT / "manifests/milu1062_manifest.csv"
UNIVERSE = read_manifest_keys(MANIFEST)
LECTURES = sorted({x[0] for x in UNIVERSE})

RAW = {
    m: read_jsonl(
        ROOT / f"results/final_task_specific/merged/{m}.jsonl"
    )
    for m in MODELS
}

IDX = {
    m: {
        task: {
            mode: index_records(RAW[m], mode=mode, task=task)
            for mode in SOURCES
        }
        for task in TASKS
    }
    for m in MODELS
}

def qtile(vals, p):
    vals = sorted(vals)
    if not vals:
        return None
    x = (len(vals) - 1) * p
    lo = int(x)
    hi = min(lo + 1, len(vals) - 1)
    f = x - lo
    return vals[lo] * (1-f) + vals[hi] * f

def mean_or_none(vals):
    return statistics.mean(vals) if vals else None

def fmt(x):
    if x is None:
        return "n/a"
    return f"{x:.4f}"

def metric_map(A, B, task):
    out = {}
    for key in UNIVERSE:
        a = A.get(key)
        b = B.get(key)
        if not (is_valid(a) and is_valid(b)):
            continue
        sa = item_set(task, a.get("parsed_output"))
        sb = item_set(task, b.get("parsed_output"))
        j = _jaccard(sa, sb)
        if j is not None:
            out[key] = j
    return out

def cluster_boot_values(vals, seed=SEED, n=NBOOT):
    return cluster_bootstrap_mean(vals, seed, n)

def paired_cluster_delta(src_map, mdl_map, seed=SEED, n=NBOOT):
    src_by = defaultdict(list)
    mdl_by = defaultdict(list)

    for (lec, slide), v in src_map.items():
        src_by[lec].append(v)

    for (lec, slide), v in mdl_map.items():
        mdl_by[lec].append(v)

    point_src = statistics.mean(src_map.values())
    point_mdl = statistics.mean(mdl_map.values())
    point_delta = point_src - point_mdl

    rng = random.Random(seed)
    boot = []

    for _ in range(n):
        sp = []
        mp = []

        for _ in range(len(LECTURES)):
            lec = LECTURES[rng.randrange(len(LECTURES))]
            sp.extend(src_by.get(lec, []))
            mp.extend(mdl_by.get(lec, []))

        if sp and mp:
            boot.append(
                statistics.mean(sp) - statistics.mean(mp)
            )

    boot.sort()

    lo = qtile(boot, .025)
    hi = qtile(boot, .975)

    p_lt0 = sum(x < 0 for x in boot) / len(boot)
    p_gt0 = sum(x > 0 for x in boot) / len(boot)

    if hi < 0:
        verdict = "SOURCE_MORE_DISRUPTIVE"
    elif lo > 0:
        verdict = "MODEL_MORE_DISRUPTIVE"
    else:
        verdict = "UNRESOLVED"

    common = sorted(set(src_map) & set(mdl_map))

    common_vals = [
        (k[0], src_map[k] - mdl_map[k])
        for k in common
    ]

    common_boot = cluster_boot_values(
        common_vals,
        seed=seed,
        n=n,
    )

    return {
        "source_agreement": point_src,
        "model_agreement": point_mdl,
        "delta_source_minus_model": point_delta,
        "ci_low": lo,
        "ci_high": hi,
        "bootstrap_p_delta_lt_0": p_lt0,
        "bootstrap_p_delta_gt_0": p_gt0,
        "verdict": verdict,
        "N_source_scored": len(src_map),
        "N_model_scored": len(mdl_map),
        "N_common_scored": len(common),
        "common_slide_delta": common_boot,
    }

# ---------------------------------------------------------
# A. NORMALIZED UNIQUE ITEM YIELD
# ---------------------------------------------------------

yield_rows = []

for m in MODELS:
    for task in TASKS:
        for mode in SOURCES:
            idx = IDX[m][task][mode]

            vals = []

            for key in UNIVERSE:
                r = idx.get(key)
                if not is_valid(r):
                    continue
                vals.append(
                    len(item_set(task, r.get("parsed_output")))
                )

            yield_rows.append({
                "model": m,
                "task": TL[task],
                "source": mode,
                "N_universe": len(UNIVERSE),
                "N_valid": len(vals),
                "mean_unique_items": mean_or_none(vals),
                "median_unique_items": statistics.median(vals) if vals else None,
                "q25": qtile(vals, .25),
                "q75": qtile(vals, .75),
                "zero_rate_pct_valid": (
                    100 * sum(v == 0 for v in vals) / len(vals)
                    if vals else None
                ),
            })

# ---------------------------------------------------------
# B. DIRECTED CONTAINMENT
# ---------------------------------------------------------

contain_rows = []

for m in MODELS:
    for task in TASKS:
        for left, right in SOURCE_PAIRS:
            A = IDX[m][task][left]
            B = IDX[m][task][right]

            jac = []
            lr = []
            rl = []

            for key in UNIVERSE:
                a = A.get(key)
                b = B.get(key)

                if not (is_valid(a) and is_valid(b)):
                    continue

                sa = item_set(task, a.get("parsed_output"))
                sb = item_set(task, b.get("parsed_output"))

                inter = len(sa & sb)

                j = _jaccard(sa, sb)

                if j is not None:
                    jac.append((key[0], j))

                if sa:
                    lr.append((key[0], inter / len(sa)))

                if sb:
                    rl.append((key[0], inter / len(sb)))

            contain_rows.append({
                "model": m,
                "task": TL[task],
                "pair": f"{left}-{right}",
                "jaccard": cluster_boot_values(jac),
                f"{left}_contained_in_{right}": cluster_boot_values(lr),
                f"{right}_contained_in_{left}": cluster_boot_values(rl),
                "N_pair_valid": sum(
                    1 for key in UNIVERSE
                    if is_valid(A.get(key)) and is_valid(B.get(key))
                ),
                f"N_{left}_denominator_nonempty": len(lr),
                f"N_{right}_denominator_nonempty": len(rl),
            })

# ---------------------------------------------------------
# C. PAIRED SOURCE-VS-MODEL BOOTSTRAP
# ---------------------------------------------------------

other = {
    ("InternVL3-14B", "concepts"): "Qwen3-VL-32B-Instruct",
    ("InternVL3-14B", "triples"): "Qwen3-VL-32B-Instruct",
    ("Gemma-4-31B", "concepts"): "Qwen3-VL-32B-Instruct",
    ("Gemma-4-31B", "triples"): "Qwen3-VL-32B-Instruct",
    ("Qwen3-VL-32B-Instruct", "concepts"): "InternVL3-14B",
    ("Qwen3-VL-32B-Instruct", "triples"): "Gemma-4-31B",
}

paired_rows = []

for m in MODELS:
    for task in TASKS:
        om = other[(m, task)]

        source_map = metric_map(
            IDX[m][task]["MM"],
            IDX[m][task]["IMG"],
            task,
        )

        model_map = metric_map(
            IDX[m][task]["MM"],
            IDX[om][task]["MM"],
            task,
        )

        r = paired_cluster_delta(
            source_map,
            model_map,
        )

        r.update({
            "model": m,
            "task": TL[task],
            "source_comparison": f"{m}: MM vs IMG",
            "model_comparison": f"{m} vs {om}: MM",
        })

        paired_rows.append(r)

# ---------------------------------------------------------
# D. QWEN DETERMINISTIC RAW REPEAT
# ---------------------------------------------------------

repeat_rows = []

for task, suffix in [
    ("concepts", "concepts512_MM"),
    ("triples", "relations1024_MM"),
]:
    base = IDX["Qwen3-VL-32B-Instruct"][task]["MM"]

    rep = index_records(
        read_jsonl(
            ROOT /
            "results/qwen_axis_completion_20260903" /
            f"qwen_P0_repeat_{suffix}" /
            "records.jsonl"
        ),
        mode="MM",
        task=task,
    )

    raw_same = 0
    struct_same = 0
    common = 0

    for key in UNIVERSE:
        a = base.get(key)
        b = rep.get(key)

        if a is None or b is None:
            continue

        common += 1

        if (a.get("raw_output") or "") == (b.get("raw_output") or ""):
            raw_same += 1

        if (
            is_valid(a) and is_valid(b) and
            item_set(task, a.get("parsed_output")) ==
            item_set(task, b.get("parsed_output"))
        ):
            struct_same += 1

    repeat_rows.append({
        "task": TL[task],
        "N_common": common,
        "N_raw_identical": raw_same,
        "raw_identical_pct": 100 * raw_same / common,
        "N_structural_set_identical": struct_same,
        "structural_set_identical_pct": 100 * struct_same / common,
    })

# ---------------------------------------------------------
# E. THREE-MODEL PROMPT VS STOCHASTIC SUMMARY
# ---------------------------------------------------------

def first_existing(names):
    for name in names:
        p = ROOT / "analysis" / name
        if p.exists():
            return p
    raise FileNotFoundError(names)

old_prompt_c = json.loads(
    first_existing([
        "prompt_sensitivity_full1062_CONCEPTS_FINAL.json",
        "prompt_sensitivity_full1062.json",
    ]).read_text()
)

old_prompt_r = json.loads(
    first_existing([
        "prompt_sensitivity_relations_TASK_SPECIFIC_FINAL.json",
    ]).read_text()
)

old_stoch_c = json.loads(
    first_existing([
        "stochastic_sensitivity_full1062_CONCEPTS_FINAL.json",
        "stochastic_sensitivity_full1062.json",
    ]).read_text()
)

old_stoch_r = json.loads(
    first_existing([
        "stochastic_sensitivity_relations_TASK_SPECIFIC_FINAL.json",
    ]).read_text()
)

q_prompt_c = json.loads(
    (OUT / "qwen_prompt_concepts.json").read_text()
)
q_prompt_r = json.loads(
    (OUT / "qwen_prompt_relations.json").read_text()
)
q_stoch_c = json.loads(
    (OUT / "qwen_stochastic_concepts.json").read_text()
)
q_stoch_r = json.loads(
    (OUT / "qwen_stochastic_relations.json").read_text()
)

def collect(ds, model, task, axis):
    vals = []

    for d in ds:
        for c in d["comparisons"]:
            if c["model"] != model:
                continue
            if c["task_label"] != task:
                continue

            if axis == "prompt":
                if {c["left"], c["right"]} <= {"P0","P1","P2"}:
                    vals.append(
                        c["union_nonempty_jaccard"]["mean"]
                    )

            elif axis == "stochastic":
                if (
                    c["left"] in {"S1","S2","S3"} and
                    c["right"] in {"S1","S2","S3"}
                ):
                    vals.append(
                        c["union_nonempty_jaccard"]["mean"]
                    )

    return vals

three_model_rows = []

for m in MODELS:
    for task in ["concepts", "relations"]:
        if task == "concepts":
            pd = [old_prompt_c, q_prompt_c]
            sd = [old_stoch_c, q_stoch_c]
        else:
            pd = [old_prompt_r, q_prompt_r]
            sd = [old_stoch_r, q_stoch_r]

        pv = collect(pd, m, task, "prompt")
        sv = collect(sd, m, task, "stochastic")

        pm = mean_or_none(pv)
        sm = mean_or_none(sv)

        if pm is None or sm is None:
            verdict = "MISSING"
        elif pm < sm:
            verdict = "PROMPT_MORE_DISRUPTIVE"
        elif sm < pm:
            verdict = "STOCHASTIC_MORE_DISRUPTIVE"
        else:
            verdict = "EQUAL"

        three_model_rows.append({
            "model": m,
            "task": task,
            "N_prompt_pairwise": len(pv),
            "N_stochastic_seed_pairwise": len(sv),
            "mean_prompt_agreement": pm,
            "mean_stochastic_agreement": sm,
            "prompt_minus_stochastic_agreement": (
                pm - sm if pm is not None and sm is not None else None
            ),
            "verdict": verdict,
        })

# ---------------------------------------------------------
# F. HUMAN TRANSITIVE-CLOSURE SENSITIVITY
# ---------------------------------------------------------

human = {
    "available": False,
}

def maximum_matching(left_nodes, right_nodes, edges):
    adj = {
        x: [y for y in right_nodes if (x,y) in edges]
        for x in left_nodes
    }

    match_r = {}

    def aug(x, seen):
        for y in adj.get(x, []):
            if y in seen:
                continue
            seen.add(y)

            if y not in match_r or aug(match_r[y], seen):
                match_r[y] = x
                return True

        return False

    matched = 0

    for x in left_nodes:
        if aug(x, set()):
            matched += 1

    return matched

try:
    def first_pass(annot):
        ann = json.loads(
            (
                HE /
                f"completed/annotator_{annot}/annotations_annotator_{annot}.json"
            ).read_text()
        )

        meta = {
            i["item_id"]: i
            for i in json.loads(
                (
                    HE /
                    f"sealed_mapping/annotator_{annot}_mapping.json"
                ).read_text()
            )["items"]
        }

        cl = {}

        for pid, rec in ann["panels"].items():
            for iid, lab in (rec.get("clusters") or {}).items():
                lab = (lab or "").strip()

                if lab and iid in meta:
                    cl[iid] = f"{pid}::{lab}"

        return cl, meta

    clA, metaA = first_pass("A")
    clB, metaB = first_pass("B")

    kA = {
        i: (
            m["model"],
            m["task"],
            m["lecture"],
            m["slide_id"],
            m["source"],
            m["normalized_key"],
        )
        for i,m in metaA.items()
    }

    kB = {
        i: (
            m["model"],
            m["task"],
            m["lecture"],
            m["slide_id"],
            m["source"],
            m["normalized_key"],
        )
        for i,m in metaB.items()
    }

    def decisions(cl, keymap):
        by = defaultdict(list)

        for iid, gid in cl.items():
            k = keymap[iid]
            by[(k[0],k[1],k[2],k[3])].append(
                (k[5],k[4],gid)
            )

        d = {}

        from itertools import combinations

        for cell, items in by.items():
            for x,y in combinations(sorted(items), 2):
                if x[1] == y[1]:
                    continue
                d[(cell,x[0],y[0])] = (
                    1 if x[2] == y[2] else 0
                )

        return d

    dA = decisions(clA, kA)
    dB = decisions(clB, kB)

    agreed_yes = {
        k for k in dA
        if dB.get(k) == 1 and dA[k] == 1
    }

    ordered = [
        k for k in sorted(set(dA) & set(dB), key=str)
        if dA[k] != dB[k]
    ]

    nodes = defaultdict(lambda: defaultdict(set))

    for iid, m in metaA.items():
        cell = (
            m["model"],
            m["task"],
            m["lecture"],
            m["slide_id"],
        )

        nodes[cell][m["source"]].add(
            m["normalized_key"]
        )

    def load_variant(v):
        rows = list(
            csv.DictReader(
                open(
                    HE /
                    f"adjudication/returned/semantic_variant_{v}.csv",
                    encoding="utf-8-sig",
                )
            )
        )

        return {
            r["case_id"]:
            (r.get("consensus_equivalent__FILL_1_or_0") or "").strip()
            for r in rows
        }

    def yes_for_variant(v):
        yes = set(agreed_yes)
        cons = load_variant(v)

        for i,k in enumerate(ordered, 1):
            if cons.get(f"S{i:05d}") == "1":
                yes.add(k)

        return yes

    def closure_scores(yes):
        adj = defaultdict(lambda: defaultdict(set))

        allnodes = defaultdict(set)

        for cell, srcs in nodes.items():
            for src, vals in srcs.items():
                for nk in vals:
                    allnodes[cell].add((nk,src))

        for cell,x,y in yes:
            xs = [
                n for n in allnodes[cell]
                if n[0] == x
            ]

            ys = [
                n for n in allnodes[cell]
                if n[0] == y
            ]

            for a in xs:
                for b in ys:
                    adj[cell][a].add(b)
                    adj[cell][b].add(a)

        rows = defaultdict(list)

        for cell, ns in allnodes.items():
            seen = {}
            cnum = 0

            for n in sorted(ns):
                if n in seen:
                    continue

                cnum += 1
                stack = [n]

                while stack:
                    u = stack.pop()

                    if u in seen:
                        continue

                    seen[u] = cnum

                    for v in adj[cell].get(u, ()):
                        if v not in seen:
                            stack.append(v)

            bysrc = defaultdict(set)

            for (nk,src), cid in seen.items():
                bysrc[src].add(cid)

            m,t,lec,slide = cell

            for l,r in SOURCE_PAIRS:
                if l not in bysrc or r not in bysrc:
                    continue

                j = _jaccard(
                    bysrc[l],
                    bysrc[r],
                )

                if j is not None:
                    rows[(m,t,f"{l}-{r}")].append(j)

        return {
            k: statistics.mean(v)
            for k,v in rows.items()
        }

    def direct_scores(yes):
        yes_by_cell = defaultdict(set)

        for cell,x,y in yes:
            yes_by_cell[cell].add(
                tuple(sorted((x,y)))
            )

        rows = defaultdict(list)

        for cell, srcs in nodes.items():
            m,t,lec,slide = cell

            for l,r in SOURCE_PAIRS:
                A = sorted(srcs.get(l, set()))
                B = sorted(srcs.get(r, set()))

                if not A and not B:
                    continue

                edges = set()

                for x in A:
                    for y in B:
                        if tuple(sorted((x,y))) in yes_by_cell[cell]:
                            edges.add((x,y))

                matches = maximum_matching(
                    A,
                    B,
                    edges,
                )

                den = len(A) + len(B) - matches

                if den:
                    j = matches / den
                    rows[(m,t,f"{l}-{r}")].append(j)

        return {
            k: statistics.mean(v)
            for k,v in rows.items()
        }

    exact_path = HE / "results_consensus_band.json"

    exact = {}

    if exact_path.exists():
        exact = json.loads(
            exact_path.read_text()
        ).get("exact_structural", {})

    variants = {}

    order_pass = 0
    order_total = 0

    for v in ("r1","r2"):
        yes = yes_for_variant(v)

        cl = closure_scores(yes)
        dr = direct_scores(yes)

        cells = {}

        keys = sorted(set(cl) | set(dr))

        for k in keys:
            m,t,pair = k
            sk = f"{m}|{t}|{pair}"

            cells[sk] = {
                "closure": cl.get(k),
                "direct_only": dr.get(k),
                "closure_minus_direct": (
                    cl[k] - dr[k]
                    if k in cl and k in dr
                    else None
                ),
                "exact_structural": exact.get(sk),
            }

        ordering = {}

        for m in MODELS:
            for t in TASKS:
                vals = {
                    p: dr.get((m,t,p))
                    for p in ("MM-TX","MM-IMG","TX-IMG")
                }

                if all(x is not None for x in vals.values()):
                    ok = (
                        vals["MM-TX"] >
                        vals["MM-IMG"] >
                        vals["TX-IMG"]
                    )

                    order_total += 1
                    order_pass += int(ok)

                    ordering[f"{m}|{TL[t]}"] = {
                        "values": vals,
                        "matches_MM_TX_gt_MM_IMG_gt_TX_IMG": ok,
                    }

        agg = {}

        for t in TASKS:
            cvals = [
                x for (m,tt,p),x in cl.items()
                if tt == t
            ]

            dvals = [
                x for (m,tt,p),x in dr.items()
                if tt == t
            ]

            evals = [
                exact.get(f"{m}|{t}|{p}")
                for m in MODELS
                for p in ("MM-TX","MM-IMG","TX-IMG")
            ]

            evals = [
                x for x in evals
                if x is not None
            ]

            ce = statistics.mean(cvals)
            de = statistics.mean(dvals)
            ee = statistics.mean(evals) if evals else None

            agg[TL[t]] = {
                "exact": ee,
                "closure_semantic": ce,
                "direct_only_semantic": de,
                "closure_effect": ce - de,
                "closure_uplift_over_exact": (
                    ce - ee if ee is not None else None
                ),
                "direct_only_uplift_over_exact": (
                    de - ee if ee is not None else None
                ),
            }

        variants[v] = {
            "cells": cells,
            "direct_only_source_ordering": ordering,
            "aggregate": agg,
        }

    human = {
        "available": True,
        "method": (
            "current connected-components semantic Jaccard versus "
            "direct-edge maximum-cardinality matching sensitivity"
        ),
        "variants": variants,
        "direct_only_ordering_preserved": order_pass,
        "direct_only_ordering_total": order_total,
    }

except Exception as e:
    human = {
        "available": False,
        "error": repr(e),
    }

# ---------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------

payload = {
    "analysis_protocol": {
        "manifest": str(MANIFEST),
        "N_slides": len(UNIVERSE),
        "N_lectures": len(LECTURES),
        "bootstrap_seed": SEED,
        "bootstrap_resamples": NBOOT,
        "note": (
            "All analyses use frozen outputs. "
            "No model inference is performed."
        ),
    },
    "normalized_unique_item_yield": yield_rows,
    "directed_containment": contain_rows,
    "paired_source_vs_model": paired_rows,
    "qwen_deterministic_repeat": repeat_rows,
    "three_model_prompt_vs_stochastic": three_model_rows,
    "human_transitive_closure_sensitivity": human,
}

json_path = OUT / "FINAL_STRENGTHENING_RESULTS.json"
json_path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)

lines = []

lines.append("FINAL STRENGTHENING ANALYSIS")
lines.append("")

lines.append("=== QWEN DETERMINISTIC REPEAT ===")

for r in repeat_rows:
    lines.append(
        f"{r['task']}: "
        f"raw_identical={r['N_raw_identical']}/{r['N_common']} "
        f"({r['raw_identical_pct']:.2f}%), "
        f"structural_set_identical="
        f"{r['N_structural_set_identical']}/{r['N_common']} "
        f"({r['structural_set_identical_pct']:.2f}%)"
    )

lines.append("")
lines.append("=== THREE-MODEL PROMPT VS STOCHASTIC ===")

for r in three_model_rows:
    lines.append(
        f"{r['model']} | {r['task']} | "
        f"prompt_mean={fmt(r['mean_prompt_agreement'])} | "
        f"stochastic_mean={fmt(r['mean_stochastic_agreement'])} | "
        f"prompt-stochastic={fmt(r['prompt_minus_stochastic_agreement'])} | "
        f"{r['verdict']}"
    )

lines.append("")
lines.append("=== NORMALIZED UNIQUE ITEM YIELD ===")

for r in yield_rows:
    lines.append(
        f"{r['model']} | {r['task']} | {r['source']} | "
        f"Nvalid={r['N_valid']} | "
        f"mean={r['mean_unique_items']:.3f} | "
        f"median={r['median_unique_items']:.3f} | "
        f"IQR={r['q25']:.3f}-{r['q75']:.3f} | "
        f"zero={r['zero_rate_pct_valid']:.2f}%"
    )

lines.append("")
lines.append("=== DIRECTED CONTAINMENT ===")

for r in contain_rows:
    pair = r["pair"]
    l,rgt = pair.split("-")

    a = r[f"{l}_contained_in_{rgt}"]
    b = r[f"{rgt}_contained_in_{l}"]

    lines.append(
        f"{r['model']} | {r['task']} | {pair} | "
        f"J={fmt(r['jaccard']['mean'])} | "
        f"{l}->{rgt}={fmt(a['mean'])} "
        f"[{fmt(a['ci_low'])},{fmt(a['ci_high'])}] | "
        f"{rgt}->{l}={fmt(b['mean'])} "
        f"[{fmt(b['ci_low'])},{fmt(b['ci_high'])}]"
    )

lines.append("")
lines.append("=== PAIRED SOURCE VS MODEL BOOTSTRAP ===")

for r in paired_rows:
    lines.append(
        f"{r['model']} | {r['task']} | "
        f"source={r['source_agreement']:.4f} | "
        f"model={r['model_agreement']:.4f} | "
        f"delta={r['delta_source_minus_model']:+.4f} | "
        f"95CI=[{r['ci_low']:+.4f},{r['ci_high']:+.4f}] | "
        f"{r['verdict']} | "
        f"Nsrc={r['N_source_scored']} "
        f"Nmodel={r['N_model_scored']} "
        f"Ncommon={r['N_common_scored']}"
    )

lines.append("")
lines.append("=== HUMAN TRANSITIVE-CLOSURE SENSITIVITY ===")

if human.get("available"):
    for v in ("r1","r2"):
        for task in ("concepts","relations"):
            a = human["variants"][v]["aggregate"][task]

            lines.append(
                f"{v} | {task} | "
                f"exact={fmt(a['exact'])} | "
                f"closure={fmt(a['closure_semantic'])} | "
                f"direct={fmt(a['direct_only_semantic'])} | "
                f"closure_effect={fmt(a['closure_effect'])} | "
                f"uplift_closure={fmt(a['closure_uplift_over_exact'])} | "
                f"uplift_direct={fmt(a['direct_only_uplift_over_exact'])}"
            )

    lines.append(
        "direct-only ordering preserved: "
        f"{human['direct_only_ordering_preserved']}/"
        f"{human['direct_only_ordering_total']}"
    )
else:
    lines.append(
        "HUMAN SENSITIVITY COULD NOT RUN: "
        + human.get("error","unknown")
    )

summary_path = OUT / "RETURN_SUMMARY.txt"
summary_path.write_text(
    "\n".join(lines) + "\n"
)

print("\n".join(lines))
print()
print("wrote", json_path)
print("wrote", summary_path)
