#!/usr/bin/env python3
"""Automated specification comparison of the frozen prompt variants P0 / P1 / P2.

Checks, objectively and without any model in the loop, that the three wordings request the
SAME task and the SAME output contract, while differing enough in surface form to count as
a genuine prompt perturbation. Run BEFORE any P1/P2 inference.

Exit code 0 iff every equivalence check passes and every divergence check passes.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"

VARIANTS = {"P0": "v2", "P1": "p1", "P2": "p2"}
TASKS = {"concepts": "concepts", "triples": "triples"}

# Specification elements that MUST be identical across variants.
CATEGORY_LABELS = [
    "modality", "anatomy", "physics", "instrumentation", "reconstruction",
    "signal_processing", "mathematics", "quality_metric", "workflow", "software",
    "data_processing", "algorithm", "ai_ml", "other",
]
PREDICATE_LABELS = [
    "uses", "measures", "represents", "produces", "depends_on",
    "improves", "degrades", "compares_with", "part_of",
]
# Phrases that would be ILLEGAL additions under the Stage 1 constraints.
LENGTH_CONSTRAINT_RE = re.compile(
    r"\b(at most|at least|no more than|no fewer than|maximum of|minimum of|"
    r"exactly \d+|up to \d+|\d+\s*(?:-|\s)?(?:to|or)\s*\d+\s+(?:items|concepts|"
    r"relations|triples)|limit(?:ed)? to)\b", re.I)
EXAMPLE_RE = re.compile(r"\b(for example|e\.g\.|for instance|example:|sample output)\b", re.I)
DOMAIN_HINT_RE = re.compile(
    r"\b(radiolog|MRI|CT scan|ultrasound|X-ray|PET|typical(?:ly) (?:slides|lectures)|"
    r"expert|specialist|domain knowledge|your knowledge of)\b", re.I)


def read(prefix: str, task: str) -> str:
    return (PROMPTS / f"{prefix}_{task}.txt").read_text(encoding="utf-8")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def schema_block(text: str) -> dict:
    """Extract and structurally parse the JSON schema block that defines the contract."""
    m = re.search(r"Schema:\s*(\{.*?\n\})", text, re.DOTALL)
    if not m:
        raise ValueError("no schema block found")
    return json.loads(m.group(1))


def schema_shape(obj):
    """Key/type skeleton of the schema, ignoring the '...' placeholder values."""
    if isinstance(obj, dict):
        return {k: schema_shape(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [schema_shape(v) for v in obj]
    return "STR"


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z_]+", text.lower()))


def content_words(text: str) -> list[str]:
    """Words outside the schema block and outside the fixed label vocabularies."""
    body = re.sub(r"Schema:\s*\{.*?\n\}", " ", text, flags=re.DOTALL)
    fixed = set(CATEGORY_LABELS) | set(PREDICATE_LABELS)
    return [w for w in re.findall(r"[a-z_]+", body.lower()) if w not in fixed]


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 1.0


def main() -> int:
    report: dict = {"variants": {}, "equivalence_checks": [], "divergence_checks": [],
                    "illegal_addition_checks": []}
    failures: list[str] = []

    texts = {(v, t): read(p, t) for v, p in VARIANTS.items() for t in TASKS}

    for (v, t), text in sorted(texts.items()):
        report["variants"][f"{v}_{t}"] = {
            "file": f"prompts/{VARIANTS[v]}_{t}.txt",
            "sha256": sha(text),
            "chars": len(text),
            "words": len(re.findall(r"\S+", text)),
        }

    def check(bucket: str, name: str, ok: bool, detail) -> None:
        report[bucket].append({"check": name, "pass": bool(ok), "detail": detail})
        if not ok:
            failures.append(name)

    for t, out_key in TASKS.items():
        shapes, blocks = {}, {}
        for v in VARIANTS:
            blocks[v] = schema_block(texts[(v, t)])
            shapes[v] = json.dumps(schema_shape(blocks[v]), sort_keys=True)

        # --- Equivalence: identical output contract -------------------------------
        check("equivalence_checks", f"[{t}] schema skeleton identical across P0/P1/P2",
              len(set(shapes.values())) == 1, shapes)
        check("equivalence_checks", f"[{t}] raw schema text identical across P0/P1/P2",
              len({re.search(r"Schema:\s*(\{.*?\n\})", texts[(v, t)], re.DOTALL).group(1)
                   for v in VARIANTS}) == 1, "byte-identical schema block")
        check("equivalence_checks", f"[{t}] top-level output key is '{out_key}' in all",
              all(list(blocks[v].keys()) == [out_key] for v in VARIANTS),
              {v: list(blocks[v].keys()) for v in VARIANTS})
        check("equivalence_checks", f"[{t}] empty-case sentinel {{\"{out_key}\": []}} in all",
              all(f'{{"{out_key}": []}}' in texts[(v, t)] for v in VARIANTS),
              [v for v in VARIANTS if f'{{"{out_key}": []}}' not in texts[(v, t)]])
        check("equivalence_checks", f"[{t}] both placeholders present in all",
              all("<<MODE_DESCRIPTION>>" in texts[(v, t)] and "<<INPUT_BLOCK>>" in texts[(v, t)]
                  for v in VARIANTS), "MODE_DESCRIPTION + INPUT_BLOCK")
        check("equivalence_checks", f"[{t}] placeholders appear exactly once each in all",
              all(texts[(v, t)].count("<<MODE_DESCRIPTION>>") == 1
                  and texts[(v, t)].count("<<INPUT_BLOCK>>") == 1 for v in VARIANTS), "count==1")
        check("equivalence_checks", f"[{t}] 'Input condition:' line identical in all",
              len({re.search(r"Input condition: <<MODE_DESCRIPTION>>", texts[(v, t)]) is not None
                   for v in VARIANTS}) == 1, "same source-conditioning line")
        check("equivalence_checks", f"[{t}] JSON-only / no-fence requirement in all",
              all(re.search(r"valid JSON", texts[(v, t)], re.I)
                  and re.search(r"fence", texts[(v, t)], re.I) for v in VARIANTS), "present")
        check("equivalence_checks", f"[{t}] grounding-in-input-condition requirement in all",
              all(re.search(r"input condition", texts[(v, t)], re.I) for v in VARIANTS), "present")
        check("equivalence_checks", f"[{t}] task noun ('{ 'concept' if t=='concepts' else 'relation' }') present in all",
              all(("concept" if t == "concepts" else "relation") in texts[(v, t)].lower()
                  for v in VARIANTS), "present")
        check("equivalence_checks", f"[{t}] number of guideline bullets identical",
              len({texts[(v, t)].count("\n- ") for v in VARIANTS}) == 1,
              {v: texts[(v, t)].count("\n- ") for v in VARIANTS})

        if t == "concepts":
            check("equivalence_checks", "[concepts] identical category label vocabulary",
                  all(all(lbl in texts[(v, t)] for lbl in CATEGORY_LABELS) for v in VARIANTS),
                  {v: [l for l in CATEGORY_LABELS if l not in texts[(v, t)]] for v in VARIANTS})
            check("equivalence_checks", "[concepts] category set stays OPEN ('such as'/'of the kind'/'appropriate') in all",
                  all(re.search(r"such as|of the kind|are appropriate|are suitable", texts[(v, t)], re.I)
                      for v in VARIANTS),
                  "no variant converts the open label list into a closed enumeration")
            check("equivalence_checks", "[concepts] admin/logistics exclusion present in all",
                  all(re.search(r"administrat|administered|logistics", texts[(v, t)], re.I)
                      for v in VARIANTS), "present")
            check("equivalence_checks", "[concepts] identical concept-type vocabulary",
                  all(all(w in texts[(v, t)].lower() for w in
                          ["technical", "scientific", "clinical", "imaging", "mathematic",
                           "software", "workflow"]) for v in VARIANTS), "present")
        else:
            check("equivalence_checks", "[triples] identical predicate vocabulary",
                  all(all(lbl in texts[(v, t)] for lbl in PREDICATE_LABELS) for v in VARIANTS),
                  {v: [l for l in PREDICATE_LABELS if l not in texts[(v, t)]] for v in VARIANTS})
            check("equivalence_checks", "[triples] no-unsupported-relation requirement in all",
                  all(re.search(r"not support|does not back|not backed", texts[(v, t)], re.I)
                      for v in VARIANTS), "present")
            check("equivalence_checks", "[triples] short-predicate requirement in all",
                  all(re.search(r"short|brief", texts[(v, t)], re.I) for v in VARIANTS), "present")

        # --- Illegal additions ----------------------------------------------------
        for v in VARIANTS:
            body = re.sub(r"Schema:\s*\{.*?\n\}", " ", texts[(v, t)], flags=re.DOTALL)
            check("illegal_addition_checks", f"[{t}][{v}] no worked examples added",
                  not EXAMPLE_RE.search(body), EXAMPLE_RE.findall(body))
            check("illegal_addition_checks", f"[{t}][{v}] no answer-length constraint added",
                  not LENGTH_CONSTRAINT_RE.search(body), LENGTH_CONSTRAINT_RE.findall(body))
            check("illegal_addition_checks", f"[{t}][{v}] no domain hint added",
                  not DOMAIN_HINT_RE.search(body), DOMAIN_HINT_RE.findall(body))

        # --- Divergence: is this actually a perturbation? -------------------------
        for a, b in (("P0", "P1"), ("P0", "P2"), ("P1", "P2")):
            ja = jaccard(set(content_words(texts[(a, t)])), set(content_words(texts[(b, t)])))
            check("divergence_checks", f"[{t}] {a} vs {b} content-word Jaccard < 0.60 "
                                       f"(materially different wording)", ja < 0.60, round(ja, 4))
            # Lines that MUST be identical: the schema block (stripped below) and the
            # source-conditioning scaffold. Every OTHER line must be freshly worded.
            mandatory = {"Input condition: <<MODE_DESCRIPTION>>", "Input:", "<<INPUT_BLOCK>>",
                         "Schema:"}

            def free_lines(txt: str) -> set[str]:
                body = re.sub(r"Schema:\s*\{.*?\n\}", " ", txt, flags=re.DOTALL)
                return {l.strip() for l in body.splitlines() if l.strip()} - mandatory

            shared = free_lines(texts[(a, t)]) & free_lines(texts[(b, t)])
            check("divergence_checks",
                  f"[{t}] {a} vs {b} share NO freely-worded line (scaffold+schema excluded)",
                  not shared, sorted(shared))

    report["overall_pass"] = not failures
    report["failed_checks"] = failures
    out = Path(__file__).resolve().parent.parent / "analysis" / "prompt_specification_equivalence.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    n_eq = sum(c["pass"] for c in report["equivalence_checks"])
    n_di = sum(c["pass"] for c in report["divergence_checks"])
    n_il = sum(c["pass"] for c in report["illegal_addition_checks"])
    print(f"equivalence   {n_eq}/{len(report['equivalence_checks'])}")
    print(f"divergence    {n_di}/{len(report['divergence_checks'])}")
    print(f"no-illegal    {n_il}/{len(report['illegal_addition_checks'])}")
    print("OVERALL:", "PASS" if report["overall_pass"] else "FAIL " + str(failures))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
