# HUMAN EVALUATION PROTOCOL

> **Post-study status note (2026-09-12).** This document is the frozen
> pre-annotation protocol and is preserved as written at protocol freeze.
> Statements below such as **"Status: no human labels exist"** and
> **"Adjudication — NOT performed"** describe the state at freeze time, not
> the final study status. The two independent first-pass annotations and the
> two adjudication variants were subsequently completed under this frozen
> protocol; the repository's scoring code reflects the completed analysis.

**Frozen:** 2026-09-01 UTC, **before any human label was created or observed.**
**Status: no human labels exist.** No annotation has been performed, inferred or fabricated.

## 1. Purpose — two hypotheses

**H1 — Semantic equivalence.** Does exact structural matching underestimate semantic
agreement among MM/TX/IMG extracted concepts and relations? The automated study's primary
metric treats `CT scanner` and `computed tomography scanner` as complete disagreement; H1
measures how much of the measured disagreement is lexical rather than semantic.

**H2 — Evidence attribution.** For MM-extracted items, is the item supported by both
modalities, transcript only, image only, neither, or is it uncertain?

**Explicitly out of scope:** clinical correctness, diagnostic accuracy, educational quality,
model preference, and overall answer quality. Annotators are never asked to judge these.

## 2. Frozen cohort — recovered, NOT resampled

The 30-slide cohort is the **existing** audit cohort from the completed V2/V3 study. It was
recovered and verified, not redrawn.

- **Provenance:** 30 slides sampled from Pilot-100 with seed **20260818**, no output-based
  selection (recorded in `experiment_v2/stage5_audit30_packet/README.txt`).
- **Source manifest:** `experiment_v2/stage5_audit30_packet/metadata/stage5_manual_audit30_manifest.csv`
  SHA256 `d45ac30a844f41b5735913b3e2970334f897be9431aee66939ba41aec08fdd66`
- **Verification performed:**
  - 30 rows, 30 unique `(lecture, slide_id)` keys;
  - all 30 confirmed present in the frozen Pilot-100 manifest;
  - the two independent V2 packets (`stage5_audit30_packet`, `stage6b_gemma4_audit30_packet`)
    contain **identical** slide sets;
  - all **60** source-file SHA256 values (30 images + 30 transcripts) re-verified against
    the files on disk — **60/60 match**.
- **Coverage:** 18 lectures.

The cohort was **not** modified, reduced or resampled.

### Slides

| # | lecture | slide | image SHA256 (first 16) | transcript SHA256 (first 16) |
|---|---|---|---|---|
| 1 | Lecture1 | Slide11 | `feac88c6512a2902` | `96764d84e0ddea65` |
| 2 | Lecture3 | Slide21 | `c71506696c65d9c3` | `3bb29ed587b82608` |
| 3 | Lecture3 | Slide31 | `e58df67b6a24706a` | `8ff91d3d5af91bea` |
| 4 | Lecture4 | Slide1 | `5dc29c2d0be52a1f` | `b5a3aa9168fdf378` |
| 5 | Lecture5 | Slide22 | `88a39d1a41823f64` | `fabe95a86106910f` |
| 6 | Lecture6 | Slide22 | `2409a855f9178e8b` | `1a3edca964036c00` |
| 7 | Lecture6 | Slide32 | `a202265b35bc37ef` | `382f85f8d84bd57e` |
| 8 | Lecture7 | Slide10 | `af860ee13cd4bcbc` | `c4edcbe01e795718` |
| 9 | Lecture7 | Slide30 | `816795d435cd5f8b` | `26c659ccd8bf2fd9` |
| 10 | Lecture8 | Slide13 | `6714bfc2aed682c4` | `420d82d03f33243b` |
| 11 | Lecture8 | Slide23 | `bc451d01fd30a3ba` | `638b7b02f3645b7e` |
| 12 | Lecture9 | Slide20 | `649a079f0de1211b` | `e818f68d293d1513` |
| 13 | Lecture10 | Slide21 | `0884fd6e80bf78d8` | `b1cbded26bb77199` |
| 14 | Lecture10 | Slide41 | `5cff915e1706f457` | `2101ea6803cd9acc` |
| 15 | Lecture12 | Slide37 | `413969d9c6762c16` | `6411cd9b358e58e2` |
| 16 | Lecture12 | Slide47 | `93eaf293b0d2d7da` | `90bd923b93e10161` |
| 17 | Lecture13 | Slide7 | `109303310aa78363` | `410d806d5b1c19f3` |
| 18 | Lecture13 | Slide27 | `a8e1ebb70b597124` | `50cfb38a753537b4` |
| 19 | Lecture14 | Slide17 | `53764ae1cf6c01a4` | `a8eac956b02fd828` |
| 20 | Lecture14 | Slide27 | `3481294385838fec` | `4a37501853e9d555` |
| 21 | Lecture16 | Slide7 | `29f0648875037a54` | `10fb63256db0760a` |
| 22 | Lecture17 | Slide2 | `7b9c81f1dc42c58b` | `7f3ae1df7d5bbc7f` |
| 23 | Lecture17 | Slide22 | `2ca56cc30740f698` | `4b7a60b25cff14df` |
| 24 | Lecture17 | Slide32 | `0787b0dd6e38ee13` | `03cc40ecb54d922d` |
| 25 | Lecture18 | Slide27 | `8501022964052e9e` | `0aca32bc0439644e` |
| 26 | Lecture18 | Slide37 | `6f734cea45f65ee9` | `b87cf06091e57ea8` |
| 27 | Lecture19 | Slide42 | `d60cb7b1112f058a` | `a794392e5e9e5b22` |
| 28 | Lecture21 | Slide27 | `c4ef02c38a3b9330` | `aa71a6651549d2da` |
| 29 | Lecture22 | Slide4 | `607c1c26a98f8d3e` | `e3b0c44298fc1c14` |
| 30 | Lecture22 | Slide14 | `41a171681a01bbfd` | `60561685f59a697e` |

## 3. Model outputs annotated

Final task-specific protocol (D-019): concepts `max_new_tokens=512`, relations `=1024`.

| model | revision |
|---|---|
| `Gemma-4-31B` | `145dc2508c480a64b47242f160d286cff94a2343` |
| `InternVL3-14B` | `419aa10d2db7da6c64382ad3124f79a60cec42aa` |
| `Qwen3-VL-32B-Instruct` | `0cfaf48183f594c314753d30a4c4974bc75f3ccb` |

Source files: `results/final_task_specific/merged/*.jsonl` (6,372 records per model).
Hashes in `HUMAN_EVAL_HASHES.txt`.

## 4. Design

### H1 — semantic panels
**180 panels per annotator** = 30 slides x 3 models x 2 tasks. Each panel shows one slide,
its transcript, and **all three source sets simultaneously**, so the annotator assigns
semantic-cluster IDs across their union. Source-pair panels were deliberately **not** used:
clustering across all three at once is what makes the three pairwise comparisons mutually
consistent.

Cluster IDs are **local to the panel**. Items with no equivalent get their own ID. Items
within the same source may share an ID if genuinely duplicative. Partial entailment is
**not** equivalence, and the instructions state this explicitly with examples.

**No cluster is pre-filled.** No automated semantic judge, embedding model, lexical matcher,
or prior automated analysis was used to seed, suggest, or bias any grouping.

### H2 — evidence attribution
**360 items per annotator** = 3 models x 2 tasks x 60, sampled **only** from valid final MM
outputs on the same 30 slides, with fixed seed **20260831**, drawn **before any human label
existed**. Sampling is round-robin across slides so high-yield slides cannot dominate.
Eligible pools were 222–341 items per cell, so every cell reached the full 60.

Selection used **none** of: prior automated labels or scores, item difficulty, lexical
content, desired category balance, model agreement, or any source result.

Categories: `both`, `transcript_only`, `image_only`, `neither`, `uncertain`. For relations,
support applies to the **full assertion**, not merely the presence of subject/object terms.

## 5. Blinding

- Models appear only as `M1`/`M2`/`M3`; sources only as `A`/`B`/`C`.
- **Different permutations for each annotator**, from frozen seeds (A: 20260901, B: 20260902).
- Item presentation order randomised within each source set.
- Item IDs are HMAC-derived and encode nothing about model or source.
- Deblinding maps live **only** in `sealed_mapping/`, never inside annotator folders.
- Annotator-facing files were scanned for model-name leakage: **0 hits** in HTML, panel and
  evidence data.

**Shown to annotators:** slide image, transcript, anonymised item sets, task type.
**Hidden:** model identity, source identity, exact structural Jaccard, prior automated
semantic scores, prior evidence labels, study hypotheses, and all perturbation results.

## 6. Independence

Packages are fully independent. No first-pass file contains the other annotator's answers,
adjudicated labels, prior automated labels, or expected answers. Annotators are instructed not
to communicate during the first pass, not to search the Internet, and to use `uncertain`
rather than guess.

## 7. Scoring (scripts written, NOT executed on real labels)

| script | purpose |
|---|---|
| `code/validate_annotations.py` | completeness, valid IDs/labels, package identity, mapping integrity |
| `code/score_semantic_agreement.py` | human-semantic Jaccard; inter-annotator equivalence agreement |
| `code/score_evidence_agreement.py` | category agreement, kappa, confusion matrix, uncertain rate |
| `code/create_disagreement_packet.py` | disagreement-only adjudication packets |
| `code/score_adjudicated_results.py` | adjudicated proportions; exact-vs-human comparison |

**Semantic metric policy** (mirrors the automated study): a cluster is present in a source if
at least one item from that source belongs to it; set Jaccard over cluster IDs; **co-empty
comparisons are excluded, never scored 1.0**; invalid sources are excluded, never treated as
empty.

**Inter-annotator reporting:** raw equivalence agreement, Cohen's kappa, positive-class
precision/recall/F1, and the number of positive equivalence decisions from each annotator.
Kappa is **never** reported alone, because equivalent pairs may be sparse.

**Uncertainty:** 10,000 bootstrap replicates; lecture-cluster bootstrap using frozen lecture
membership for semantic source-pair agreement. The 30-slide cohort spans
18 lectures; where a clustered estimate is unstable the
scripts emit an explicit `stability_warning` **rather than silently switching method**.
Evidence-category proportions use Wilson 95% intervals.

## 8. Adjudication — NOT performed

Adjudication happens only after both completed first passes exist. Packets contain
**disagreement cases only**, do not reveal model or source identity, and provide empty
consensus fields. **Nothing is resolved automatically.**

## 9. Comparison with exact structural metrics

After adjudication, per model x task x source pair: exact structural Jaccard, human-semantic
Jaccard, absolute difference, relative difference where mathematically meaningful, and CIs.

**Two reporting rules are enforced in the scripts and must hold in the paper:**
1. `1 - Jaccard` is **never** described as a 'semantic error rate'.
2. A low exact structural Jaccard is **never** equated with semantic disagreement.

Relative difference is undefined where the exact structural Jaccard is 0 and is reported as
such rather than as a large or infinite ratio.

No automated semantic or evidence labels were used as ground truth or shown during
annotation.
