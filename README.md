# Configuration Sensitivity and Output-Budget Censoring in Structured Multimodal Extraction

Code and protocol release for a controlled audit of structured multimodal extraction under changes in evidence source, model choice, prompt wording, stochastic decoding, deterministic reruns, and generation budget.

The repository accompanies an anonymous manuscript under review.

## Study scope

The frozen study cohort contains 1,062 paired educational slide images and aligned lecture transcripts from 22 medical-imaging lectures.

Three evidence conditions are evaluated:

- `MM`: slide image + aligned transcript
- `TX`: transcript only
- `IMG`: slide image only

Two structured extraction tasks are evaluated separately:

- concept extraction
- relation extraction as subject-predicate-object triples

Concept and relation results are never pooled.

## Final model panel

The final panel contains three vision-language models:

- `OpenGVLab/InternVL3-14B`
  - revision `419aa10d2db7da6c64382ad3124f79a60cec42aa`

- `google/gemma-4-31b-it`
  - revision `145dc2508c480a64b47242f160d286cff94a2343`

- `Qwen/Qwen3-VL-32B-Instruct`
  - revision `0cfaf48183f594c314753d30a4c4974bc75f3ccb`

Evidence-source and model-substitution analyses cover all three models.

Prompt-paraphrase, stochastic-decoding, and deterministic-rerun analyses also include all three models in the final study. Qwen completion conditions were added under a frozen amendment before their scientific results were inspected.

## Final generation protocol

Concept extraction uses:

- `max_new_tokens = 512`

Relation extraction uses:

- `max_new_tokens = 1024`

The larger relation budget was introduced symmetrically after a frozen diagnostic showed that the earlier 512-token ceiling preferentially censored relation-rich generations.

Deterministic conditions use greedy decoding.

Stochastic conditions use:

- `temperature = 0.7`
- `top_p = 1.0`
- base seeds `20260821`, `20260822`, and `20260823`

The V3 runner derives a deterministic per-record seed from the run seed and record identity so stochastic runs are resumable without changing downstream samples.

## Primary structural metric

The primary agreement measure is validity-aware exact structural Jaccard.

A pair is scored only when:

1. both outputs pass the frozen technical-validity definition; and
2. the union of the normalized extracted item sets is non-empty.

Invalid pairs are excluded.

Valid co-empty pairs are not assigned an agreement score of one.

Uncertainty uses a lecture-level cluster bootstrap with 10,000 resamples over the 22 lectures.

## Final strengthening analyses

The final analysis adds three checks that are important for interpreting raw Jaccard.

### Output yield

For each model × task × evidence-source cell, the analysis reports normalized unique-item set size, including mean, median, interquartile range, and zero-output rate.

### Directed containment

For source conditions `A` and `B`, the analysis reports both:

`|A ∩ B| / |A|`

and:

`|A ∩ B| / |B|`

with lecture-cluster bootstrap intervals.

This separates set-size asymmetry from genuine content divergence.

### Paired source-versus-model bootstrap

The final paper does not infer perturbation ordering from overlap between independently computed confidence intervals.

Instead, source substitution and model substitution are compared within a paired lecture-cluster bootstrap using identical sampled lecture clusters in every bootstrap replicate.

The reported quantity is:

`delta = source agreement - model-substitution agreement`

A 95% interval crossing zero is treated as unresolved.

## Prompt and stochastic sensitivity

Three meaning-preserving prompt conditions are used:

- `P0`: original frozen prompt
- `P1`: paraphrase 1
- `P2`: paraphrase 2

Stochastic sensitivity uses three independently seeded sampled generations.

The final three-model analysis is intentionally descriptive about prompt-versus-stochastic ordering; no universal perturbation hierarchy is assumed.

## Human semantic evaluation

The human-evaluation directory contains the protocol and analysis software.

The study distinguishes exact structural disagreement from semantic equivalence.

The final strengthening analysis additionally recomputes semantic agreement using a conservative direct-edge-only maximum-cardinality matching procedure. This sensitivity analysis does not infer semantic equivalence transitively through connected components.

Completed annotations, adjudication returns, sealed mappings, source material, and derived human-evaluation result files are not redistributed.

## Repository structure

- `code/`
  - frozen extraction runners
  - V3 perturbation runner
  - structural metrics
  - technical-gate code
  - source, prompt, stochastic, and budget analyses
  - final strengthening analysis

- `configs/final/`
  - final source, prompt, stochastic, Qwen-repeat, Qwen-completion, and task-budget specifications

- `configs/diagnostics/`
  - technical-gate and relation-budget diagnostic specifications

- `prompts/`
  - original prompt and the two frozen meaning-preserving variants

- `protocol/`
  - frozen Qwen-axis completion amendment
  - frozen final strengthening analysis plan

- `human_eval/`
  - annotation protocol, instructions, and scoring software

- `provenance/`
  - hashes for the earlier exact execution release
  - sanitization record
  - pinned model revisions

- `environment/`
  - recorded software environment information

## Path sanitization

The original experimental environment used machine-specific absolute filesystem paths.

The public release removes those machine-specific path defaults so the repository can be used anonymously and portably.

This sanitization does not change prompts, schemas, parsing, normalization, metrics, generation parameters, model revisions, seeds, data selection, or statistical definitions.

The exact hashes of the earlier execution release are retained under `provenance/`.

For historical V2 artifacts referenced by some archival analysis scripts, place equivalent local artifacts under:

`external_artifacts/experiment_v2/`

or adapt those local paths in your private working copy.

## Data and result release policy

This repository intentionally does not redistribute:

- lecture images
- lecture transcripts
- private manifests containing source paths
- generated model outputs
- result JSON/JSONL files
- completed human annotations
- adjudication returns
- sealed reviewer mappings
- manuscript PDFs
- model weights
- model caches
- credentials

Therefore, some archival analysis commands require the corresponding locally held experimental artifacts.

## Reproducibility

For new inference, use:

    python code/v3_runner.py --help

For source-conditioned and cross-model agreement:

    python code/analyze_source_axis.py --help

For prompt or stochastic sensitivity:

    python code/analyze_prompt_sensitivity.py --help

For technical validity:

    python code/technical_gate.py --help

For the relation-budget diagnostic:

    python code/analyze_token_budget_sensitivity.py --help

The final strengthening analysis is:

    python code/final_strengthening_analysis.py

It consumes the frozen generated outputs and human-evaluation artifacts, which are intentionally not redistributed here.

## Release integrity

`RELEASE_FILE_HASHES.sha256` contains SHA-256 hashes for the current anonymous-safe release.

The earlier September execution-release hashes are retained separately under `provenance/`.

## License

Code is released under the MIT License.
