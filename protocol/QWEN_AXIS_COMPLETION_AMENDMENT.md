# Amendment Q1: Qwen perturbation-axis completion

Recorded before any Qwen prompt, stochastic, or deterministic-repeat extension inference in this phase.
Recorded UTC: 2026-09-04T14:32:11Z

## Scientific motivation

Complete the previously ragged perturbation design for Qwen3-VL-32B-Instruct after manuscript audit identified that prompt-versus-stochastic comparisons were based on only two models. This amendment is design-completion, not result-contingent tuning.

## Frozen scope

- Model: Qwen/Qwen3-VL-32B-Instruct
- Revision: 0cfaf48183f594c314753d30a4c4974bc75f3ccb
- Cohort: frozen 1,062-slide manifest
- Evidence mode: MM only
- Tasks: concepts and relations/triples, analyzed separately
- P0: frozen v2 prompt
- P1 and P2: already-frozen paraphrases
- Concepts max_new_tokens: 512
- Relations max_new_tokens: 1024
- Prompt P1/P2 decoding: deterministic greedy
- Deterministic-repeat control: P0 greedy, seed 20260815
- Stochastic seeds: 20260821, 20260822, 20260823
- Stochastic temperature: 0.7
- Stochastic top_p: 1.0
- Per-record seed derivation: unchanged frozen V3 implementation
- Parser/schema/normalization: unchanged
- Technical gate: unchanged
- Aggregate technical validity threshold: at least 95 percent
- Per-cell technical validity threshold: at least 90 percent
- Generation errors allowed: zero
- Degenerate repetition maximum: 1 percent
- Scientific overlap metric and lecture-cluster bootstrap remain unchanged
- Scientific overlap results will not be inspected until all new conditions finish and pass the technical gate

## Outcome-independent commitment

All completed conditions and all outcomes will be retained and reported.

No prompt, seed, temperature, top_p, token budget, parser, gate threshold, model revision, dataset, or evaluation rule may be changed after observing these outputs.

No additional model, dataset, or human annotation is authorized by this amendment.
