# Final strengthening analysis freeze

This analysis plan is recorded after the Qwen extension passed its
predefined technical gate and before scientific overlap results from
the extension are inspected.

## A. Qwen perturbation completion

Prompt sensitivity:
P0-P1, P0-P2, P1-P2.

Stochastic sensitivity:
S1-S2, S1-S3, S2-S3 are the primary stochastic perturbation
comparisons.

P0-S1, P0-S2, P0-S3 are retained as secondary deterministic-to-sampled
comparisons.

Concepts use the frozen 512-token protocol.
Relations use the frozen 1024-token protocol.

The metric remains validity-aware union-nonempty exact structural
Jaccard with the existing normalization and parser.

Uncertainty remains a 10,000-resample lecture-cluster bootstrap.

## B. Output-yield analysis

For every model x task x evidence-source cell, report normalized unique
item-set size among technically valid records:

N valid
mean
median
25th percentile
75th percentile
zero-output rate

The normalized unique set is used because it is the actual set entering
the Jaccard computation.

## C. Directed containment

For source pair A,B:

A contained in B = size(intersection(A,B)) / size(A)

B contained in A = size(intersection(A,B)) / size(B)

A direction is scored only when both model outputs are technically valid
and the denominator set for that direction is nonempty.

Confidence intervals use the same 10,000-resample lecture-cluster
bootstrap.

Containment is a decomposition/sensitivity analysis and does not replace
the frozen primary Jaccard metric.

## D. Source-versus-model paired bootstrap

This analysis corrects the previous practice of judging the difference
between perturbation axes by overlap of separate confidence intervals.

For each model-task cell, compare the exact two perturbations that
defined the strongest source and strongest model-choice candidates in
the previously frozen analysis.

The source comparison is MM versus IMG in all six cells.

The frozen strongest model-choice comparator is:

InternVL concepts: InternVL versus Qwen at MM
InternVL relations: InternVL versus Qwen at MM
Gemma concepts: Gemma versus Qwen at MM
Gemma relations: Gemma versus Qwen at MM
Qwen concepts: Qwen versus InternVL at MM
Qwen relations: Qwen versus Gemma at MM

The same sampled lecture clusters are used for the source and model
statistics within every bootstrap replicate.

Define:

delta = source agreement - model-choice agreement

Because lower agreement means a larger perturbation:

delta below zero means source substitution is more disruptive.
delta above zero means model substitution is more disruptive.
a 95 percent interval crossing zero is unresolved.

A common-slide paired difference is additionally reported as a
sensitivity analysis.

No threshold is changed after seeing results.

## E. Human transitive-closure sensitivity

The published human semantic calculation uses connected components over
direct equivalence judgments.

To quantify the effect of transitive closure, recompute a conservative
direct-equivalence score for each source pair using maximum-cardinality
one-to-one matching over only directly endorsed equivalence edges.

For a source pair with set sizes A and B and M directly matched items:

direct semantic Jaccard = M / (A + B - M)

This score does not infer equivalence through a chain.

Both returned adjudication variants r1 and r2 are analyzed.

Report:
current connected-component score,
direct-only matching score,
difference caused by closure,
semantic uplift over exact structural Jaccard with and without closure,
and preservation of the source ordering.

## F. Outcome-independent reporting

All Qwen outcomes are retained.

No prompt, seed, model, generation parameter, parser, gate, threshold,
bootstrap count, data subset, or human label is changed after scientific
unblinding.

No further inference is authorized by this analysis plan unless an
implementation error is demonstrated.
