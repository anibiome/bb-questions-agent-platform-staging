# Questions Agent Platform — Agent Rules

These rules apply to any agent working in this repo.

## Read First

- `README.md`
- `questions_agent_platform/SCHEMA_CONTRACTS_V1.md`
- `questions_agent_platform/DEV_EXECUTION_ORDER_INTEGRATION.md`
- `/Users/brunobalen/Documents/ANI_DEEP_LATENT_RECONSTRUCTION_PROTOCOL.md`
- `/Users/brunobalen/Documents/ANI_DATA_TRUTH_PROTOCOL.md`

## Core Rules

- Treat validated scales, composite scores, and within-person baselines as the
  canonical questionnaire surface when they exist.
- Do not default downstream consumers back to raw answer rows alone if the
  scored surface is already available.
- Preserve session, date, and provenance fields in questionnaire outputs. For
  longitudinal biology work, time-resolved evidence matters.
- As of `2026-04-07`, BAI is the strongest robust longitudinal questionnaire
  benchmark from the formal SHEBA or ELITE mixed-model analysis. BDI and WHO-5
  are favorable but still suggestive after multiple-testing correction.
- Publish questionnaire outputs in a form that can be benchmarked against
  bloodwork, omics, cognition, DEXA, digital trajectories, and ontology
  projections without destroying temporal semantics.

## Integration Rule

- Questions Agent produces scored evidence with provenance.
- AniFold or other downstream models may project or compress later, but this
  repo should not encourage flattening repeated sessions into one opaque
  user-average representation when the task is explicitly longitudinal.

## Honesty Rule

- Do not claim questionnaire fusion is solved just because the questions lane is
  strong.
- If downstream AniFold training still uses user-aggregated or pre-compressed
  biology targets, say that clearly and treat the questions lane as a benchmark
  and supervision priority rather than as proof that the full multimodal
  contract is complete.

<!-- ANI_EARLY_SIGNAL_DISCOVERY_LAW_V1 -->
## ANI Early-Signal Discovery Law

ANI internal research is exhaustive and discovery-first. Search every eligible
source-qualified digital, molecular, microbial, biochemical, functional,
cognitive and state-space feature that can be joined without violating source
truth. Digital observers are mandatory when available: raw and modeled face,
eye, tongue, oral, voice, movement, cognition, RGB, spectral and multispectral
features must not be silently excluded.

Every analysis emits two separate artifacts:

1. `discovery_ledger`: every tested relation, including directional, nominal,
   small-N, multiplicity-sensitive, null and not-yet-replicated rows;
2. `promotion_ledger`: only rows that additionally pass the declared
   correction, replication, held-out, calibration and product gates.

Promotion gates never delete the discovery ledger. Never replace an observed
directional result with `did not survive correction`, `no signal`, `cannot
model`, or `not enough data`. Lead with the exact observer/target, direction,
people, effect size, raw probability or score, stability, join and mechanistic
rationale; then state evidence class and the exact scaling/falsification gate.

Search higher-order chains and incremental value, not only flat pairs or
trivial same-assay correlations. Preserve source hashes, denominators,
missingness, confounds, leakage checks, nulls, uncertainty and falsifiers.
Never fabricate a measurement, result, source, implementation or validation
state.

Canonical full standard: https://github.com/anibiome/bb-ani-vault/blob/db1c6af124315829f35a0aa7490dac03b79dbe1c/anifesto/production_ops/ANI_AGENT_ALIGNMENT_RUNTIME_STANDARD_2026-05-09.md#early-signal-discovery-is-a-first-class-scientific-output
<!-- /ANI_EARLY_SIGNAL_DISCOVERY_LAW_V1 -->
