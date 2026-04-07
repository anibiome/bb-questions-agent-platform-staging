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
