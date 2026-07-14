# Questions Agent Platform v17

## 2026-04-05 Ani Alignment

This repo provides one contextual modality for Ani, not Ani's identity.
Questionnaire signals should enrich state, response tracking, and context; they should
not force ontology-heavy or scan-heavy public explanations by default.

> Part of **ANI BIOME** — the behavioral signal encoder of the organism.
> For the full picture, read [`ani-vault/00_BOOTSTRAP.md`](https://github.com/anibiome/ani-vault/blob/main/00_BOOTSTRAP.md).

## What This Repo Does

IRT-driven adaptive questionnaire engine that encodes behavioral, psychological, and lifestyle signals into structured latent features. 941 items across 65 validated clinical instruments, selected adaptively using Item Response Theory.

**47,760 lines Python. 781 tests. 52+ FastAPI endpoints. 5 patent claim families.**

**Input**: User responses to adaptively selected questions
**Output**: Behavioral latent features → feed into shared state, memory, and response tracking

## Where It Fits

```
User answers adaptive questions (IRT-driven selection)
       ↓
  [questions-agent-platform]  ← YOU ARE HERE
       ↓
  65 clinical instruments → behavioral latent encoding
       ↓
  Identity Mask: h_t (memory/trajectory slot)
       ↓
  → UniFold / shared latent-state packet
  → optional trajectory and advanced geometry views when the runtime uses them
  → ani-scan or other downstream surfaces when that path is active
```

## Connections

| Repo | How it connects |
|------|----------------|
| `ani-vault` | Source of truth: coherence definition, Identity Mask spec |
| `ani-scan` | Questions may compose with the scan flow, but are not a mandatory final step in every runtime |
| `ani-medical-platform` | Shared statistical infrastructure (IRT, Bayesian) |
| `ani-proteomics` / `ani-metabolomics` | Behavioral signals correlate with molecular state — cross-modal coherence |

## For AI Agents

- IRT adaptive selection minimizes question burden while maximizing information
- 65 instruments cover: sleep, stress, diet, exercise, mood, cognition, pain, social, environmental
- Latent features are non-reconstructive (can't recover individual answers)
- Patent claims cover: adaptive selection, coherence scoring, cross-modal fusion, trajectory analysis, early warning

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
