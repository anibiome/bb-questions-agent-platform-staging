# Questions Agent Platform v17

## 2026-04-05 Ani Alignment

This repo provides one contextual modality for Ani, not Ani's identity.
Questionnaire signals should enrich state, response tracking, and context; they should
not force ontology-heavy or scan-heavy public explanations by default.

> Part of **ANI BIOME** — the behavioral signal encoder of the organism.
> For the full picture, read [`ani-vault/00_BOOTSTRAP.md`](https://github.com/anibiome/ani-vault/blob/main/00_BOOTSTRAP.md).

## What This Repo Does

IRT-oriented adaptive questionnaire engine that turns authorized behavioral responses into derived evidence packets. Instrument, scoring, and scale claims remain registry-, license-, provenance-, calibration-, and release-review bound.

**Internal runtime surface: Python reference engine, FastAPI-compatible service layer, tests, and patent-intake ideas remain gated by current source, permissions, validation, and release review.**

**Input**: User responses to adaptively selected questions
**Output**: Behavioral latent features → feed into shared state, memory, and response tracking

## Where It Fits

```
User answers adaptive questions (IRT-driven selection)
       ↓
  [questions-agent-platform]  ← YOU ARE HERE
       ↓
  registry-backed instruments → behavioral latent encoding
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
| `ani-proteomics` / `ani-metabolomics` | Behavioral signals can be calibrated against molecular-state observers after paired validation |

## For AI Agents

- IRT adaptive selection minimizes question burden while maximizing information
- Registry-backed instruments cover: sleep, stress, diet, exercise, mood, cognition, pain, social, environmental
- Latent features are non-reconstructive (can't recover individual answers)
- Patent-intake candidates require source-linked review; do not treat repo text as patent-ready proof
