# Questions Agent Platform v17

> Part of **ANI BIOME** — the behavioral signal encoder of the organism.
> For the full picture, read [`ani-vault/00_BOOTSTRAP.md`](https://github.com/anibiome/ani-vault/blob/main/00_BOOTSTRAP.md).

## What This Repo Does

IRT-driven adaptive questionnaire engine that encodes behavioral, psychological, and lifestyle signals into structured latent features. 941 items across 65 validated clinical instruments, selected adaptively using Item Response Theory.

**47,760 lines Python. 781 tests. 52+ FastAPI endpoints. 5 patent claim families.**

**Input**: User responses to adaptively selected questions
**Output**: Behavioral latent features → feeds into Identity Mask and Coherence Circle

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
  → UniFold (behavioral signal fused with omics)
  → Coherence Circle (behavioral coherence component)
  → ani-scan (QA is final step of video scan flow)
```

## Connections

| Repo | How it connects |
|------|----------------|
| `ani-vault` | Source of truth: coherence definition, Identity Mask spec |
| `ani-scan` | Questions run as final step of the video scan experience |
| `ani-medical-platform` | Shared statistical infrastructure (IRT, Bayesian) |
| `ani-proteomics` / `ani-metabolomics` | Behavioral signals correlate with molecular state — cross-modal coherence |

## For AI Agents

- IRT adaptive selection minimizes question burden while maximizing information
- 65 instruments cover: sleep, stress, diet, exercise, mood, cognition, pain, social, environmental
- Latent features are non-reconstructive (can't recover individual answers)
- Patent claims cover: adaptive selection, coherence scoring, cross-modal fusion, trajectory analysis, early warning
