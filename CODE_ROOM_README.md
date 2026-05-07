# CODE_ROOM_README - bb-questions-agent-platform

## Role In ANI.AI

Question and response intelligence platform.

This repo should be read as part of ANI.AI's biological state-space
system, not as a standalone product. It belongs to route `product_runtime_conversation`.

## Why It Matters

ANI.AI reconstructs biological state from sparse, messy observations.
This repo contributes to that engine by owning one or more of: capture,
observer inference, omics linkage, state reconstruction, intervention
response, runtime answer generation, clinical/trial execution, or
evidence hygiene.

## Current Audit Posture

- Overall soundness score: `67`
- Architecture score: `88`
- Soundness verdict: `credible_needs_diligence_polish`
- Right-sizing verdict: `acceptable_with_polish`
- Strengths: test surface; Python syntax probes clean; typed contracts/schema logic; input validation/error gates; statistical correction/control language; provenance/artifact manifest language; clinical/trial context; build/dependency manifest; CI surface; README/front door; container/runtime packaging; explicit executable entrypoints; state-space/coherence implementation signal
- Gates: review eval/exec usage; tighten broad exception handling; quarantine proxy/synthetic language from measured-claim paths

## Reviewer Path

1. Read the root README and manifest/build files.
2. Find the executable entrypoints listed in the soundness audit.
3. Run only lightweight smoke/contract tests unless a VM-backed run card
   explicitly authorizes heavier compute.
4. Map the repo's outputs back to `ObservationBundle`, `PersonState`,
   observer agreement, omics bridge, intervention operator, or product
   runtime.
5. Keep evidence classes explicit: measured, derived, model-inferred,
   exploratory, proxy/bridge, or archival.

## Claim Boundary

This repo supports code-room technical diligence. It does not by itself
convert exploratory, proxy, synthetic, or model-inferred outputs into
public clinical claims. Claims must be tied to measured source artifacts,
validation traces, model checkpoints, or repo-backed reports.
