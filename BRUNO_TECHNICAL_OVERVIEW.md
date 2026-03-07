# Questions Agent — Full Technical Overview

**For:** Bruno Balen (Founder)
**Date:** 2026-02-27
**State:** 721 tests passing, 16,454 lines pipeline + 5,642 lines production

---

## What We Actually Built

The Questions Agent is a complete adaptive psychometric engine. Here's the full stack, layer by layer.

---

## Layer 1: The Science Core

### 1.1 IRT Scoring Engine (`irt.py` — 442 lines)

The Graded Response Model (Samejima, 1969). For every answer a person gives:
- Computes latent trait theta (where they sit on the construct)
- Computes SE(theta) — how confident we are
- Computes Fisher information — how much that answer taught us
- Uses Expected A Posteriori (EAP) estimation with N(0,1) prior and 61-point quadrature

This is the mathematical foundation. Everything else builds on this.

### 1.2 Cross-Instrument Multiplexing (`selection.py` — 880 lines)

**This is the core invention.** When a person answers one question about physical activity:
- FINDRISC gets Fisher information for diabetes risk
- IPAQ-SF gets information for activity level
- Lee-NAFLD gets information for liver risk
- EZ-CVD gets information for cardiovascular risk

The selection engine computes:
```
Score(item) = sum_over_scales( I_item(theta_scale) * w_scale )
```
where w_scale = SE(theta_scale) — invest more measurement in under-measured constructs. Then picks top-k subject to constraints (cooldown, intrusiveness, sensitivity, diversity).

No existing CAT system does this. They all optimise one instrument at a time.

### 1.3 Scoring Engine (`scoring.py` — 837 lines)

Six cardiometabolic instruments fully implemented:
- **FINDRISC** (0-26 scale, T2D 10-year risk) — Lindstrom & Tuomilehto 2003
- **EZ-CVD** (non-lab cardiovascular risk) — Gaziano et al. 2008
- **IPAQ-SF** (physical activity categorical) — Craig et al. 2003
- **Lee NAFLD** (liver risk 0-12) — Lee et al. 2018
- **SCORED** (CKD screening 0-14) — Bansal et al. 2007
- **AUDIT-C** (alcohol use 0-12) — Bush et al. 1998

Three vitality instruments:
- **WHO-5** (wellbeing 0-100) — WHO 1998
- **SF-36 Vitality** (4-item subscale 0-100) — Ware et al. 1993
- **PROMIS Fatigue-7a** (T-score) — Cella et al. 2010

Each produces: raw_score, normalized_score (0-1), risk_tier, confidence, IRT theta+SE+reliability+information. Plus the behavioural metadata adjustment (se_theta_adjusted, uncertainty_multiplier, engagement_quality).

### 1.4 Cardiometabolic Risk Index (`cardio_risk_index.py` — 229 lines)

Composite weighted score across all 6 cardio instruments:
```
FINDRISC 0.25 | EZ-CVD 0.20 | IPAQ-SF 0.20 | Lee-NAFLD 0.15 | SCORED 0.10 | AUDIT-C 0.10
```
IPAQ-SF is inverted (more activity = lower risk). Weights renormalised over available instruments so partial coverage is valid.

---

## Layer 2: The Longitudinal Intelligence

### 2.1 N-of-1 Calibration (`n_of_1.py` — 579 lines)

**This is where it gets personal.** Each (user, scale) pair has a PersonalCalibration state:

- **Kalman filter** tracks theta over time with process noise:
  - predict: sigma2_pred = sigma2_post + sigma2_process * delta_t
  - update: K = sigma2_pred / (sigma2_pred + SE_obs^2)
  - Result: person-specific posterior that adapts to their unique trajectory

- **Calibration phases**: warming (<10 obs) -> calibrating (10-29) -> calibrated (30+)

- **Reliable Change Index** (Jacobson-Truax): RCI = delta_theta / sqrt(SE_personal^2 + SE_baseline^2). When |RCI| > 1.96, we can say with 95% confidence the person genuinely changed.

- **Measurement sufficiency**: Can we detect a 0.5 SD change? If not, keep measuring.

This transforms population psychometrics into N-of-1 precision medicine.

### 2.2 Active Coherence Detection (`coherence.py` — 604 lines)

6 signals detect careless/inattentive responding:
1. **Speed index** — % items below 2s (Huang et al. 2012)
2. **Longstring** — longest run of identical answers (Johnson 2005)
3. **RT variability** — low CV implies mechanical responding
4. **Intra-scale variance** — near-zero with multiple options = straightlining
5. **Fatigue slope** — negative log-RT trend = progressive disengagement
6. **Skip rate** — proportion skipped

Multiple-hurdle approach (DeSimone 2015): >= 2 flags = suspect, >= 4 = invalid. Composite coherence_score = 1.0 - sum(severity * weight). Tiers: valid/suspect/invalid.

### 2.3 Behavioural Uncertainty Modifiers (`response_metadata.py` — 424 lines)

Per-response metadata captured:
- response_latency_ms, edit_count, was_skipped, was_declined
- channel (tap/voice/keyboard/swipe), voice_hesitation_ms
- time_of_day_hour (circadian context)

Uncertainty modifier model: SE_adjusted = SE_irt * multiplier
- Fast + no edits -> multiplier < 1.0 (more confident)
- Slow + edits -> multiplier > 1.0 (less confident)
- Skip -> multiplier >> 1.0

Session-level profile aggregates into engagement_quality: focused/normal/distracted/fatigued. This feeds back into next session's item selection.

### 2.4 Drift-Triggered Anamnesis (`anamnesis.py` — 427 lines)

When drift is detected:
1. Evaluates triggers (EWS > 0.6, radius > 2sigma from baseline, high velocity, cross-modal disagreement)
2. Maps drift domain to instruments via routing table (e.g., metabolic drift -> FINDRISC probe)
3. Creates anamnesis episode, enqueues targeted follow-up items
4. Tracks: active -> confirmed/denied -> resolved/expired

This makes the system a proactive diagnostic probe, not a passive questionnaire.

---

## Layer 3: The Geometry

### 3.1 9D Latent State (`state_snapshots.py` — 379 lines)

Nine health dimensions:
```
energy_vitality | sleep_quality | gut_gi | glycemic_risk | cardiovascular_load
mood_affect | cognitive_control | agency_purpose | social_connectedness
```

Each dimension gets x_hat (estimate) and x_uncertainty (confidence). Updated daily from scale scores via tag-to-dimension mapping (e.g., "metabolic" -> glycemic_risk + cardiovascular_load).

### 3.2 Coherence Circle (`questions_fold.py` — 636 lines)

The 9D state now uses a two-level geometry:
- z = (z1, z2) — current position
- z* = (z1*, z2*) — current feasible optimum (best state the person can stably reach now)
- z† = (z1†, z2†) — Supercoherence reference (ideal coherent biology)
- r = |z - z*| — acute dysregulation radius
- r_struct = |z* - z†| — structural aging gap
- r_abs = |z - z†| — absolute rejuvenation gap
- theta = angle(z - z*) — which direction current dysregulation is drifting
- local_coherence and absolute_coherence are computed separately so symptom control is not confused with rejuvenation

4 MiniFold sub-circles: metabolic, mind, body, social — each covering a subset of the 9 dimensions.

### 3.3 Early Warning System (`trajectory_metrics.py` — 340 lines)

Critical slowing down detection (from dynamical systems theory):
- var(r) — increased variance signals approaching regime transition
- AC1(r) — increased lag-1 autocorrelation signals critical slowing
- trend_speed — systematic drift
- recovery_rate — how fast r returns after perturbation
- Composite EWS score [0, 1]

When EWS triggers -> drift event -> anamnesis probe.

---

## Layer 4: The QuestionsFold Output

### Identity Mask Fragment (`questions_fold.py`)

This is what the Questions Agent exports — either for standalone use or for AniFold:

```python
m_questions(t) = {
    mu_t, sigma_t,                    # 9D latent posterior
    z_t, z*_t, z†_t,                  # current position, feasible reference, Supercoherence
    r_t, s_t, q_t, theta_t,           # acute, structural, absolute distances + angle
    kappa_local_t, kappa_abs_t,       # regulation score vs rejuvenation score
    sigma_kappa_t,                    # coherence uncertainty
    c_t,                              # coverage vector
    h_t,                              # history envelope (velocity, accel, EWS)
    minifolds,                        # 4 sub-circles
}
```

For AniFold fusion: `fold.for_anifold_fusion()` extracts mu + precision (1/sigma^2) for the cross-attention layer.

---

## Layer 5: The Intelligence

### 5.1 Progressive Measurement Plan (`measurement_packages.py` — 669 lines)

Sequences instrument packages based on clinical priority:
1. Cardiometabolic Core (FINDRISC, EZ-CVD)
2. Activity & Lifestyle (IPAQ-SF, AUDIT-C)
3. Vitality Battery (WHO-5, SF-36, PROMIS-Fatigue)
4. Extended Risk (Lee-NAFLD, SCORED)

Adapts based on decoherence mode — if metabolic drift detected, reprioritise metabolic instruments.

### 5.2 Domain Governance (`governance.py` — 162 lines)

Controls which health domains are active per user:
- Onboarding lock: cardiometabolic first, then unlock others
- Domain promotion: automatic after sufficient data
- Cardiometabolic gate: must complete core instruments before branching

### 5.3 Safety System

Automatic flagging when:
- Answer triggers safety keywords (suicidal ideation, self-harm)
- Policy-live decisions violate constraints
- Cross-modal context flags risk

Clinician review workflow: open -> resolved, with audit trail.

### 5.4 Contextual Bandit Policy Layer (`policy/` — 11 files)

An optional reranking layer over the deterministic selection:
- UCB-based contextual bandit (Thompson sampling variant)
- Features: user history, time context, domain state, engagement quality
- Constraints: mandatory items, blocked items, k-core budget
- Rollout: deterministic -> policy_shadow (logging only) -> policy_live
- Auto-rollback guard: monitors completion rate, burden, safety violations
- Full audit trail: every decision has counterfactuals + propensities for offline evaluation

### 5.5 Concordance Framework (`concordance.py` — 389 lines)

Built for the validation study:
- ICC(3,1) with 95% CI and F-test (threshold: 0.85)
- Bland-Altman: mean difference, SD, limits of agreement
- Cohen's kappa for risk tier classification (threshold: 0.80)
- Per-item agreement analysis

---

## Layer 6: Production

### 6.1 FastAPI Application (`prod/app.py` — 1,049 lines)

40+ endpoints. Full CRUD for users, sessions, answers. Admin endpoints for registry, policy, SLO, audit, privacy retention. All behind API key auth.

### 6.2 PostgreSQL + Alembic (`prod/models.py`, `prod/alembic/`)

23 SQLAlchemy tables, 4 Alembic migrations. Schema auto-creates on startup.

### 6.3 Docker (`prod/Dockerfile`, `prod/docker-compose.yml`)

Single-stage build on python:3.11-slim. Postgres 15 for data. Ready for deployment.

### 6.4 Operational Tools (`tools/` — 17 scripts)

Load testing, backup/restore, policy audit, clinical audit export, PHI retention, SLO alerting, registry validation, offline evaluation.

---

## What's Cool / Novel / Patentable

| Innovation | Why it's novel | Where |
|------------|---------------|-------|
| Cross-instrument multiplexing | One question feeds N instruments simultaneously | `selection.py` |
| Behavioural uncertainty modulation | HOW you answer modulates confidence | `response_metadata.py` |
| Drift-triggered anamnesis | System probes you when it detects change | `anamnesis.py` |
| N-of-1 Kalman calibration | Population -> person-specific over time | `n_of_1.py` |
| Coherence Circle geometry | 9D health state as a 2D circle | `questions_fold.py` |
| EWS critical slowing | Predict health transitions before they happen | `trajectory_metrics.py` |
| Active coherence detection | Real-time careless responding detection | `coherence.py` |
| Progressive measurement plan | Adapts instrument sequence to decoherence mode | `measurement_packages.py` |
| Engagement feedback loop | Previous session quality affects next selection | `selection.py` + `response_metadata.py` |

---

## Architecture Diagram (Simplified)

```
USER answers 5 Qs/day
        |
        v
  [Registry]  ->  [Selection Engine]  ->  Daily Session (5 items)
                    (Fisher info x               |
                     multiplexing)               v
                                          [Answer Processing]
                                            |          |
                                    Behavioural    Scale Evidence
                                    Metadata       (multiplexed)
                                            |          |
                                            v          v
                                    [SE Adjustment] [IRT Scoring]
                                            |          |
                                            v          v
                                    [9D State Update] [Coherence Detection]
                                            |                  |
                                            v                  v
                                    [Circle Geometry]  [Quality Gating]
                                            |
                                     +------+------+
                                     |             |
                                     v             v
                                [EWS/Drift]  [QuestionsFold]
                                     |             |
                                     v             v
                              [Anamnesis]   [Identity Mask] --> AniFold
                                     |
                                     v
                              [Follow-up Queue] --> Next session
```

---

## What AniFold Cross-Validation Looks Like

When AniFold detects (from omics, voice, imaging) that someone's ApoB or metabolic markers are drifting:

1. AniFold sends context to Questions Agent via `/v1/users/{uid}/daily-questions/select`
2. Context includes `safety_trigger_active: true` or drift domain hints
3. Questions Agent enters safety mode / anamnesis mode
4. Selects targeted cardiometabolic instruments (via drift routing table)
5. Probes the person with relevant questions
6. Reports back via `/v1/users/{uid}/projection/questions` with fresh evidence
7. AniFold incorporates fresh questionnaire evidence into its next Z estimate

The loop closes: omics detect biochemical change -> Questions Agent validates subjectively -> AniFold has both objective + subjective evidence.

---

## Numbers

- **16,454 lines** pipeline code
- **5,642 lines** production code
- **721 tests** all passing
- **9** health dimensions
- **9** validated instruments (6 cardio + 3 vitality)
- **5** patent claim families
- **23** database tables
- **40+** API endpoints
- **6** coherence detection signals
- **4** MiniFold sub-circles
- **3** deployment configs (wellness/CDS/SaMD)
- **4** Alembic migrations

---

## What's Next

1. **DEFERRED: Multi-ontology** (ICD-10/SNOMED-CT/LOINC mapping — remind me)
2. **Wire N-of-1 + Coherence + Cardio Risk into production API**
3. **Web app demo** — live visualization of the Coherence Circle
4. **Retrospective data** — calculate first real scores from actual user responses
5. **Docker hardening** — multi-stage build, health checks, non-root user
6. **Structured logging** — throughout service layer
