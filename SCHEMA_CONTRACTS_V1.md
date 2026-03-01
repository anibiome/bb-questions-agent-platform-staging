# Questions Agent Contract Schemas (v1.0)

Date: 2026-02-06  
Canonical schema version: `1.0`  
Compatibility aliases accepted at ingress: `1`, `1.0`, `1.0.0`, `v1`, `v1.0`

## 1) Fusion -> Questions (selection context contract)

Contract name: `fusion_to_questions_context`  
Used in: `POST /v1/users/{user_id}/daily-questions/select`

Top-level request fields:
- `schema_version` (optional; defaults to `1.0`)
- `date` (optional, `YYYY-MM-DD`)
- `selection_mode` (`deterministic | policy_live | policy_shadow | policy_offline_replay`)
- `k_core` (optional int > 0)
- `allow_context_batches` (optional bool)
- `identity_mask_id` (optional string)
- `include_explanations` (optional bool)
- `context` (optional object)

`context` object:
- `contract_name` (optional, must equal `fusion_to_questions_context`)
- `schema_version` (optional; defaults to `1.0`)
- `anifold_z` (optional float array)
- `z_uncertainty_diag` (optional float array; if `anifold_z` exists, lengths must match)
- `z_velocity` (optional float array; if `anifold_z` exists, lengths must match)
- `z_distance_to_attractor` (optional float)
- `completion_rate_7d` (optional float)
- `completion_rate_14d` (optional float)
- `completion_rate_30d` (optional float)
- `burden_ms_median_14d` (optional float)
- `safety_trigger_active` (optional bool)
- `allow_context_batches` (optional bool)

Validation:
- Unknown fields are rejected (`extra=forbid`) for `daily-questions/select`, `context`, and outcome update payloads.
- `selection_mode` is normalized to lowercase and strictly validated.

## 2) Fusion -> Questions (outcome attachment contract)

Contract name: `fusion_to_questions_outcome`  
Used in: `POST /v1/users/{user_id}/policy/outcomes`

Fields:
- `contract_name` (optional, must equal `fusion_to_questions_outcome`)
- `schema_version` (optional; defaults to `1.0`)
- `source_event_id` (optional string)
- `decision_id` (required string)
- `z_before` (optional float array)
- `z_after` (optional float array)
- `uncertainty_before_diag` (optional float array)
- `uncertainty_after_diag` (optional float array)
- `reward_overrides` (optional object)

Vector guards:
- if `z_before` and `z_after` exist, lengths must match
- if both uncertainty vectors exist, lengths must match
- `uncertainty_before_diag` must match `z_before` length when both exist
- `uncertainty_after_diag` must match `z_after` length when both exist

## 3) Questions -> Fusion (evidence contract)

Contract name: `questions_to_fusion_evidence`  
Used in:
- `GET /v1/users/{user_id}/projection/questions` (prod)
- `GET /v1/users/{user_id}/projection/questions` (reference)

Payload requirements:
- `contract` object:
  - `name = questions_to_fusion_evidence`
  - `schema_version = 1.0`
  - `compatibility = ["1.0"]`
  - `producer` (pipeline/prod identifier)
  - `join_keys`:
    - `subject_id`
    - `date`
    - `session_id` (nullable)
    - `decision_id` (nullable)
- `subject_id`
- `timestamp`
- `modality_projections.questionnaires` with vectors:
  - `projection` length 128
  - `uncertainty.diag` length 128
  - `velocity` length 128
  - `attractor_candidate` length 128

Contract validation executes before payload persistence/return.

## 4) Join key chain

Canonical chain:
- selection request -> `daily_session.session_id`
- policy decision (optional) -> `policy_decision.decision_id`
- outcome attachment -> `policy_outcomes.decision_id`
- evidence payload -> `contract.join_keys` carries `session_id` + `decision_id` for fusion join

## 5) Backward compatibility behavior

- Legacy clients that omit `schema_version` continue to work (defaults to `1.0`).
- Version aliases map to canonical `1.0`.
- Unknown versions are rejected at validation.
- Unknown fields are rejected on contract-bound request models.
