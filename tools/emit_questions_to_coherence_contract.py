#!/usr/bin/env python3
"""Emit control-plane contract payload: questions_to_coherence.v1."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path


def _commit_sha(repo_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            return proc.stdout.strip() or "unknown"
    except Exception:
        pass
    return "unknown"


def build_payload(
    repo_root: Path,
    *,
    run_id: str | None = None,
    user_id: str = "demo-user-1",
    run_date: str | None = None,
    generated_at: str | None = None,
    dataset_id: str = "geef_proteomics_v1",
    chain_id: str | None = None,
    upstream_run_id: str | None = None,
    upstream_contract: str = "statistician_to_questions",
) -> dict:
    generated = str(generated_at or datetime.now(timezone.utc).isoformat())
    rid = str(run_id or f"qa-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}")
    chain_token = str(chain_id or "chain_demo")
    mask_id = f"mask_{str(user_id)}"
    return {
        "contract": "questions_to_coherence",
        "version": 1,
        "run_id": rid,
        "user_id": str(user_id),
        "date": str(run_date or date.today().isoformat()),
        "coherence": {
            "r": 0.37,
            "theta": 1.24,
            "theta_defined": True,
            "z": [0.12, 0.35],
            "velocity": -0.08,
            "acceleration": -0.01,
            "uncertainty": 0.11,
            "semantic_axis_scores": {
                "metabolic": 0.12,
                "autonomic": 0.18,
                "immune": 0.08,
                "affective": 0.44,
            },
            "semantic_concentration": 0.68,
            "projection_kind": "canonical_coherence",
            "kappa": 0.71,
            "sigma_kappa": 0.14,
            "concordance": 0.77,
            "components": {
                "distance": 0.68,
                "stability": 0.81,
                "concordance": 0.77,
                "recovery": 0.74,
                "alignment": 0.68,
            },
            "attractor_state": {
                "sleep_quality": 0.58,
                "energy_vitality": 0.62,
                "gut_gi": 0.55,
                "glycemic_risk": 0.48,
            },
            "attractor_sigma_diag": {
                "sleep_quality": 0.11,
                "energy_vitality": 0.12,
                "gut_gi": 0.13,
                "glycemic_risk": 0.10,
            },
        },
        "identity_mask": {
            "mu": [0.14, -0.08, 0.03, 0.27],
            "sigma": [0.10, 0.09, 0.12, 0.07],
            "constraints": {
                "gcv_available": False,
                "safety_guard": "standard",
            },
        },
        "candidate_state_context": {
            "state_alignment": {
                "run_id": rid,
                "dataset_id": str(dataset_id),
                "chain_id": chain_token,
                "source_contract": "questions_to_coherence",
                "source_repo": "questions-agent-platform",
                "upstream_contract": str(upstream_contract),
            },
            "subject_scope": {
                "subject_id": str(user_id),
                "identity_mask_id": mask_id,
            },
            "coherence_anchor": {
                "r": 0.37,
                "theta": 1.24,
                "uncertainty": 0.11,
                "kappa": 0.71,
                "projection_kind": "canonical_coherence",
            },
            "questionnaire_pressure": {
                "completion_rate_7d": 0.82,
                "completion_rate_14d": 0.79,
                "completion_rate_30d": 0.76,
                "burden_ms_median_14d": 74000,
                "safety_trigger_active": False,
                "recommended_question_domains": ["metabolic", "immune", "sleep", "stress"],
            },
            "identity_signal": {
                "identity_mask_id": mask_id,
                "safety_guard": "standard",
                "gcv_available": False,
            },
        },
        "metadata": {
            "repo": "questions-agent-platform",
            "commit_sha": _commit_sha(repo_root),
            "generated_at": generated,
            "dataset_id": str(dataset_id),
            "chain_id": chain_token,
            "upstream_run_id": str(upstream_run_id or rid),
            "upstream_contract": str(upstream_contract),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit questions_to_coherence payload")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--run-id", default=None, help="Optional shared run id for cross-repo chain validation")
    parser.add_argument("--user-id", default="demo-user-1", help="User identifier")
    parser.add_argument("--date", default=None, help="Logical date for the coherence payload")
    parser.add_argument("--generated-at", default=None, help="Optional shared generated_at timestamp")
    parser.add_argument("--dataset-id", default="geef_proteomics_v1", help="Dataset identifier carried in metadata")
    parser.add_argument("--chain-id", default=None, help="Optional shared chain identifier")
    parser.add_argument("--upstream-run-id", default=None, help="Run id of the upstream statistician payload")
    parser.add_argument("--upstream-contract", default="statistician_to_questions", help="Upstream contract name")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            build_payload(
                repo_root,
                run_id=args.run_id,
                user_id=args.user_id,
                run_date=args.date,
                generated_at=args.generated_at,
                dataset_id=args.dataset_id,
                chain_id=args.chain_id,
                upstream_run_id=args.upstream_run_id,
                upstream_contract=args.upstream_contract,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(str(out))


if __name__ == "__main__":
    main()
