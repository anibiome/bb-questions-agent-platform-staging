from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from questions_agent_platform.policy.bandit import PolicyParams
from questions_agent_platform.policy.features import FeatureMapping, FEATURE_VERSION


class PolicyArtifactError(RuntimeError):
    """Raised when a configured policy artifact is missing or unreadable."""


def list_policy_versions(policy_root: str) -> List[str]:
    root = Path(policy_root)
    versions = root / "versions"
    if not versions.exists():
        return []
    out = []
    for p in versions.iterdir():
        if p.is_dir() and (p / "policy_params.json").exists():
            out.append(p.name)
    out.sort()
    return out


def get_active_policy_version(policy_root: str) -> Optional[str]:
    path = Path(policy_root) / "active_policy_version.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    v = data.get("active_version")
    return str(v) if v else None


def set_active_policy_version(policy_root: str, version: str) -> None:
    path = Path(policy_root) / "active_policy_version.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"active_version": str(version)}, ensure_ascii=False, indent=2), encoding="utf-8")


def load_policy_params(policy_root: str, policy_version: str) -> PolicyParams:
    path = Path(policy_root) / "versions" / str(policy_version) / "policy_params.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return PolicyParams(
        policy_version=str(data.get("policy_version") or policy_version),
        feature_version=str(data.get("feature_version") or FEATURE_VERSION),
        lambda_reg=float(data.get("lambda_reg") or 1.0),
        A=tuple(tuple(float(x) for x in row) for row in (data.get("A") or [])),
        b=tuple(float(x) for x in (data.get("b") or [])),
        feature_means=tuple(float(x) for x in (data.get("feature_means") or [])),
        feature_stds=tuple(float(x) for x in (data.get("feature_stds") or [])),
    )


def save_policy_params(policy_root: str, params: PolicyParams, *, metadata: Optional[Dict[str, Any]] = None) -> str:
    version = str(params.policy_version)
    base = Path(policy_root) / "versions" / version
    base.mkdir(parents=True, exist_ok=True)

    payload = {
        "policy_version": params.policy_version,
        "feature_version": params.feature_version,
        "lambda_reg": float(params.lambda_reg),
        "A": [list(row) for row in params.A],
        "b": list(params.b),
        "feature_means": list(params.feature_means),
        "feature_stds": list(params.feature_stds),
    }
    (base / "policy_params.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if metadata is not None:
        (base / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return version


def make_default_policy_params(*, policy_version: str, mapping: FeatureMapping, lambda_reg: float = 1.0) -> PolicyParams:
    """
    Initialize a safe default policy that reproduces deterministic scoring.

    We set the prior mean to emphasize the deterministic_score feature so the policy
    is useful immediately even before training.
    """
    d = len(mapping.names)
    A = [[0.0] * d for _ in range(d)]
    for i in range(d):
        A[i][i] = float(lambda_reg)

    b = [0.0] * d
    det_idx = mapping.index.get("deterministic_score")
    if det_idx is not None:
        b[det_idx] = float(lambda_reg) * 1.0

    return PolicyParams(
        policy_version=str(policy_version),
        feature_version=mapping.feature_version,
        lambda_reg=float(lambda_reg),
        A=tuple(tuple(row) for row in A),
        b=tuple(b),
        feature_means=tuple(0.0 for _ in range(d)),
        feature_stds=tuple(1.0 for _ in range(d)),
    )


def resolve_policy_params(
    policy_root: str,
    *,
    requested_version: Optional[str] = None,
    default_version: Optional[str] = None,
    mapping: Optional[FeatureMapping] = None,
    lambda_reg: float = 1.0,
    allow_bootstrap_default: bool = False,
) -> PolicyParams:
    """Load a configured policy artifact or bootstrap a default only for an empty registry."""

    root = Path(policy_root)
    available_versions = list_policy_versions(policy_root)
    available_set = set(available_versions)

    requested = str(requested_version or "").strip() or None
    active = str(get_active_policy_version(policy_root) or "").strip() or None
    fallback = str(default_version or "").strip() or None

    version = requested or active
    if version is None:
        if fallback and fallback in available_set:
            version = fallback
        elif len(available_versions) == 1:
            version = available_versions[0]
        elif len(available_versions) > 1:
            raise PolicyArtifactError(
                f"Multiple policy versions exist under '{root}' but no active version is configured."
            )
        elif allow_bootstrap_default:
            if mapping is None:
                raise ValueError("mapping is required when bootstrapping default policy params")
            return make_default_policy_params(
                policy_version=fallback or "v1",
                mapping=mapping,
                lambda_reg=lambda_reg,
            )
        else:
            raise PolicyArtifactError(f"No policy artifacts found under '{root}'.")

    artifact_path = root / "versions" / str(version) / "policy_params.json"
    if not artifact_path.exists():
        if allow_bootstrap_default and not available_versions and active is None:
            if mapping is None:
                raise ValueError("mapping is required when bootstrapping default policy params")
            return make_default_policy_params(
                policy_version=str(version),
                mapping=mapping,
                lambda_reg=lambda_reg,
            )
        known = ", ".join(available_versions) if available_versions else "none"
        raise PolicyArtifactError(
            f"Policy artifact for version '{version}' is missing under '{root}' (available: {known})."
        )

    try:
        return load_policy_params(policy_root, str(version))
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        raise PolicyArtifactError(
            f"Policy artifact for version '{version}' under '{root}' is unreadable: {exc}"
        ) from exc
