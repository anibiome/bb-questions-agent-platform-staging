from __future__ import annotations

"""Consolidated offline scientific evaluation (full replay + ablations + calibration)."""

import argparse
import json
import math
import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from questions_agent_platform.policy.features import build_feature_mapping_v1
from questions_agent_platform.policy.offline_eval import (
    _candidate_set_from_json,
    _context_from_json,
    _dot,
    _feature_stats_from_params,
    _fetch_eval_rows,
    _json_list,
    _json_obj,
    _phi_sum,
    _ridge_fit,
    evaluate_sqlite,
)
from questions_agent_platform.policy.registry import resolve_policy_params

MappingLike = Dict[str, Any]

_ABLATION_SPECS: Dict[str, Sequence[str]] = {
    "without_z_features": ("anifold_z", "z_velocity", "z_distance_to_attractor"),
    "without_uncertainty_features": ("z_uncertainty_diag",),
    "without_z_and_uncertainty": ("anifold_z", "z_velocity", "z_distance_to_attractor", "z_uncertainty_diag"),
}

_COMPARATIVE_METRICS: Dict[str, str] = {
    "dr": "dr_policy",
    "ips": "ips_policy",
}


def build_scientific_offline_eval_report(
    *,
    db_path: str,
    policy_root: str,
    policy_version: Optional[str] = None,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
    include_shadow: bool = False,
) -> Dict[str, Any]:
    full = evaluate_sqlite(
        db_path=db_path,
        policy_root=policy_root,
        policy_version=policy_version,
        min_date=min_date,
        max_date=max_date,
        include_shadow=include_shadow,
    )
    ablations = _build_ablations(
        db_path=db_path,
        policy_root=policy_root,
        policy_version=policy_version,
        min_date=min_date,
        max_date=max_date,
        include_shadow=include_shadow,
    )
    calibration = _compute_reward_calibration(
        db_path=db_path,
        policy_root=policy_root,
        policy_version=policy_version,
        min_date=min_date,
        max_date=max_date,
        include_shadow=include_shadow,
    )

    return {
        "version": "scientific_offline_eval_v1",
        "generated_at_utc": _now_iso(),
        "inputs": {
            "db_path": db_path,
            "policy_root": policy_root,
            "policy_version": policy_version,
            "min_date": min_date,
            "max_date": max_date,
            "include_shadow": bool(include_shadow),
        },
        "full_model": full,
        "ablations": ablations,
        "comparative": _build_comparative_metrics(ablations=ablations, full=full),
        "calibration": calibration,
        "notes": (
            "Ablations remove context keys before replay to estimate dependency on Z and uncertainty signals. "
            "Calibration is based on a ridge reward model fit on logged slate-level features."
        ),
    }


def _build_ablations(
    *,
    db_path: str,
    policy_root: str,
    policy_version: Optional[str],
    min_date: Optional[str],
    max_date: Optional[str],
    include_shadow: bool,
) -> Dict[str, Dict[str, Any]]:
    return {
        ablation_name: _evaluate_with_context_ablation(
            db_path=db_path,
            policy_root=policy_root,
            policy_version=policy_version,
            min_date=min_date,
            max_date=max_date,
            include_shadow=include_shadow,
            drop_keys=drop_keys,
        )
        for ablation_name, drop_keys in _ABLATION_SPECS.items()
    }


def _build_comparative_metrics(*, ablations: MappingLike, full: MappingLike) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for short_name, metric_name in _COMPARATIVE_METRICS.items():
        out[f"{short_name}_delta_without_z_minus_full"] = _delta_metric(
            ablations.get("without_z_features", {}),
            full,
            metric=metric_name,
        )
        out[f"{short_name}_delta_without_uncertainty_minus_full"] = _delta_metric(
            ablations.get("without_uncertainty_features", {}),
            full,
            metric=metric_name,
        )
        out[f"{short_name}_delta_without_z_uncertainty_minus_full"] = _delta_metric(
            ablations.get("without_z_and_uncertainty", {}),
            full,
            metric=metric_name,
        )
    return out


def _evaluate_with_context_ablation(
    *,
    db_path: str,
    policy_root: str,
    policy_version: Optional[str],
    min_date: Optional[str],
    max_date: Optional[str],
    include_shadow: bool,
    drop_keys: Sequence[str],
) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        tmp_db = Path(td) / "ablated.sqlite"
        shutil.copy2(db_path, tmp_db)
        _strip_context_keys(str(tmp_db), drop_keys)
        report = evaluate_sqlite(
            db_path=str(tmp_db),
            policy_root=policy_root,
            policy_version=policy_version,
            min_date=min_date,
            max_date=max_date,
            include_shadow=include_shadow,
        )
    report["ablation"] = {"dropped_context_keys": list(drop_keys)}
    return report


def _strip_context_keys(db_path: str, drop_keys: Sequence[str]) -> None:
    keys = {str(k) for k in drop_keys if str(k).strip()}
    if not keys:
        return
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT decision_id, context_json FROM policy_decisions WHERE context_json IS NOT NULL;").fetchall()
        for decision_id, context_json in rows:
            try:
                obj = json.loads(str(context_json))
            except Exception:
                obj = {}
            if not isinstance(obj, dict):
                obj = {}
            changed = False
            for key in keys:
                if key in obj:
                    obj.pop(key, None)
                    changed = True
            if changed:
                conn.execute(
                    "UPDATE policy_decisions SET context_json=? WHERE decision_id=?;",
                    (json.dumps(obj, ensure_ascii=False), str(decision_id)),
                )
        conn.commit()
    finally:
        conn.close()


def _compute_reward_calibration(
    *,
    db_path: str,
    policy_root: str,
    policy_version: Optional[str],
    min_date: Optional[str],
    max_date: Optional[str],
    include_shadow: bool,
) -> Dict[str, Any]:
    mapping = build_feature_mapping_v1()
    params = resolve_policy_params(
        policy_root,
        requested_version=policy_version,
        default_version="v1",
        mapping=mapping,
        lambda_reg=1.0,
        allow_bootstrap_default=True,
    )

    means, stds = _feature_stats_from_params(params.feature_means, params.feature_stds, dim=len(mapping.names))

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = _fetch_eval_rows(conn, min_date=min_date, max_date=max_date, include_shadow=include_shadow)
        X: List[List[float]] = []
        y: List[float] = []
        for r in rows:
            prop = _json_obj(r["propensities_json"])
            p_slate = float(prop.get("__slate__") or 0.0)
            if p_slate <= 0.0:
                continue

            ctx = _context_from_json(_json_obj(r["context_json"]))
            cand = _candidate_set_from_json(_json_obj(r["candidate_set_json"]))
            logged_selected = [str(i) for i in _json_list(r["selected_item_ids_json"])]
            if not logged_selected:
                continue
            reward = float(r["total_reward"])
            phi_logged = _phi_sum(ctx, cand, logged_selected, mapping, feature_means=means, feature_stds=stds)
            X.append(phi_logged)
            y.append(reward)
    finally:
        conn.close()

    if not X:
        return {
            "n_eval_rows": 0,
            "model": "ridge_reward",
            "mae": None,
            "rmse": None,
            "normalised_mae": None,
            "bins": [],
        }

    theta = _ridge_fit(X, y, lambda_reg=1.0)
    preds = [_dot(theta, xi) for xi in X]
    mae = sum(abs(float(p) - float(o)) for p, o in zip(preds, y)) / float(len(y))
    rmse = math.sqrt(sum((float(p) - float(o)) ** 2 for p, o in zip(preds, y)) / float(len(y)))
    y_span = (max(y) - min(y)) if y else 0.0
    nmae = (mae / y_span) if y_span > 1e-9 else None

    bins = _make_calibration_bins(preds, y, n_bins=5)
    ece = _expected_calibration_error(bins)
    return {
        "n_eval_rows": int(len(y)),
        "model": "ridge_reward",
        "mae": _f(mae),
        "rmse": _f(rmse),
        "normalised_mae": _f(nmae),
        "ece": _f(ece),
        "bins": bins,
    }


def _make_calibration_bins(preds: Sequence[float], observed: Sequence[float], *, n_bins: int) -> List[Dict[str, Any]]:
    pairs = sorted((float(p), float(o)) for p, o in zip(preds, observed))
    if not pairs:
        return []
    k = max(1, int(n_bins))
    chunk = int(math.ceil(len(pairs) / float(k)))
    out: List[Dict[str, Any]] = []
    for idx in range(k):
        start = idx * chunk
        end = min(len(pairs), start + chunk)
        if start >= end:
            break
        bucket = pairs[start:end]
        pred_vals = [p for p, _ in bucket]
        obs_vals = [o for _, o in bucket]
        avg_pred = sum(pred_vals) / float(len(pred_vals))
        avg_obs = sum(obs_vals) / float(len(obs_vals))
        out.append(
            {
                "bin_index": int(idx),
                "count": int(len(bucket)),
                "pred_range": {"min": _f(min(pred_vals)), "max": _f(max(pred_vals))},
                "avg_predicted_reward": _f(avg_pred),
                "avg_observed_reward": _f(avg_obs),
                "abs_error": _f(abs(avg_pred - avg_obs)),
            }
        )
    return out


def _expected_calibration_error(bins: Sequence[Dict[str, Any]]) -> Optional[float]:
    if not bins:
        return None
    total = sum(int(b.get("count", 0)) for b in bins)
    if total <= 0:
        return None
    acc = 0.0
    for b in bins:
        n = max(0, int(b.get("count", 0)))
        err = b.get("abs_error")
        if err is None:
            continue
        acc += (float(n) / float(total)) * float(err)
    return acc


def _delta_metric(a: MappingLike, b: MappingLike, *, metric: str) -> Optional[float]:
    av = _metric_value(a, metric)
    bv = _metric_value(b, metric)
    if av is None or bv is None:
        return None
    return float(av) - float(bv)


def _metric_value(obj: MappingLike, metric: str) -> Optional[float]:
    raw = obj.get(metric) if isinstance(obj, dict) else None
    if raw is None:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def _f(v: Optional[float]) -> Optional[float]:
    return float(v) if v is not None else None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run consolidated offline scientific evaluation (full + ablations).")
    parser.add_argument("--db", required=True, help="Path to SQLite database.")
    parser.add_argument("--policy-root", required=True, help="Path to policy artifact registry root.")
    parser.add_argument("--policy-version", default=None, help="Policy version to evaluate (default: active).")
    parser.add_argument("--min-date", default=None, help="Optional lower date bound (YYYY-MM-DD).")
    parser.add_argument("--max-date", default=None, help="Optional upper date bound (YYYY-MM-DD).")
    parser.add_argument("--include-shadow", action="store_true", help="Include policy_shadow rows in replay dataset.")
    parser.add_argument("--out-json", default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    report = build_scientific_offline_eval_report(
        db_path=str(args.db),
        policy_root=str(args.policy_root),
        policy_version=(str(args.policy_version) if args.policy_version else None),
        min_date=(str(args.min_date) if args.min_date else None),
        max_date=(str(args.max_date) if args.max_date else None),
        include_shadow=bool(args.include_shadow),
    )

    if args.out_json:
        out_path = Path(str(args.out_json))
        _write_json(out_path, report)
        print(json.dumps({"ok": True, "out_json": str(out_path)}, ensure_ascii=False))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
