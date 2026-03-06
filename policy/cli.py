from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from questions_agent_platform.policy.bandit import PolicyParams
from questions_agent_platform.policy.features import (
    build_feature_mapping_v1,
    compute_running_stats,
    default_stats,
    featurize_raw_v1,
    featurize_v1,
)
from questions_agent_platform.policy.registry import (
    make_default_policy_params,
    resolve_policy_params,
    save_policy_params,
    set_active_policy_version,
)
from questions_agent_platform.policy.types import CandidateItem, CandidateSet, PolicyContext


def main() -> None:
    p = argparse.ArgumentParser(prog="questions-agent-policy")
    sub = p.add_subparsers(dest="cmd", required=True)

    init_p = sub.add_parser("init-demo", help="Create a default policy artifact + set active version")
    init_p.add_argument("--policy-root", required=True)
    init_p.add_argument("--version", default="v1")

    train_p = sub.add_parser("train-sqlite", help="Train/update posterior params from SQLite logs")
    train_p.add_argument("--db", required=True, help="Path to qa.sqlite")
    train_p.add_argument("--policy-root", required=True)
    train_p.add_argument("--in-version", default=None, help="Policy version to start from (default: active)")
    train_p.add_argument("--out-version", required=True)
    train_p.add_argument("--min-date", default=None, help="YYYY-MM-DD (optional)")
    train_p.add_argument("--max-date", default=None, help="YYYY-MM-DD (optional)")

    eval_p = sub.add_parser("eval-sqlite", help="Offline eval (IPS + doubly robust) on exploration logs")
    eval_p.add_argument("--db", required=True, help="Path to qa.sqlite")
    eval_p.add_argument("--policy-root", required=True)
    eval_p.add_argument("--policy-version", default=None, help="Policy version to evaluate (default: active)")
    eval_p.add_argument("--min-date", default=None, help="YYYY-MM-DD (optional)")
    eval_p.add_argument("--max-date", default=None, help="YYYY-MM-DD (optional)")

    args = p.parse_args()

    if args.cmd == "init-demo":
        _cmd_init_demo(policy_root=args.policy_root, version=args.version)
        return

    if args.cmd == "train-sqlite":
        _cmd_train_sqlite(
            db_path=args.db,
            policy_root=args.policy_root,
            in_version=args.in_version,
            out_version=args.out_version,
            min_date=args.min_date,
            max_date=args.max_date,
        )
        return

    if args.cmd == "eval-sqlite":
        _cmd_eval_sqlite(
            db_path=args.db,
            policy_root=args.policy_root,
            policy_version=args.policy_version,
            min_date=args.min_date,
            max_date=args.max_date,
        )
        return


def _cmd_init_demo(*, policy_root: str, version: str) -> None:
    mapping = build_feature_mapping_v1()
    params = make_default_policy_params(policy_version=str(version), mapping=mapping, lambda_reg=1.0)
    save_policy_params(policy_root, params, metadata={"created_at": _now_iso(), "kind": "default_demo"})
    set_active_policy_version(policy_root, str(version))
    print(json.dumps({"ok": True, "policy_root": policy_root, "active_version": str(version)}, indent=2))


def _cmd_train_sqlite(
    *,
    db_path: str,
    policy_root: str,
    in_version: Optional[str],
    out_version: str,
    min_date: Optional[str],
    max_date: Optional[str],
) -> None:
    mapping = build_feature_mapping_v1()
    params = resolve_policy_params(
        policy_root,
        requested_version=in_version,
        default_version="v1",
        mapping=mapping,
        lambda_reg=1.0,
        allow_bootstrap_default=True,
    )
    base_version = str(params.policy_version or "v1")

    A, b = _ensure_dense_posterior(params, dim=len(mapping.names))

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = _fetch_training_rows(conn, min_date=min_date, max_date=max_date)
        train_events: List[Tuple[PolicyContext, CandidateItem, float]] = []
        raw_vectors: List[List[float]] = []
        used = 0
        skipped = 0
        for r in rows:
            ctx_json = r["context_json"]
            cand_json = r["candidate_set_json"]
            if not ctx_json or not cand_json:
                skipped += 1
                continue
            total_reward = r["total_reward"]
            if total_reward is None:
                skipped += 1
                continue

            ctx = _context_from_json(json.loads(ctx_json))
            cand = _candidate_set_from_json(json.loads(cand_json))
            selected = json.loads(r["selected_item_ids_json"])
            if not selected:
                skipped += 1
                continue

            per_item_reward = float(total_reward) / max(1, len(selected))
            by_id = {c.item_id: c for c in cand.candidates}
            for item_id in selected:
                c = by_id.get(str(item_id))
                if not c:
                    continue
                train_events.append((ctx, c, per_item_reward))
                raw_vectors.append(featurize_raw_v1(ctx, c, mapping))
            used += 1

        means, stds = _feature_stats_from_training(
            raw_vectors=raw_vectors,
            fallback_means=params.feature_means,
            fallback_stds=params.feature_stds,
            dim=len(mapping.names),
        )
        for ctx, c, per_item_reward in train_events:
            phi = featurize_v1(ctx, c, mapping, feature_means=means, feature_stds=stds)
            _rank1_update(A, phi)
            _axpy(b, phi, per_item_reward)

        out_params = PolicyParams(
            policy_version=str(out_version),
            feature_version=params.feature_version,
            lambda_reg=float(params.lambda_reg),
            A=tuple(tuple(float(x) for x in row) for row in A),
            b=tuple(float(x) for x in b),
            feature_means=tuple(float(x) for x in means),
            feature_stds=tuple(float(x) for x in stds),
        )
        save_policy_params(
            policy_root,
            out_params,
            metadata={
                "trained_from": base_version,
                "trained_at": _now_iso(),
                "rows_used": int(used),
                "rows_skipped": int(skipped),
                "feature_rows": int(len(raw_vectors)),
                "min_date": min_date,
                "max_date": max_date,
            },
        )

        print(json.dumps({"ok": True, "out_version": str(out_version), "rows_used": used, "rows_skipped": skipped}, indent=2))
    finally:
        conn.close()


def _cmd_eval_sqlite(
    *,
    db_path: str,
    policy_root: str,
    policy_version: Optional[str],
    min_date: Optional[str],
    max_date: Optional[str],
) -> None:
    from questions_agent_platform.policy.offline_eval import evaluate_sqlite

    report = evaluate_sqlite(
        db_path=db_path,
        policy_root=policy_root,
        policy_version=policy_version,
        min_date=min_date,
        max_date=max_date,
        include_shadow=False,
    )
    print(json.dumps(report, indent=2))


def _fetch_training_rows(conn: sqlite3.Connection, *, min_date: Optional[str], max_date: Optional[str]) -> List[sqlite3.Row]:
    where = ["d.selection_mode='policy_live'", "d.mode='live'", "o.total_reward IS NOT NULL"]
    params: List[Any] = []
    if min_date:
        where.append("d.date>=?")
        params.append(str(min_date))
    if max_date:
        where.append("d.date<=?")
        params.append(str(max_date))
    sql = f"""
    SELECT d.*, o.total_reward
    FROM policy_decisions d
    JOIN policy_outcomes o ON o.decision_id = d.decision_id
    WHERE {' AND '.join(where)}
    ORDER BY d.date ASC;
    """
    return conn.execute(sql, params).fetchall()


def _fetch_eval_rows(conn: sqlite3.Connection, *, min_date: Optional[str], max_date: Optional[str]) -> List[sqlite3.Row]:
    where = ["d.propensities_json IS NOT NULL", "o.total_reward IS NOT NULL"]
    params: List[Any] = []
    if min_date:
        where.append("d.date>=?")
        params.append(str(min_date))
    if max_date:
        where.append("d.date<=?")
        params.append(str(max_date))
    sql = f"""
    SELECT d.*, o.total_reward
    FROM policy_decisions d
    JOIN policy_outcomes o ON o.decision_id = d.decision_id
    WHERE {' AND '.join(where)}
    ORDER BY d.date ASC;
    """
    return conn.execute(sql, params).fetchall()


def _boolish(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if not text:
        return default
    if text in {"false", "0", "no", "n", "off", "none", "null"}:
        return False
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    return default


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, *, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def _normalize_string_list(value: Any) -> Tuple[str, ...]:
    if isinstance(value, (list, tuple, set)):
        items = value
    elif value is None:
        items = ()
    else:
        items = (value,)
    return tuple(str(item).strip() for item in items if str(item).strip())


def _normalize_dict_rows(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, dict):
        return [dict(value)]
    if isinstance(value, (list, tuple)):
        return [dict(item) for item in value if isinstance(item, dict)]
    return []


def _context_from_json(obj: Dict[str, Any]) -> PolicyContext:
    return PolicyContext(
        anifold_z=obj.get("anifold_z"),
        z_uncertainty_diag=obj.get("z_uncertainty_diag"),
        z_velocity=obj.get("z_velocity"),
        z_distance_to_attractor=obj.get("z_distance_to_attractor"),
        completion_rate_7d=obj.get("completion_rate_7d"),
        completion_rate_14d=obj.get("completion_rate_14d"),
        completion_rate_30d=obj.get("completion_rate_30d"),
        burden_ms_median_14d=obj.get("burden_ms_median_14d"),
        day_of_week=obj.get("day_of_week"),
        safety_trigger_active=_boolish(obj.get("safety_trigger_active", False), default=False),
        allow_context_batches=_boolish(obj.get("allow_context_batches", True), default=True),
        identity_mask_id=obj.get("identity_mask_id"),
    )


def _candidate_set_from_json(obj: Dict[str, Any]) -> CandidateSet:
    candidates = []
    for c in _normalize_dict_rows(obj.get("candidates")):
        candidates.append(
            CandidateItem(
                item_id=str(c.get("item_id")),
                item_type=str(c.get("item_type")),
                scale_ids=_normalize_string_list(c.get("scale_ids")),
                deterministic_score=_safe_float(c.get("deterministic_score"), default=0.0),
                constraint_tags=_normalize_string_list(c.get("constraint_tags")),
                reason_codes=_normalize_string_list(c.get("reason_codes")),
                features=dict(c.get("features") or {}),
            )
        )

    return CandidateSet(
        user_id=str(obj.get("user_id") or ""),
        day=_parse_date(str(obj.get("day") or "1970-01-01")),
        k_core=_safe_int(obj.get("k_core"), default=5),
        candidates=tuple(candidates),
        mandatory_item_ids=_normalize_string_list(obj.get("mandatory_item_ids")),
        deterministic_baseline_selected=_normalize_string_list(obj.get("deterministic_baseline_selected")),
    )


def _ensure_dense_posterior(params: PolicyParams, *, dim: int) -> Tuple[List[List[float]], List[float]]:
    d = int(dim)
    if params.A and params.b and len(params.b) == d and all(len(r) == d for r in params.A):
        return [list(row) for row in params.A], [float(x) for x in params.b]
    # Default to ridge prior.
    A = [[0.0] * d for _ in range(d)]
    for i in range(d):
        A[i][i] = float(params.lambda_reg or 1.0)
    b = [0.0] * d
    return A, b


def _rank1_update(A: List[List[float]], v: Sequence[float]) -> None:
    d = len(v)
    for i in range(d):
        vi = float(v[i])
        for j in range(d):
            A[i][j] += vi * float(v[j])


def _axpy(y: List[float], x: Sequence[float], a: float) -> None:
    for i in range(len(y)):
        y[i] += float(a) * float(x[i])


def _posterior_mean(params: PolicyParams, *, dim: int) -> List[float]:
    d = int(dim)
    if not params.A or not params.b:
        return [0.0] * d
    A = [list(row) for row in params.A]
    b = [float(x) for x in params.b]
    if len(b) != d or any(len(r) != d for r in A):
        return [0.0] * d
    # Solve A w = b via Cholesky.
    L = _cholesky(A)
    y = _solve_lower(L, b)
    return _solve_upper_transpose(L, y)


def _greedy_select(
    ctx: PolicyContext,
    cand: CandidateSet,
    mapping,
    weights: Sequence[float],
    *,
    feature_means: Sequence[float],
    feature_stds: Sequence[float],
) -> Tuple[str, ...]:
    by_id = {c.item_id: c for c in cand.candidates}
    mandatory = [i for i in cand.mandatory_item_ids if i in by_id and "blocked" not in by_id[i].constraint_tags]
    k_remaining = max(0, int(cand.k_core) - len(mandatory))

    optionals = [c for c in cand.candidates if c.item_id not in mandatory and "blocked" not in c.constraint_tags]
    scored = []
    for c in optionals:
        phi = featurize_v1(ctx, c, mapping, feature_means=feature_means, feature_stds=feature_stds)
        scored.append((_dot(weights, phi), float(c.deterministic_score), c.item_id))
    scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)

    chosen = [cid for _, _, cid in scored[:k_remaining]]
    out = tuple(mandatory + chosen)
    if len(out) != int(cand.k_core):
        return tuple(cand.deterministic_baseline_selected)
    return out


def _phi_sum(
    ctx: PolicyContext,
    cand: CandidateSet,
    selected_ids: Sequence[str],
    mapping,
    *,
    feature_means: Sequence[float],
    feature_stds: Sequence[float],
) -> List[float]:
    by_id = {c.item_id: c for c in cand.candidates}
    out = [0.0] * len(mapping.names)
    for item_id in selected_ids:
        c = by_id.get(str(item_id))
        if not c:
            continue
        phi = featurize_v1(ctx, c, mapping, feature_means=feature_means, feature_stds=feature_stds)
        for i in range(len(out)):
            out[i] += float(phi[i])
    # Normalize by k for stability.
    k = max(1, len(selected_ids))
    return [float(v) / float(k) for v in out]


def _ridge_fit(X: List[List[float]], y: List[float], *, lambda_reg: float) -> List[float]:
    if not X:
        return []
    n = len(X)
    d = len(X[0])
    A = [[0.0] * d for _ in range(d)]
    b = [0.0] * d
    for i in range(d):
        A[i][i] = float(lambda_reg)
    for xi, yi in zip(X, y):
        for i in range(d):
            b[i] += float(xi[i]) * float(yi)
            for j in range(d):
                A[i][j] += float(xi[i]) * float(xi[j])
    L = _cholesky(A)
    y2 = _solve_lower(L, b)
    return _solve_upper_transpose(L, y2)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(float(x) * float(y) for x, y in zip(a, b)))


def _feature_stats_from_params(*, params: PolicyParams, dim: int) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    if len(params.feature_means) == dim and len(params.feature_stds) == dim:
        return tuple(float(x) for x in params.feature_means), tuple(float(x) for x in params.feature_stds)
    return default_stats(dim)


def _feature_stats_from_training(
    *,
    raw_vectors: Sequence[Sequence[float]],
    fallback_means: Sequence[float],
    fallback_stds: Sequence[float],
    dim: int,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    if raw_vectors:
        means, stds = compute_running_stats(raw_vectors)
        if len(means) == dim and len(stds) == dim:
            return tuple(float(x) for x in means), tuple(float(x) for x in stds)
    if len(fallback_means) == dim and len(fallback_stds) == dim:
        return tuple(float(x) for x in fallback_means), tuple(float(x) for x in fallback_stds)
    return default_stats(dim)


def _cholesky(A: List[List[float]]) -> List[List[float]]:
    d = len(A)
    L = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(i + 1):
            s = 0.0
            for k in range(j):
                s += L[i][k] * L[j][k]
            if i == j:
                val = float(A[i][i]) - s
                if val <= 1e-12:
                    raise ValueError("Matrix not positive definite")
                L[i][j] = float(val) ** 0.5
            else:
                L[i][j] = (float(A[i][j]) - s) / max(1e-12, L[j][j])
    return L


def _solve_lower(L: List[List[float]], b: Sequence[float]) -> List[float]:
    d = len(L)
    x = [0.0] * d
    for i in range(d):
        s = float(b[i])
        for j in range(i):
            s -= L[i][j] * x[j]
        x[i] = s / max(1e-12, L[i][i])
    return x


def _solve_upper_transpose(L: List[List[float]], b: Sequence[float]) -> List[float]:
    d = len(L)
    x = [0.0] * d
    for i in reversed(range(d)):
        s = float(b[i])
        for j in range(i + 1, d):
            s -= L[j][i] * x[j]
        x[i] = s / max(1e-12, L[i][i])
    return x


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_date(s: str):
    # YYYY-MM-DD
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return datetime(1970, 1, 1, tzinfo=timezone.utc).date()


if __name__ == "__main__":
    main()
