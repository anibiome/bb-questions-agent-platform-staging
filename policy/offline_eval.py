from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from questions_agent_platform.policy.features import build_feature_mapping_v1, default_stats, featurize_v1
from questions_agent_platform.policy.registry import resolve_policy_params
from questions_agent_platform.policy.types import CandidateItem, CandidateSet, PolicyContext


def evaluate_sqlite(
    *,
    db_path: str,
    policy_root: str,
    policy_version: Optional[str] = None,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
    include_shadow: bool = False,
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
    ver = str(params.policy_version or "v1")

    mean_w = _posterior_mean(params.A, params.b, dim=len(mapping.names))
    means, stds = _feature_stats_from_params(params.feature_means, params.feature_stds, dim=len(mapping.names))

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = _fetch_eval_rows(conn, min_date=min_date, max_date=max_date, include_shadow=include_shadow)
        # Reward model for DR.
        X: List[List[float]] = []
        y: List[float] = []
        parsed_rows: List[Dict[str, Any]] = []

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
            parsed_rows.append(
                {
                    "propensity": p_slate,
                    "ctx": ctx,
                    "cand": cand,
                    "logged_selected": tuple(logged_selected),
                    "reward": reward,
                    "selection_mode": str(r["selection_mode"] or ""),
                    "mode": str(r["mode"] or ""),
                    "user_id": str(r["user_id"] or ""),
                    "date": str(r["date"] or "1970-01-01"),
                    "completion_rate": float(r["completion_rate"]) if r["completion_rate"] is not None else None,
                    "z_delta_norm": float(r["z_delta_norm"]) if r["z_delta_norm"] is not None else None,
                    "context_completion_rate_30d": (
                        float(ctx.completion_rate_30d) if ctx.completion_rate_30d is not None else None
                    ),
                }
            )

        theta = _ridge_fit(X, y, lambda_reg=1.0) if X else [0.0] * len(mapping.names)
        seg_meta = _annotate_segments(parsed_rows)
        overall = _compute_eval_metrics(
            parsed_rows=parsed_rows,
            mapping=mapping,
            mean_w=mean_w,
            theta=theta,
            feature_means=means,
            feature_stds=stds,
        )

        segments: Dict[str, Any] = {}
        for name in ("new_users", "existing_users", "high_drift_windows", "low_adherence_users"):
            subset = [r for r in parsed_rows if name in r.get("segments", ())]
            segments[name] = _compute_eval_metrics(
                parsed_rows=subset,
                mapping=mapping,
                mean_w=mean_w,
                theta=theta,
                feature_means=means,
                feature_stds=stds,
            )

        return {
            "policy_version": ver,
            "generated_at_utc": _now_iso(),
            **overall,
            "segments": segments,
            "segment_meta": seg_meta,
            "notes": (
                "IPS/DR computed on logs with explicit propensities and observed rewards. "
                "Use policy_live for valid off-policy estimates; shadow mode can be included for diagnostics only."
            ),
        }
    finally:
        conn.close()


def _fetch_eval_rows(
    conn: sqlite3.Connection,
    *,
    min_date: Optional[str],
    max_date: Optional[str],
    include_shadow: bool,
) -> List[sqlite3.Row]:
    where = ["d.propensities_json IS NOT NULL", "o.total_reward IS NOT NULL", "d.context_json IS NOT NULL", "d.candidate_set_json IS NOT NULL"]
    if include_shadow:
        where.append("(d.selection_mode IN ('policy_live','policy_shadow'))")
    else:
        where.append("d.selection_mode='policy_live'")
        where.append("d.mode='live'")
    params: List[Any] = []
    if min_date:
        where.append("d.date>=?")
        params.append(str(min_date))
    if max_date:
        where.append("d.date<=?")
        params.append(str(max_date))
    sql = f"""
    SELECT d.user_id, d.date, d.selection_mode, d.mode, d.context_json, d.candidate_set_json,
           d.selected_item_ids_json, d.propensities_json, d.deterministic_baseline_selected_json,
           o.total_reward, o.completion_rate, o.z_delta_norm
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
    candidates: List[CandidateItem] = []
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
    return tuple((mandatory + [cid for _, _, cid in scored[:k_remaining]])[: int(cand.k_core)])


def _sanitize_baseline(cand: CandidateSet, baseline_ids: Sequence[str]) -> Tuple[str, ...]:
    by_id = {c.item_id: c for c in cand.candidates}
    selected: List[str] = []
    for item_id in cand.mandatory_item_ids:
        c = by_id.get(str(item_id))
        if c and "blocked" not in c.constraint_tags and str(item_id) not in selected:
            selected.append(str(item_id))
    for item_id in baseline_ids:
        c = by_id.get(str(item_id))
        if not c or "blocked" in c.constraint_tags:
            continue
        if str(item_id) in selected:
            continue
        selected.append(str(item_id))
        if len(selected) >= int(cand.k_core):
            break
    if len(selected) < int(cand.k_core):
        remaining = [c for c in cand.candidates if c.item_id not in selected and "blocked" not in c.constraint_tags]
        remaining.sort(key=lambda c: (float(c.deterministic_score), c.item_id), reverse=True)
        for c in remaining:
            selected.append(c.item_id)
            if len(selected) >= int(cand.k_core):
                break
    return tuple(selected[: int(cand.k_core)])


def _compute_eval_metrics(
    *,
    parsed_rows: Sequence[Dict[str, Any]],
    mapping,
    mean_w: Sequence[float],
    theta: Sequence[float],
    feature_means: Sequence[float],
    feature_stds: Sequence[float],
) -> Dict[str, Any]:
    n = 0
    n_match_policy = 0
    n_match_baseline = 0
    ips_policy_sum = 0.0
    ips_baseline_sum = 0.0
    dr_policy_sum = 0.0
    dr_baseline_sum = 0.0

    for row in parsed_rows:
        p_slate = float(row["propensity"])
        ctx = row["ctx"]
        cand = row["cand"]
        logged_selected = row["logged_selected"]
        reward = float(row["reward"])

        policy_selected = _greedy_select(
            ctx, cand, mapping, mean_w, feature_means=feature_means, feature_stds=feature_stds
        )
        baseline_selected = _sanitize_baseline(cand, cand.deterministic_baseline_selected)

        match_policy = tuple(logged_selected) == tuple(policy_selected)
        match_baseline = tuple(logged_selected) == tuple(baseline_selected)

        phi_logged = _phi_sum(ctx, cand, logged_selected, mapping, feature_means=feature_means, feature_stds=feature_stds)
        phi_policy = _phi_sum(ctx, cand, policy_selected, mapping, feature_means=feature_means, feature_stds=feature_stds)
        phi_baseline = _phi_sum(ctx, cand, baseline_selected, mapping, feature_means=feature_means, feature_stds=feature_stds)

        r_hat_logged = _dot(theta, phi_logged)
        r_hat_policy = _dot(theta, phi_policy)
        r_hat_baseline = _dot(theta, phi_baseline)

        if match_policy:
            ips_policy_sum += reward / p_slate
            n_match_policy += 1
        if match_baseline:
            ips_baseline_sum += reward / p_slate
            n_match_baseline += 1

        dr_policy = r_hat_policy + ((reward - r_hat_logged) / p_slate if match_policy else 0.0)
        dr_baseline = r_hat_baseline + ((reward - r_hat_logged) / p_slate if match_baseline else 0.0)
        dr_policy_sum += dr_policy
        dr_baseline_sum += dr_baseline
        n += 1

    ips_policy = (ips_policy_sum / n) if n else None
    ips_baseline = (ips_baseline_sum / n) if n else None
    dr_policy = (dr_policy_sum / n) if n else None
    dr_baseline = (dr_baseline_sum / n) if n else None
    return {
        "n_eval_rows": int(n),
        "n_match_policy": int(n_match_policy),
        "n_match_baseline": int(n_match_baseline),
        "ips_policy": _f(ips_policy),
        "ips_baseline": _f(ips_baseline),
        "ips_delta_policy_minus_baseline": _f(_delta(ips_policy, ips_baseline)),
        "dr_policy": _f(dr_policy),
        "dr_baseline": _f(dr_baseline),
        "dr_delta_policy_minus_baseline": _f(_delta(dr_policy, dr_baseline)),
    }


def _annotate_segments(parsed_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_user: Dict[str, List[Dict[str, Any]]] = {}
    for row in parsed_rows:
        by_user.setdefault(str(row.get("user_id") or ""), []).append(row)
    for _, seq in by_user.items():
        seq.sort(key=lambda r: str(r.get("date") or ""))
        for i, r in enumerate(seq, start=1):
            r["user_decision_index"] = int(i)

    z_vals = [float(r["z_delta_norm"]) for r in parsed_rows if r.get("z_delta_norm") is not None]
    drift_threshold = _quantile(z_vals, 0.75) if z_vals else None
    new_user_cutoff = 30
    low_adherence_threshold = 0.80
    for row in parsed_rows:
        labels: List[str] = []
        idx = int(row.get("user_decision_index") or 0)
        if idx <= new_user_cutoff:
            labels.append("new_users")
        if idx > new_user_cutoff:
            labels.append("existing_users")
        z = row.get("z_delta_norm")
        if drift_threshold is not None and z is not None and float(z) >= float(drift_threshold):
            labels.append("high_drift_windows")
        completion = row.get("completion_rate")
        c30 = row.get("context_completion_rate_30d")
        if (completion is not None and float(completion) < low_adherence_threshold) or (
            c30 is not None and float(c30) < low_adherence_threshold
        ):
            labels.append("low_adherence_users")
        row["segments"] = tuple(labels)
    return {
        "new_user_decision_cutoff": int(new_user_cutoff),
        "low_adherence_threshold": float(low_adherence_threshold),
        "high_drift_threshold_z_delta_norm_p75": _f(drift_threshold),
    }


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
    k = max(1, len(selected_ids))
    return [float(v) / float(k) for v in out]


def _ridge_fit(X: List[List[float]], y: List[float], *, lambda_reg: float) -> List[float]:
    if not X:
        return []
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


def _posterior_mean(A_in: Sequence[Sequence[float]], b_in: Sequence[float], *, dim: int) -> List[float]:
    d = int(dim)
    if not A_in or not b_in:
        return [0.0] * d
    A = [list(row) for row in A_in]
    b = [float(x) for x in b_in]
    if len(b) != d or any(len(r) != d for r in A):
        return [0.0] * d
    L = _cholesky(A)
    y = _solve_lower(L, b)
    return _solve_upper_transpose(L, y)


def _feature_stats_from_params(
    means_in: Sequence[float],
    stds_in: Sequence[float],
    *,
    dim: int,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    if len(means_in) == dim and len(stds_in) == dim:
        return tuple(float(x) for x in means_in), tuple(float(x) for x in stds_in)
    return default_stats(dim)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(float(x) * float(y) for x, y in zip(a, b)))


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


def _parse_date(s: str):
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return datetime(1970, 1, 1, tzinfo=timezone.utc).date()


def _json_obj(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    try:
        obj = json.loads(str(raw))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _json_list(raw: Any) -> List[Any]:
    if raw is None:
        return []
    try:
        obj = json.loads(str(raw))
        return obj if isinstance(obj, list) else []
    except Exception:
        return []


def _f(v: Optional[float]) -> Optional[float]:
    return float(v) if v is not None else None


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def _quantile(vals: Sequence[float], q: float) -> Optional[float]:
    arr = sorted(float(v) for v in vals)
    if not arr:
        return None
    qq = max(0.0, min(1.0, float(q)))
    idx = qq * float(len(arr) - 1)
    lo = int(idx)
    hi = min(len(arr) - 1, lo + 1)
    frac = idx - float(lo)
    return float(arr[lo] * (1.0 - frac) + arr[hi] * frac)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
