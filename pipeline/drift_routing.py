from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from questions_agent_platform.pipeline.governance import canonical_domain_id


@dataclass(frozen=True)
class DriftRoute:
    route_id: str
    drift_domain: str
    action: str
    reason_code: str
    priority: int = 100
    scale_ids: Tuple[str, ...] = ()
    domain_ids: Tuple[str, ...] = ()
    triggered_instrument: Optional[str] = None


DEFAULT_DRIFT_ROUTING_VERSION = "drift_routing_v1"


def default_drift_routes() -> List[DriftRoute]:
    return [
        DriftRoute(
            route_id="route_glucose_metabolic",
            drift_domain="metabolic",
            action="targeted_scale_probe",
            reason_code="route_glucose_metabolic",
            priority=10,
            scale_ids=("scale_cm_findrisc",),
            domain_ids=("metabolic", "glucose", "cardiometabolic"),
            triggered_instrument="scale_cm_findrisc",
        ),
        DriftRoute(
            route_id="route_lipid_cardiovascular",
            drift_domain="cardiovascular",
            action="targeted_scale_probe",
            reason_code="route_lipid_cardiovascular",
            priority=20,
            scale_ids=("scale_cm_ez_cvd",),
            domain_ids=("cardiovascular", "lipid", "inflammation", "cardiometabolic"),
            triggered_instrument="scale_cm_ez_cvd",
        ),
        DriftRoute(
            route_id="route_kidney_creatinine",
            drift_domain="kidney",
            action="targeted_scale_probe",
            reason_code="route_kidney_creatinine",
            priority=30,
            scale_ids=("scale_cm_scored",),
            domain_ids=("kidney", "renal", "creatinine", "cardiometabolic"),
            triggered_instrument="scale_cm_scored",
        ),
        DriftRoute(
            route_id="route_liver_nafld",
            drift_domain="liver",
            action="targeted_scale_probe",
            reason_code="route_liver_nafld",
            priority=40,
            scale_ids=("scale_cm_lee_nafld",),
            domain_ids=("liver", "hepatic", "nafld", "cardiometabolic"),
            triggered_instrument="scale_cm_lee_nafld",
        ),
        DriftRoute(
            route_id="route_activity_lifestyle",
            drift_domain="activity",
            action="targeted_scale_probe",
            reason_code="route_activity_lifestyle",
            priority=50,
            scale_ids=("scale_cm_ipaq_sf",),
            domain_ids=("activity", "lifestyle", "exercise", "cardiometabolic"),
            triggered_instrument="scale_cm_ipaq_sf",
        ),
        DriftRoute(
            route_id="route_behavioral_alcohol",
            drift_domain="behavioral",
            action="targeted_scale_probe",
            reason_code="route_behavioral_alcohol",
            priority=60,
            scale_ids=("scale_cm_audit_c",),
            domain_ids=("behavioral", "alcohol", "cardiometabolic"),
            triggered_instrument="scale_cm_audit_c",
        ),
        DriftRoute(
            route_id="route_general_rescan",
            drift_domain="general",
            action="full_27_rescan",
            reason_code="route_general_rescan",
            priority=999,
            domain_ids=("general",),
        ),
    ]


def load_drift_routes(path: Optional[str]) -> Tuple[str, List[DriftRoute]]:
    raw_path = str(path or "").strip()
    if not raw_path:
        return DEFAULT_DRIFT_ROUTING_VERSION, default_drift_routes()
    p = Path(raw_path)
    if not p.exists():
        raise ValueError(f"drift routing table path does not exist: {raw_path}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    return parse_drift_route_payload(payload)


def parse_drift_route_payload(payload: Any) -> Tuple[str, List[DriftRoute]]:
    if isinstance(payload, list):
        version = DEFAULT_DRIFT_ROUTING_VERSION
        rows = payload
    elif isinstance(payload, dict):
        version = str(payload.get("version") or DEFAULT_DRIFT_ROUTING_VERSION)
        rows = payload.get("routes")
    else:
        raise ValueError("drift route payload must be an object or list")
    if not isinstance(rows, list) or not rows:
        raise ValueError("drift route payload must include non-empty routes[]")
    routes = [_parse_route_obj(obj) for obj in rows]
    return version, sorted(routes, key=lambda r: (int(r.priority), str(r.route_id)))


def route_contract(version: str, routes: Sequence[DriftRoute]) -> Dict[str, Any]:
    return {
        "version": str(version),
        "routes": [asdict(r) for r in routes],
    }


def resolve_drift_route(
    *,
    latest_scores: Mapping[str, Mapping[str, Any]],
    scale_domains: Mapping[str, Set[str]],
    routes: Sequence[DriftRoute],
) -> Dict[str, Any]:
    ordered_routes = sorted(list(routes), key=lambda r: (int(r.priority), str(r.route_id)))
    ranked_scales = sorted(
        [str(scale_id) for scale_id in latest_scores.keys()],
        key=lambda sid: _score_strength(latest_scores.get(sid) or {}),
        reverse=True,
    )
    for scale_id in ranked_scales:
        domains = {
            canonical_domain_id(d)
            for d in (scale_domains.get(scale_id) or {"general"})
            if str(d or "").strip()
        }
        for route in ordered_routes:
            if _route_matches(route, scale_id=scale_id, domains=domains):
                return {
                    "route_id": route.route_id,
                    "drift_domain": route.drift_domain,
                    "action": route.action,
                    "reason_code": route.reason_code,
                    "triggered_instrument": route.triggered_instrument or scale_id,
                    "matched_scale_id": scale_id,
                }

    fallback = _fallback_route(ordered_routes)
    return {
        "route_id": fallback.route_id,
        "drift_domain": fallback.drift_domain,
        "action": fallback.action,
        "reason_code": fallback.reason_code,
        "triggered_instrument": fallback.triggered_instrument,
        "matched_scale_id": None,
    }


def _fallback_route(routes: Sequence[DriftRoute]) -> DriftRoute:
    for route in routes:
        if str(route.action) == "full_27_rescan":
            return route
    if routes:
        return routes[-1]
    return DriftRoute(
        route_id="route_default_fallback",
        drift_domain="general",
        action="full_27_rescan",
        reason_code="route_default_fallback",
        priority=9999,
        domain_ids=("general",),
    )


def _parse_route_obj(obj: Any) -> DriftRoute:
    if not isinstance(obj, dict):
        raise ValueError("route row must be an object")
    route_id = str(obj.get("route_id") or "").strip()
    if not route_id:
        raise ValueError("route.route_id is required")
    drift_domain = canonical_domain_id(obj.get("drift_domain"))
    action = str(obj.get("action") or "").strip() or "targeted_scale_probe"
    if action not in {"targeted_scale_probe", "full_27_rescan"}:
        raise ValueError(f"invalid route.action: {action}")
    reason_code = str(obj.get("reason_code") or route_id).strip()
    priority = int(obj.get("priority") if obj.get("priority") is not None else 100)
    scale_ids = tuple(sorted({str(x).strip() for x in (obj.get("scale_ids") or []) if str(x).strip()}))
    domain_ids = tuple(
        sorted(
            {
                canonical_domain_id(str(x))
                for x in (obj.get("domain_ids") or [])
                if str(x or "").strip()
            }
        )
    )
    trig = obj.get("triggered_instrument")
    triggered_instrument = str(trig).strip() if isinstance(trig, str) and trig.strip() else None
    return DriftRoute(
        route_id=route_id,
        drift_domain=drift_domain,
        action=action,
        reason_code=reason_code,
        priority=priority,
        scale_ids=scale_ids,
        domain_ids=domain_ids,
        triggered_instrument=triggered_instrument,
    )


def _route_matches(route: DriftRoute, *, scale_id: str, domains: Set[str]) -> bool:
    if route.scale_ids and str(scale_id) in set(route.scale_ids):
        return True
    if route.domain_ids and (set(route.domain_ids) & set(domains)):
        return True
    return False


def _score_strength(v: Mapping[str, Any]) -> float:
    try:
        personal_z = abs(float(v.get("personal_z") or 0.0))
    except Exception:
        personal_z = 0.0
    try:
        delta = abs(float(v.get("delta_vs_prev") or 0.0))
    except Exception:
        delta = 0.0
    return max(personal_z, delta)

