import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ALLOWED_SCORING_METHODS = {
    "sum",
    "mean",
    "instrument_findrisc",
    "instrument_ez_cvd",
    "instrument_scored",
    "instrument_lee_nafld",
    "instrument_ipaq_sf",
    "instrument_audit_c",
    # Vitality battery (v4)
    "instrument_who5",
    "instrument_sf36_vt",
    "instrument_promis_fatigue_7a",
}


@dataclass(frozen=True)
class Item:
    id: str
    text: str
    response_type: str
    tags: Tuple[str, ...] = ()
    sensitivity: str = "low"
    timeframes_allowed: Tuple[str, ...] = ("last_7_days",)
    intrusiveness: str = "low"
    declinable: bool = True
    onboarding_order: Optional[int] = None


@dataclass(frozen=True)
class ScaleItem:
    item_id: str
    reverse: bool = False
    weight: float = 1.0


@dataclass(frozen=True)
class Scale:
    id: str
    questionnaire_id: str
    version: str
    name: str
    method: str  # sum | mean
    min_items_required: int
    unlock_window_days: int
    retest_interval_days: int
    response_type: str
    normalize_min: float
    normalize_max: float
    items: Tuple[ScaleItem, ...]
    tags: Tuple[str, ...] = ()
    ewma_alpha: float = 0.2  # per-scale EWMA smoothing; higher = more reactive
    citation: str = ""       # source instrument citation / DOI


@dataclass(frozen=True)
class Questionnaire:
    id: str
    version: str
    name: str
    domains: Tuple[str, ...] = ()
    license: str = "internal_demo"
    source: Optional[str] = None


@dataclass(frozen=True)
class Registry:
    version: str
    items: Dict[str, Item]
    scales: Dict[str, Scale]
    questionnaires: Dict[str, Questionnaire]


def load_registry(registry_root: str, active_version: str) -> Registry:
    version_dir = Path(registry_root) / "versions" / active_version
    items_path = version_dir / "items.json"
    scales_path = version_dir / "scales.json"
    questionnaires_path = version_dir / "questionnaires.json"

    items_raw = _read_json(items_path)
    scales_raw = _read_json(scales_path)
    questionnaires_raw = _read_json(questionnaires_path)

    items = _parse_items(items_raw)
    questionnaires = _parse_questionnaires(questionnaires_raw)
    scales = _parse_scales(scales_raw)

    _validate_registry(items=items, scales=scales, questionnaires=questionnaires)
    return Registry(
        version=active_version,
        items={i.id: i for i in items},
        scales={s.id: s for s in scales},
        questionnaires={q.id: q for q in questionnaires},
    )


def validate_registry_bundle(bundle: Dict[str, Any]) -> Registry:
    version = _require_str(bundle, "version")
    items = _parse_items(_require_list(bundle, "items"))
    questionnaires = _parse_questionnaires(_require_list(bundle, "questionnaires"))
    scales = _parse_scales(_require_list(bundle, "scales"))
    _validate_registry(items=items, scales=scales, questionnaires=questionnaires)
    return Registry(
        version=version,
        items={i.id: i for i in items},
        scales={s.id: s for s in scales},
        questionnaires={q.id: q for q in questionnaires},
    )


def save_registry(registry_root: str, registry: Registry) -> None:
    version_dir = Path(registry_root) / "versions" / registry.version
    version_dir.mkdir(parents=True, exist_ok=True)
    _write_json(version_dir / "items.json", [_item_to_json(i) for i in registry.items.values()])
    _write_json(
        version_dir / "questionnaires.json",
        [_questionnaire_to_json(q) for q in registry.questionnaires.values()],
    )
    _write_json(version_dir / "scales.json", [_scale_to_json(s) for s in registry.scales.values()])


def list_versions(registry_root: str) -> List[str]:
    versions_dir = Path(registry_root) / "versions"
    if not versions_dir.exists():
        return []
    return sorted([p.name for p in versions_dir.iterdir() if p.is_dir()])


def _parse_items(raw: Any) -> List[Item]:
    if not isinstance(raw, list):
        raise ValueError("items.json must be a list")
    out: List[Item] = []
    for obj in raw:
        if not isinstance(obj, dict):
            raise ValueError("Each item must be an object")
        out.append(
            Item(
                id=_require_str(obj, "id"),
                text=_require_str(obj, "text"),
                response_type=_require_str(obj, "response_type"),
                tags=tuple(_optional_str_list(obj.get("tags"))),
                sensitivity=str(obj.get("sensitivity", "low")),
                timeframes_allowed=tuple(
                    _optional_str_list(obj.get("timeframes_allowed")) or ["last_7_days"]
                ),
                intrusiveness=str(obj.get("intrusiveness", "low")),
                declinable=bool(obj.get("declinable", True)),
                onboarding_order=_optional_int(obj.get("onboarding_order")),
            )
        )
    return out


def _parse_questionnaires(raw: Any) -> List[Questionnaire]:
    if not isinstance(raw, list):
        raise ValueError("questionnaires.json must be a list")
    out: List[Questionnaire] = []
    for obj in raw:
        if not isinstance(obj, dict):
            raise ValueError("Each questionnaire must be an object")
        out.append(
            Questionnaire(
                id=_require_str(obj, "id"),
                version=_require_str(obj, "version"),
                name=_require_str(obj, "name"),
                domains=tuple(_optional_str_list(obj.get("domains"))),
                license=str(obj.get("license", "internal_demo")),
                source=_optional_str(obj.get("source")),
            )
        )
    return out


def _parse_scales(raw: Any) -> List[Scale]:
    if not isinstance(raw, list):
        raise ValueError("scales.json must be a list")
    out: List[Scale] = []
    for obj in raw:
        if not isinstance(obj, dict):
            raise ValueError("Each scale must be an object")
        scoring = obj.get("scoring", {})
        if not isinstance(scoring, dict):
            raise ValueError("scale.scoring must be an object")
        items_raw = obj.get("items")
        if not isinstance(items_raw, list) or not items_raw:
            raise ValueError("scale.items must be a non-empty list")
        items: List[ScaleItem] = []
        for it in items_raw:
            if not isinstance(it, dict):
                raise ValueError("scale.items[] must be an object")
            items.append(
                ScaleItem(
                    item_id=_require_str(it, "item_id"),
                    reverse=bool(it.get("reverse", False)),
                    weight=float(it.get("weight", 1.0)),
                )
            )

        out.append(
            Scale(
                id=_require_str(obj, "id"),
                questionnaire_id=_require_str(obj, "questionnaire_id"),
                version=_require_str(obj, "version"),
                name=_require_str(obj, "name"),
                method=str(scoring.get("method", "sum")),
                min_items_required=int(obj.get("min_items_required") or len(items)),
                unlock_window_days=_int_with_default(obj.get("unlock_window_days"), default=14),
                retest_interval_days=_int_with_default(obj.get("retest_interval_days"), default=90),
                response_type=_require_str(obj, "response_type"),
                normalize_min=float(scoring.get("normalize_min", 0.0)),
                normalize_max=float(scoring.get("normalize_max", 100.0)),
                items=tuple(items),
                tags=tuple(_optional_str_list(obj.get("tags"))),
                ewma_alpha=float(obj.get("ewma_alpha", 0.2)),
                citation=str(obj.get("citation", "")),
            )
        )
    return out


def _validate_registry(
    *,
    items: List[Item],
    scales: List[Scale],
    questionnaires: List[Questionnaire],
) -> None:
    item_ids = {i.id for i in items}
    if len(item_ids) != len(items):
        raise ValueError("Duplicate item ids in registry")

    questionnaire_ids = {q.id for q in questionnaires}
    if len(questionnaire_ids) != len(questionnaires):
        raise ValueError("Duplicate questionnaire ids in registry")

    scale_ids = {s.id for s in scales}
    if len(scale_ids) != len(scales):
        raise ValueError("Duplicate scale ids in registry")

    for s in scales:
        if s.questionnaire_id not in questionnaire_ids:
            raise ValueError(f"Scale {s.id} references unknown questionnaire {s.questionnaire_id}")
        if s.method not in ALLOWED_SCORING_METHODS:
            raise ValueError(f"Scale {s.id} has invalid scoring method: {s.method}")
        if s.min_items_required < 1 or s.min_items_required > len(s.items):
            raise ValueError(f"Scale {s.id} min_items_required is out of range")
        for si in s.items:
            if si.item_id not in item_ids:
                raise ValueError(f"Scale {s.id} references unknown item {si.item_id}")


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def _item_to_json(item: Item) -> Dict[str, Any]:
    return {
        "id": item.id,
        "text": item.text,
        "response_type": item.response_type,
        "tags": list(item.tags),
        "sensitivity": item.sensitivity,
        "timeframes_allowed": list(item.timeframes_allowed),
        "intrusiveness": item.intrusiveness,
        "declinable": bool(item.declinable),
        "onboarding_order": item.onboarding_order,
    }


def _questionnaire_to_json(q: Questionnaire) -> Dict[str, Any]:
    return {
        "id": q.id,
        "version": q.version,
        "name": q.name,
        "domains": list(q.domains),
        "license": q.license,
        "source": q.source,
    }


def _scale_to_json(s: Scale) -> Dict[str, Any]:
    return {
        "id": s.id,
        "questionnaire_id": s.questionnaire_id,
        "version": s.version,
        "name": s.name,
        "response_type": s.response_type,
        "min_items_required": s.min_items_required,
        "unlock_window_days": s.unlock_window_days,
        "retest_interval_days": s.retest_interval_days,
        "tags": list(s.tags),
        "scoring": {
            "method": s.method,
            "normalize_min": s.normalize_min,
            "normalize_max": s.normalize_max,
        },
        "items": [
            {"item_id": it.item_id, "reverse": it.reverse, "weight": it.weight} for it in s.items
        ],
        "ewma_alpha": s.ewma_alpha,
        "citation": s.citation,
    }


def _optional_str(val: Any) -> Optional[str]:
    if val is None:
        return None
    if not isinstance(val, str):
        raise ValueError("Expected string or null")
    s = val.strip()
    return s or None


def _optional_int(val: Any) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(val)
    except Exception as e:
        raise ValueError("Expected int or null") from e


def _int_with_default(val: Any, *, default: int) -> int:
    if val in (None, ""):
        return int(default)
    try:
        return int(val)
    except Exception as e:
        raise ValueError("Expected int-compatible value") from e


def _optional_str_list(val: Any) -> List[str]:
    if val is None:
        return []
    if not isinstance(val, list):
        raise ValueError("Expected list of strings")
    out: List[str] = []
    for v in val:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Expected list of non-empty strings")
        out.append(v.strip())
    return out


def _require_str(obj: Dict[str, Any], key: str) -> str:
    val = obj.get(key)
    if not isinstance(val, str) or not val.strip():
        raise ValueError(f"Missing required field: {key}")
    return val.strip()


def _require_list(obj: Dict[str, Any], key: str) -> List[Any]:
    val = obj.get(key)
    if not isinstance(val, list):
        raise ValueError(f"Missing required list: {key}")
    return val
