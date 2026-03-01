from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from questions_agent_platform.pipeline.registry import Registry


CARDIOMETABOLIC_DOMAIN = "cardiometabolic"
DOMAIN_ALIAS_TO_CARDIO = {
    "cardio",
    "cardiometabolic",
    "metabolic",
    "glucose",
    "cardiovascular",
    "kidney",
    "liver",
}
EMOTION_TAGS = {"mood", "anxiety", "depression", "stress", "affect", "emotion", "sleep", "energy"}


def canonical_domain_id(value: Optional[str]) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return "general"
    if raw in DOMAIN_ALIAS_TO_CARDIO:
        return CARDIOMETABOLIC_DOMAIN
    return raw.replace(" ", "_")


def parse_json_str_list(value: Any, *, fallback: Optional[Sequence[str]] = None) -> List[str]:
    if isinstance(value, list):
        raw = value
    else:
        try:
            raw = json.loads(str(value))
        except Exception:
            raw = list(fallback or [])
    out: List[str] = []
    for item in raw:
        if item is None:
            continue
        domain = canonical_domain_id(str(item))
        if domain not in out:
            out.append(domain)
    return out


def encode_json_str_list(items: Iterable[str]) -> str:
    normalized = sorted({canonical_domain_id(x) for x in items if str(x or "").strip()})
    return json.dumps(normalized, ensure_ascii=False)


def registry_scale_domains(registry: Registry) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for scale in registry.scales.values():
        domains: Set[str] = set()
        questionnaire = registry.questionnaires.get(scale.questionnaire_id)
        if questionnaire:
            domains.update(canonical_domain_id(x) for x in questionnaire.domains if str(x or "").strip())
        domains.update(canonical_domain_id(x) for x in scale.tags if str(x or "").strip())
        if not domains:
            domains = {"general"}
        out[scale.id] = domains
    return out


def registry_item_domains(registry: Registry) -> Dict[str, Set[str]]:
    scale_domains = registry_scale_domains(registry)
    out: Dict[str, Set[str]] = {item_id: set() for item_id in registry.items.keys()}
    for scale in registry.scales.values():
        domains = scale_domains.get(scale.id, {"general"})
        for scale_item in scale.items:
            if scale_item.item_id in out:
                out[scale_item.item_id].update(domains)
    for item_id, item in registry.items.items():
        if not out.get(item_id):
            inferred = {canonical_domain_id(tag) for tag in item.tags if str(tag or "").strip()}
            out[item_id] = inferred or {"general"}
    for item_id, domains in out.items():
        if any(d in DOMAIN_ALIAS_TO_CARDIO for d in domains):
            out[item_id].add(CARDIOMETABOLIC_DOMAIN)
        item_tags = set(registry.items[item_id].tags)
        if "metabolic" in item_tags or "cardiometabolic" in item_tags:
            out[item_id].add(CARDIOMETABOLIC_DOMAIN)
    return out


def registry_domains(registry: Registry) -> List[str]:
    domains: Set[str] = {CARDIOMETABOLIC_DOMAIN}
    for scale_domain_set in registry_scale_domains(registry).values():
        domains.update(scale_domain_set)
    for item in registry.items.values():
        domains.update(canonical_domain_id(tag) for tag in item.tags if str(tag or "").strip())
    domains = {d for d in domains if d}
    return sorted(domains)


def sync_user_domain_state(
    *,
    mode: str,
    onboarding_complete: bool,
    active_domains: Sequence[str],
    queued_domains: Sequence[str],
    promoted_domains: Sequence[str],
    registry_domains_all: Sequence[str],
    max_active_new_domains: int,
) -> Tuple[List[str], List[str], List[str], Set[str]]:
    mode_norm = str(mode or "consumer").strip().lower()
    max_new = max(0, int(max_active_new_domains))
    all_domains = [canonical_domain_id(d) for d in registry_domains_all if str(d or "").strip()]
    all_set = set(all_domains)

    active = [d for d in parse_json_str_list(list(active_domains), fallback=[CARDIOMETABOLIC_DOMAIN]) if d in all_set]
    queued = [d for d in parse_json_str_list(list(queued_domains)) if d in all_set]
    promoted = [d for d in parse_json_str_list(list(promoted_domains), fallback=[CARDIOMETABOLIC_DOMAIN]) if d in all_set]

    if CARDIOMETABOLIC_DOMAIN not in active:
        active.insert(0, CARDIOMETABOLIC_DOMAIN)
    if CARDIOMETABOLIC_DOMAIN not in promoted:
        promoted.insert(0, CARDIOMETABOLIC_DOMAIN)

    for d in all_domains:
        if d in active or d in queued:
            continue
        queued.append(d)

    if mode_norm == "trial" or not onboarding_complete:
        active = [CARDIOMETABOLIC_DOMAIN]
        queued = [d for d in queued if d != CARDIOMETABOLIC_DOMAIN]
        promoted = sorted(set(promoted + [CARDIOMETABOLIC_DOMAIN]))
        return active, queued, promoted, {CARDIOMETABOLIC_DOMAIN}

    non_cardio_active = [d for d in active if d != CARDIOMETABOLIC_DOMAIN]
    if len(non_cardio_active) > max_new:
        keep = non_cardio_active[:max_new]
        demoted = [d for d in non_cardio_active if d not in keep]
        queued = sorted(set(queued + demoted))
        non_cardio_active = keep

    if len(non_cardio_active) < max_new:
        for domain in list(queued):
            if domain == CARDIOMETABOLIC_DOMAIN:
                continue
            if domain in non_cardio_active:
                continue
            non_cardio_active.append(domain)
            queued.remove(domain)
            if len(non_cardio_active) >= max_new:
                break

    active = [CARDIOMETABOLIC_DOMAIN] + sorted(set(non_cardio_active))
    queued = [d for d in queued if d not in active]
    allowed_domains = set(active)
    return active, queued, sorted(set(promoted)), allowed_domains


def item_matches_domains(item_domains: Dict[str, Set[str]], item_id: str, allowed_domains: Set[str]) -> bool:
    if not allowed_domains:
        return True
    domains = item_domains.get(item_id, {"general"})
    return bool(set(domains) & set(allowed_domains))
