import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _boolish(value: Any, *, default: bool) -> bool:
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


def _safe_int(value: Any, *, default: int) -> int:
    try:
        if value in (None, ""):
            return default
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def _safe_float(value: Any, *, default: float) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class QuestionsAgentConfig:
    database_path: str
    registry_root: str
    host: str = "127.0.0.1"
    port: int = 8088
    dashboard_enabled: bool = True
    timezone: str = "UTC"

    # Daily UX knobs
    core_questions_per_day: int = 5
    extra_batch_size: int = 5
    extra_batches_max_per_day: int = 3

    # Selection / scoring knobs
    item_repeat_cooldown_days: int = 7
    default_unlock_window_days: int = 14
    default_retest_interval_days: int = 90
    max_active_new_domains: int = 2
    safety_event_lookback_days: int = 30
    safety_min_questions_override: int = 3
    emotion_deep_dive_item_count: int = 10

    # Progressive measurement packages
    # package_focus_boost controls how strongly the selection engine
    # prioritises items from the currently active measurement package.
    # 5.0 is high enough to outweigh generic multiplex or retest
    # signals (3.0), but lower than onboarding priority (50-400) and
    # comparable to near-unlock (4.0) so clinical signals still win.
    package_focus_boost: float = 5.0

    # Policy layer (constrained contextual bandit) - optional
    policy_root: str = "questions_agent_platform/data/policy_demo"
    policy_default_version: str = "v1"
    policy_default_mode: str = "policy_shadow"  # deterministic | policy_live | policy_shadow | policy_offline_replay
    policy_epsilon_explore: float = 0.05
    policy_log_context_snapshot: bool = True
    policy_log_candidate_set_snapshot: bool = True
    policy_store_z_snapshot: bool = False
    policy_store_z_delta: bool = True
    policy_quantize_decimals: int = 3
    policy_auto_rollback_enabled: bool = True
    policy_rollback_window_days: int = 14
    policy_rollback_min_outcomes: int = 40
    policy_rollback_max_safety_violations: int = 0
    policy_rollback_min_completion_drop: float = 0.03
    policy_rollback_burden_increase_ratio: float = 1.10

    # Drift routing protocol table (optional override; defaults to built-in table)
    drift_routing_table_path: Optional[str] = None

    # Site configuration binding (consumer | cds_sheba | samd_full | trial)
    site_config_id: str = "consumer"


def load_config(path: str) -> QuestionsAgentConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return QuestionsAgentConfig(
        database_path=_require_str(raw, "database_path"),
        registry_root=_require_str(raw, "registry_root"),
        host=str(raw.get("host", "127.0.0.1")),
        port=_safe_int(raw.get("port", 8088), default=8088),
        dashboard_enabled=_boolish(raw.get("dashboard_enabled", True), default=True),
        timezone=str(raw.get("timezone", "UTC")),
        core_questions_per_day=_safe_int(raw.get("core_questions_per_day", 5), default=5),
        extra_batch_size=_safe_int(raw.get("extra_batch_size", 5), default=5),
        extra_batches_max_per_day=_safe_int(raw.get("extra_batches_max_per_day", 3), default=3),
        item_repeat_cooldown_days=_safe_int(raw.get("item_repeat_cooldown_days", 7), default=7),
        default_unlock_window_days=_safe_int(raw.get("default_unlock_window_days", 14), default=14),
        default_retest_interval_days=_safe_int(raw.get("default_retest_interval_days", 90), default=90),
        max_active_new_domains=_safe_int(raw.get("max_active_new_domains", 2), default=2),
        safety_event_lookback_days=_safe_int(raw.get("safety_event_lookback_days", 30), default=30),
        safety_min_questions_override=_safe_int(raw.get("safety_min_questions_override", 3), default=3),
        emotion_deep_dive_item_count=_safe_int(raw.get("emotion_deep_dive_item_count", 10), default=10),
        package_focus_boost=_safe_float(raw.get("package_focus_boost", 5.0), default=5.0),
        policy_root=str(raw.get("policy_root", "questions_agent_platform/data/policy_demo")),
        policy_default_version=str(raw.get("policy_default_version", "v1")),
        policy_default_mode=str(raw.get("policy_default_mode", "policy_shadow")),
        policy_epsilon_explore=_safe_float(raw.get("policy_epsilon_explore", 0.05), default=0.05),
        policy_log_context_snapshot=_boolish(raw.get("policy_log_context_snapshot", True), default=True),
        policy_log_candidate_set_snapshot=_boolish(raw.get("policy_log_candidate_set_snapshot", True), default=True),
        policy_store_z_snapshot=_boolish(raw.get("policy_store_z_snapshot", False), default=False),
        policy_store_z_delta=_boolish(raw.get("policy_store_z_delta", True), default=True),
        policy_quantize_decimals=_safe_int(raw.get("policy_quantize_decimals", 3), default=3),
        policy_auto_rollback_enabled=_boolish(raw.get("policy_auto_rollback_enabled", True), default=True),
        policy_rollback_window_days=_safe_int(raw.get("policy_rollback_window_days", 14), default=14),
        policy_rollback_min_outcomes=_safe_int(raw.get("policy_rollback_min_outcomes", 40), default=40),
        policy_rollback_max_safety_violations=_safe_int(raw.get("policy_rollback_max_safety_violations", 0), default=0),
        policy_rollback_min_completion_drop=_safe_float(raw.get("policy_rollback_min_completion_drop", 0.03), default=0.03),
        policy_rollback_burden_increase_ratio=_safe_float(raw.get("policy_rollback_burden_increase_ratio", 1.10), default=1.10),
        drift_routing_table_path=(
            str(raw.get("drift_routing_table_path")).strip()
            if raw.get("drift_routing_table_path") is not None and str(raw.get("drift_routing_table_path")).strip()
            else None
        ),
        site_config_id=str(raw.get("site_config_id", "consumer")).strip() or "consumer",
    )


def ensure_paths(cfg: QuestionsAgentConfig) -> None:
    Path(cfg.registry_root).mkdir(parents=True, exist_ok=True)
    db_path = Path(cfg.database_path)
    if db_path.parent and str(db_path.parent) not in ("", "."):
        db_path.parent.mkdir(parents=True, exist_ok=True)


def _require_str(obj: Dict[str, Any], key: str) -> str:
    val = obj.get(key)
    if not isinstance(val, str) or not val.strip():
        raise ValueError(f"Missing required config key: {key}")
    return val
