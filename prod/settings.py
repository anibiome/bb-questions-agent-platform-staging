from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    database_url: str
    registry_root: str
    api_key: str
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "info"
    db_migration_mode: str = "require"  # auto | require | off
    db_migration_baseline_existing: bool = True

    # Site configuration
    site_config_default: str = "consumer"

    # Selection knobs
    core_questions_per_day: int = 5
    extra_batch_size: int = 5
    extra_batches_max_per_day: int = 3
    item_repeat_cooldown_days: int = 7
    max_active_new_domains: int = 2
    safety_event_lookback_days: int = 30
    safety_min_questions_override: int = 3
    emotion_deep_dive_item_count: int = 10

    # Policy layer (constrained contextual bandit) - optional
    policy_root: str = "/app/policy"
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
    drift_routing_table_path: str = ""

    # Operational SLOs
    slo_window_days: int = 14
    slo_safe_fallback_rate_max: float = 0.10
    slo_answer_ingestion_p95_ms_max: float = 750.0
    slo_snapshot_write_failure_rate_max: float = 0.01

    # PHI controls
    phi_retention_answer_raw_days: int = 365
    phi_retention_observation_raw_days: int = 365
    phi_retention_policy_context_days: int = 180
    phi_export_redact_user_ids: bool = True
    phi_export_redact_raw_payloads: bool = True
    phi_export_user_hash_salt: str = ""


def load_settings() -> Settings:
    database_url = _require("DATABASE_URL")
    registry_root = _require("REGISTRY_ROOT")
    api_key = _require("API_KEY")
    host = os.getenv("HOST", "0.0.0.0")
    port = _int_env("PORT", 8080)
    log_level = os.getenv("LOG_LEVEL", "info")

    migration_mode = os.getenv("DB_MIGRATION_MODE", "require").strip().lower()
    if migration_mode not in ("auto", "require", "off"):
        raise RuntimeError("DB_MIGRATION_MODE must be one of: auto, require, off")

    return Settings(
        database_url=database_url,
        registry_root=registry_root,
        api_key=api_key,
        host=host,
        port=port,
        log_level=log_level,
        db_migration_mode=migration_mode,
        db_migration_baseline_existing=_bool(os.getenv("DB_MIGRATION_BASELINE_EXISTING", "true")),
        site_config_default=os.getenv("SITE_CONFIG_DEFAULT", "consumer"),
        core_questions_per_day=_int_env("CORE_QUESTIONS_PER_DAY", 5),
        extra_batch_size=_int_env("EXTRA_BATCH_SIZE", 5),
        extra_batches_max_per_day=_int_env("EXTRA_BATCHES_MAX_PER_DAY", 3),
        item_repeat_cooldown_days=_int_env("ITEM_REPEAT_COOLDOWN_DAYS", 7),
        max_active_new_domains=_int_env("MAX_ACTIVE_NEW_DOMAINS", 2),
        safety_event_lookback_days=_int_env("SAFETY_EVENT_LOOKBACK_DAYS", 30),
        safety_min_questions_override=_int_env("SAFETY_MIN_QUESTIONS_OVERRIDE", 3),
        emotion_deep_dive_item_count=_int_env("EMOTION_DEEP_DIVE_ITEM_COUNT", 10),
        policy_root=os.getenv("POLICY_ROOT", "/app/policy"),
        policy_default_version=os.getenv("POLICY_DEFAULT_VERSION", "v1"),
        policy_default_mode=os.getenv("POLICY_DEFAULT_MODE", "policy_shadow"),
        policy_epsilon_explore=_float_env("POLICY_EPSILON_EXPLORE", 0.05),
        policy_log_context_snapshot=_bool(os.getenv("POLICY_LOG_CONTEXT_SNAPSHOT", "true")),
        policy_log_candidate_set_snapshot=_bool(os.getenv("POLICY_LOG_CANDIDATE_SET_SNAPSHOT", "true")),
        policy_store_z_snapshot=_bool(os.getenv("POLICY_STORE_Z_SNAPSHOT", "false")),
        policy_store_z_delta=_bool(os.getenv("POLICY_STORE_Z_DELTA", "true")),
        policy_quantize_decimals=_int_env("POLICY_QUANTIZE_DECIMALS", 3),
        policy_auto_rollback_enabled=_bool(os.getenv("POLICY_AUTO_ROLLBACK_ENABLED", "true")),
        policy_rollback_window_days=_int_env("POLICY_ROLLBACK_WINDOW_DAYS", 14),
        policy_rollback_min_outcomes=_int_env("POLICY_ROLLBACK_MIN_OUTCOMES", 40),
        policy_rollback_max_safety_violations=_int_env("POLICY_ROLLBACK_MAX_SAFETY_VIOLATIONS", 0),
        policy_rollback_min_completion_drop=_float_env("POLICY_ROLLBACK_MIN_COMPLETION_DROP", 0.03),
        policy_rollback_burden_increase_ratio=_float_env("POLICY_ROLLBACK_BURDEN_INCREASE_RATIO", 1.10),
        drift_routing_table_path=str(os.getenv("DRIFT_ROUTING_TABLE_PATH", "")).strip(),
        slo_window_days=_int_env("SLO_WINDOW_DAYS", 14),
        slo_safe_fallback_rate_max=_float_env("SLO_SAFE_FALLBACK_RATE_MAX", 0.10),
        slo_answer_ingestion_p95_ms_max=_float_env("SLO_ANSWER_INGESTION_P95_MS_MAX", 750.0),
        slo_snapshot_write_failure_rate_max=_float_env("SLO_SNAPSHOT_WRITE_FAILURE_RATE_MAX", 0.01),
        phi_retention_answer_raw_days=_int_env("PHI_RETENTION_ANSWER_RAW_DAYS", 365),
        phi_retention_observation_raw_days=_int_env("PHI_RETENTION_OBSERVATION_RAW_DAYS", 365),
        phi_retention_policy_context_days=_int_env("PHI_RETENTION_POLICY_CONTEXT_DAYS", 180),
        phi_export_redact_user_ids=_bool(os.getenv("PHI_EXPORT_REDACT_USER_IDS", "true")),
        phi_export_redact_raw_payloads=_bool(os.getenv("PHI_EXPORT_REDACT_RAW_PAYLOADS", "true")),
        phi_export_user_hash_salt=str(os.getenv("PHI_EXPORT_USER_HASH_SALT", "")),
    )


def _require(key: str) -> str:
    val = os.getenv(key)
    if not val or not val.strip():
        raise RuntimeError(f"Missing required env var: {key}")
    return val.strip()


def _int_env(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{key} must be numeric") from exc


def _float_env(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{key} must be numeric") from exc


def _bool(v: str) -> bool:
    s = str(v or "").strip().lower()
    return s in ("1", "true", "yes", "y", "on")
