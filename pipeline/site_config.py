"""
Site Configuration Registry for Multi-Version Deployment.

Supports multiple deployment configurations (app versions):
  - Consumer wellness app (ani.ai) → Config A: wellness language, no disease claims
  - Clinician CDS dashboard → Config B: transparent scoring, clinical thresholds
  - Site-specific sub-versions → Config B variants with locale/instrument overrides
  - Full AI SaMD → Config C: AniFold integration, AI recommendations

CRITICAL REGULATORY REQUIREMENT:
  Config B (CDS) and Config C (SaMD) MUST be strictly separated.
  No AniFold latent inference may touch clinician-facing outputs in Config B.
  This is enforced by the `anifold_enabled` flag — when False, no AniFold
  integration code is reachable.

Architecture:
  base/           ← shared items, scales, IRT params (the science)
  configs/
    consumer/     ← Config A: wellness language, no thresholds, no disease names
    site_sheba/   ← Config B: CDS dashboard, transparent scoring, Hebrew locale
    site_trial/   ← Config B+: concordance study mode, paired measurements
    samd_full/    ← Config C: AniFold integration, AI recommendations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SiteConfig:
    """Configuration for one deployment site/version."""

    # Identity
    config_id: str                                 # e.g. "consumer", "site_sheba", "samd_full"
    display_name: str                              # e.g. "ani.ai Consumer", "Sheba Medical CDS"
    config_type: str                               # "wellness" | "cds" | "samd" | "trial"

    # Registry version
    registry_version: str = "v2"                   # which registry bundle to use
    locale: str = "en"                             # language/locale for item text

    # Instrument configuration
    active_scale_ids: FrozenSet[str] = frozenset() # which scales are enabled (empty = all)
    disabled_scale_ids: FrozenSet[str] = frozenset()  # explicitly disabled scales

    # Display mode
    show_clinical_thresholds: bool = False         # Config A: False, Config B: True
    show_disease_names: bool = False               # Config A: False, Config B: True
    show_risk_tiers: bool = True                   # show risk tier labels
    score_display_mode: str = "wellness"           # "wellness" | "clinical" | "research"
    wellness_language: bool = True                  # use wellness-safe language

    # AniFold integration (REGULATORY: must be False for Config B)
    anifold_enabled: bool = False                  # Config C only!
    anifold_endpoint: Optional[str] = None         # AniFold API URL

    # Concordance study mode
    concordance_mode: bool = False                 # enables paired measurement collection
    concordance_alternating_weeks: bool = False     # alternating adaptive/full-form weeks

    # Feature flags
    behavioural_metadata_enabled: bool = True      # capture response timing/edits
    anamnesis_enabled: bool = True                  # drift-triggered branching
    policy_mode: str = "deterministic"             # "deterministic" | "policy_live" | "policy_shadow"

    # Operational
    daily_question_budget: int = 5
    max_extra_batches: int = 3
    item_repeat_cooldown_days: int = 7

    # Branding
    brand_name: str = "ani"
    brand_color: str = "#4A90D9"


# ---------------------------------------------------------------------------
# Predefined configurations
# ---------------------------------------------------------------------------

def config_consumer_wellness() -> SiteConfig:
    """Config A: Consumer wellness app (ani.ai). No FDA submission required."""
    return SiteConfig(
        config_id="consumer",
        display_name="ani.ai",
        config_type="wellness",
        show_clinical_thresholds=False,
        show_disease_names=False,
        show_risk_tiers=True,
        score_display_mode="wellness",
        wellness_language=True,
        anifold_enabled=False,
        behavioural_metadata_enabled=True,
        anamnesis_enabled=True,
        policy_mode="policy_shadow",
    )


def config_clinician_cds(site_id: str = "default", locale: str = "en") -> SiteConfig:
    """Config B: Clinician CDS dashboard. Non-device CDS under Cures Act.

    REGULATORY: anifold_enabled MUST be False.
    If AniFold is enabled, CDS exemption collapses → uncleared medical device.
    """
    return SiteConfig(
        config_id=f"cds_{site_id}",
        display_name=f"ani medical CDS ({site_id})",
        config_type="cds",
        locale=locale,
        show_clinical_thresholds=True,
        show_disease_names=True,
        show_risk_tiers=True,
        score_display_mode="clinical",
        wellness_language=False,
        anifold_enabled=False,      # MUST be False for CDS!
        behavioural_metadata_enabled=True,
        anamnesis_enabled=True,
        policy_mode="deterministic",
    )


def config_trial_concordance(site_id: str = "trial") -> SiteConfig:
    """Config B+: Concordance study mode. Enables paired measurement collection."""
    return SiteConfig(
        config_id=f"trial_{site_id}",
        display_name=f"ani trial ({site_id})",
        config_type="trial",
        show_clinical_thresholds=True,
        show_disease_names=True,
        score_display_mode="research",
        wellness_language=False,
        anifold_enabled=False,
        concordance_mode=True,
        concordance_alternating_weeks=True,
        behavioural_metadata_enabled=True,
        anamnesis_enabled=True,
        policy_mode="deterministic",
    )


def config_samd_full() -> SiteConfig:
    """Config C: Full AI-driven SaMD. Requires De Novo classification.

    This is the ONLY config where anifold_enabled=True.
    """
    return SiteConfig(
        config_id="samd_full",
        display_name="ani medical AI",
        config_type="samd",
        show_clinical_thresholds=True,
        show_disease_names=True,
        score_display_mode="clinical",
        wellness_language=False,
        anifold_enabled=True,       # Only in SaMD!
        anifold_endpoint="https://api.ani.ai/anifold/v1",
        behavioural_metadata_enabled=True,
        anamnesis_enabled=True,
        policy_mode="policy_live",
    )


# ---------------------------------------------------------------------------
# Configuration registry
# ---------------------------------------------------------------------------

class SiteConfigRegistry:
    """Registry of all deployment configurations."""

    def __init__(self) -> None:
        self._configs: Dict[str, SiteConfig] = {}

    def register(self, config: SiteConfig) -> None:
        """Register a site configuration."""
        _validate_config(config)
        self._configs[config.config_id] = config

    def get(self, config_id: str) -> Optional[SiteConfig]:
        """Get a configuration by ID."""
        return self._configs.get(config_id)

    def list_configs(self) -> List[SiteConfig]:
        """List all registered configurations."""
        return list(self._configs.values())

    def list_by_type(self, config_type: str) -> List[SiteConfig]:
        """List configurations by type."""
        return [c for c in self._configs.values() if c.config_type == config_type]


def default_site_registry() -> SiteConfigRegistry:
    """Create a registry with default configurations."""
    registry = SiteConfigRegistry()
    registry.register(config_consumer_wellness())
    registry.register(config_clinician_cds("default"))
    registry.register(config_trial_concordance())
    registry.register(config_samd_full())
    return registry


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ConfigValidationError(Exception):
    """Raised when a site configuration violates regulatory constraints."""
    pass


def _validate_config(config: SiteConfig) -> None:
    """Validate that a configuration meets regulatory constraints."""

    # CRITICAL: CDS configs must NOT have AniFold enabled
    if config.config_type == "cds" and config.anifold_enabled:
        raise ConfigValidationError(
            f"REGULATORY VIOLATION: CDS config '{config.config_id}' has "
            f"anifold_enabled=True. This collapses the CDS exemption under "
            f"Cures Act Section 520(o)(1)(E) Criterion 1. AniFold optical "
            f"sensing constitutes a signal acquisition system."
        )

    # Wellness configs must not show disease names or clinical thresholds
    if config.config_type == "wellness":
        if config.show_disease_names:
            raise ConfigValidationError(
                f"REGULATORY VIOLATION: Wellness config '{config.config_id}' "
                f"has show_disease_names=True. This crosses into medical device "
                f"territory under FDA General Wellness guidance."
            )
        if config.show_clinical_thresholds:
            raise ConfigValidationError(
                f"REGULATORY VIOLATION: Wellness config '{config.config_id}' "
                f"has show_clinical_thresholds=True. Clinical thresholds imply "
                f"diagnostic intent, violating General Wellness exemption."
            )

    # Trial configs should have concordance mode enabled
    if config.config_type == "trial" and not config.concordance_mode:
        raise ConfigValidationError(
            f"Trial config '{config.config_id}' should have concordance_mode=True."
        )

    # Validate daily budget
    if config.daily_question_budget < 1 or config.daily_question_budget > 20:
        raise ConfigValidationError(
            f"Invalid daily_question_budget: {config.daily_question_budget}"
        )


# ---------------------------------------------------------------------------
# Score display formatter
# ---------------------------------------------------------------------------

def format_score_display(
    *,
    config: SiteConfig,
    scale_name: str,
    raw_score: float,
    normalized_score: float,
    risk_tier: Optional[str],
    original_metric_label: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Format a score for display based on the site configuration.

    Wellness mode: personal trends, no disease language
    Clinical mode: full instrument scores, thresholds, risk tiers
    Research mode: all data plus concordance fields
    """
    display: Dict[str, Any] = {
        "scale_name": scale_name,
    }

    if config.score_display_mode == "wellness":
        # Wellness: show as percentage, personal trend, no disease names
        display["score_display"] = f"{normalized_score:.0f}/100"
        display["trend_label"] = _wellness_trend_label(normalized_score)
        if config.show_risk_tiers and risk_tier:
            display["tier"] = _wellness_tier_label(risk_tier)
        # NO disease names, NO clinical thresholds
    elif config.score_display_mode == "clinical":
        # Clinical: full transparency
        display["raw_score"] = raw_score
        display["normalized_score"] = normalized_score
        if original_metric_label:
            display["metric_label"] = original_metric_label
        if config.show_risk_tiers and risk_tier:
            display["risk_tier"] = risk_tier
        if config.show_clinical_thresholds:
            display["threshold_note"] = "See instrument publication for cut-offs"
    elif config.score_display_mode == "research":
        # Research: everything
        display["raw_score"] = raw_score
        display["normalized_score"] = normalized_score
        if original_metric_label:
            display["metric_label"] = original_metric_label
        if risk_tier:
            display["risk_tier"] = risk_tier
        display["concordance_eligible"] = config.concordance_mode

    return display


def _wellness_trend_label(normalized: float) -> str:
    """Convert normalized score to wellness-safe language."""
    if normalized >= 75:
        return "Looking strong"
    if normalized >= 50:
        return "Room for improvement"
    if normalized >= 25:
        return "Worth attention"
    return "Consider focusing here"


def _wellness_tier_label(risk_tier: str) -> str:
    """Convert clinical risk tier to wellness-safe language.

    NEVER use disease names in wellness mode.
    """
    tier_map = {
        "low": "Balanced",
        "slightly_elevated": "Shifting",
        "moderate": "Actively changing",
        "high": "Needs attention",
        "very_high": "Priority focus area",
        "low_risk": "Balanced",
        "elevated_risk": "Shifting",
        "high_risk": "Needs attention",
        "negative_screen": "Balanced",
        "positive_screen": "Worth discussing",
        "no_use": "Balanced",
        "low_use": "Balanced",
        "low_activity": "Room to move more",
        "moderate_activity": "Active",
        "high_activity": "Very active",
    }
    return tier_map.get(risk_tier, "See your patterns")
