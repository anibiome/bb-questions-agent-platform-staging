"""
Tests for Site Configuration Registry and Multi-Version Deployment.

Validates:
  1. Predefined configurations are valid
  2. CDS regulatory constraint: no AniFold
  3. Wellness regulatory constraint: no disease names
  4. Registry operations
  5. Score display formatting per config type
  6. Configuration validation catches violations
"""

import unittest

from questions_agent_platform.pipeline.site_config import (
    SiteConfig,
    SiteConfigRegistry,
    ConfigValidationError,
    config_consumer_wellness,
    config_clinician_cds,
    config_trial_concordance,
    config_samd_full,
    default_site_registry,
    format_score_display,
)


class TestPredefinedConfigs(unittest.TestCase):
    """Test that predefined configurations are valid."""

    def test_consumer_wellness(self):
        """Consumer wellness config should be valid and safe."""
        cfg = config_consumer_wellness()
        self.assertEqual(cfg.config_type, "wellness")
        self.assertFalse(cfg.anifold_enabled)
        self.assertFalse(cfg.show_disease_names)
        self.assertFalse(cfg.show_clinical_thresholds)
        self.assertTrue(cfg.wellness_language)

    def test_clinician_cds(self):
        """Clinician CDS config should be valid and transparent."""
        cfg = config_clinician_cds("sheba", locale="he")
        self.assertEqual(cfg.config_type, "cds")
        self.assertFalse(cfg.anifold_enabled)  # CRITICAL
        self.assertTrue(cfg.show_clinical_thresholds)
        self.assertTrue(cfg.show_disease_names)
        self.assertEqual(cfg.locale, "he")

    def test_trial_concordance(self):
        """Trial config should enable concordance mode."""
        cfg = config_trial_concordance("pilot_30")
        self.assertEqual(cfg.config_type, "trial")
        self.assertTrue(cfg.concordance_mode)
        self.assertTrue(cfg.concordance_alternating_weeks)
        self.assertFalse(cfg.anifold_enabled)

    def test_samd_full(self):
        """SaMD config is the only one with AniFold enabled."""
        cfg = config_samd_full()
        self.assertEqual(cfg.config_type, "samd")
        self.assertTrue(cfg.anifold_enabled)
        self.assertIsNotNone(cfg.anifold_endpoint)


class TestRegulatoryConstraints(unittest.TestCase):
    """Test that regulatory constraints are enforced."""

    def test_cds_with_anifold_rejected(self):
        """CDS config with AniFold must be rejected."""
        bad_cfg = SiteConfig(
            config_id="bad_cds",
            display_name="Bad CDS",
            config_type="cds",
            anifold_enabled=True,  # VIOLATION!
        )
        registry = SiteConfigRegistry()
        with self.assertRaises(ConfigValidationError) as ctx:
            registry.register(bad_cfg)
        self.assertIn("REGULATORY VIOLATION", str(ctx.exception))
        self.assertIn("CDS", str(ctx.exception))

    def test_wellness_with_disease_names_rejected(self):
        """Wellness config with disease names must be rejected."""
        bad_cfg = SiteConfig(
            config_id="bad_wellness",
            display_name="Bad Wellness",
            config_type="wellness",
            show_disease_names=True,  # VIOLATION!
        )
        registry = SiteConfigRegistry()
        with self.assertRaises(ConfigValidationError) as ctx:
            registry.register(bad_cfg)
        self.assertIn("show_disease_names", str(ctx.exception))

    def test_wellness_with_clinical_thresholds_rejected(self):
        """Wellness config with clinical thresholds must be rejected."""
        bad_cfg = SiteConfig(
            config_id="bad_wellness2",
            display_name="Bad Wellness 2",
            config_type="wellness",
            show_clinical_thresholds=True,  # VIOLATION!
        )
        registry = SiteConfigRegistry()
        with self.assertRaises(ConfigValidationError) as ctx:
            registry.register(bad_cfg)
        self.assertIn("clinical_thresholds", str(ctx.exception).lower())

    def test_trial_without_concordance_rejected(self):
        """Trial config without concordance mode must be rejected."""
        bad_cfg = SiteConfig(
            config_id="bad_trial",
            display_name="Bad Trial",
            config_type="trial",
            concordance_mode=False,  # VIOLATION!
        )
        registry = SiteConfigRegistry()
        with self.assertRaises(ConfigValidationError):
            registry.register(bad_cfg)


class TestSiteRegistry(unittest.TestCase):
    """Test registry operations."""

    def test_default_registry(self):
        """Default registry should have 4 configs."""
        reg = default_site_registry()
        configs = reg.list_configs()
        self.assertEqual(len(configs), 4)

    def test_get_by_id(self):
        """Should retrieve config by ID."""
        reg = default_site_registry()
        consumer = reg.get("consumer")
        self.assertIsNotNone(consumer)
        self.assertEqual(consumer.config_type, "wellness")

    def test_list_by_type(self):
        """Should filter configs by type."""
        reg = default_site_registry()
        cds_configs = reg.list_by_type("cds")
        self.assertGreater(len(cds_configs), 0)
        for cfg in cds_configs:
            self.assertEqual(cfg.config_type, "cds")

    def test_register_custom(self):
        """Should register custom site configs."""
        reg = default_site_registry()
        custom = SiteConfig(
            config_id="cds_tel_aviv",
            display_name="Tel Aviv Medical CDS",
            config_type="cds",
            locale="he",
            show_clinical_thresholds=True,
            show_disease_names=True,
            wellness_language=False,
        )
        reg.register(custom)
        retrieved = reg.get("cds_tel_aviv")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.locale, "he")

    def test_nonexistent_config(self):
        """Should return None for nonexistent config ID."""
        reg = default_site_registry()
        self.assertIsNone(reg.get("nonexistent"))


class TestScoreDisplay(unittest.TestCase):
    """Test score formatting per config type."""

    def test_wellness_display_no_disease(self):
        """Wellness display must NOT include disease language."""
        cfg = config_consumer_wellness()
        display = format_score_display(
            config=cfg,
            scale_name="Metabolic Health",
            raw_score=15.0,
            normalized_score=65.0,
            risk_tier="moderate",
        )
        self.assertIn("score_display", display)
        self.assertNotIn("raw_score", display)
        self.assertIn("trend_label", display)
        # Should NOT contain disease-specific language
        for value in display.values():
            if isinstance(value, str):
                self.assertNotIn("diabetes", value.lower())
                self.assertNotIn("disease", value.lower())

    def test_clinical_display_full_transparency(self):
        """Clinical display should show raw scores and risk tiers."""
        cfg = config_clinician_cds()
        display = format_score_display(
            config=cfg,
            scale_name="FINDRISC",
            raw_score=15.0,
            normalized_score=65.0,
            risk_tier="high",
            original_metric_label="FINDRISC (0-26)",
        )
        self.assertIn("raw_score", display)
        self.assertEqual(display["raw_score"], 15.0)
        self.assertIn("risk_tier", display)
        self.assertEqual(display["risk_tier"], "high")

    def test_research_display_includes_concordance(self):
        """Research display should include concordance flag."""
        cfg = config_trial_concordance()
        display = format_score_display(
            config=cfg,
            scale_name="FINDRISC",
            raw_score=15.0,
            normalized_score=65.0,
            risk_tier="high",
        )
        self.assertIn("concordance_eligible", display)
        self.assertTrue(display["concordance_eligible"])

    def test_wellness_tier_labels_safe(self):
        """Wellness tier labels must use safe language."""
        cfg = config_consumer_wellness()
        # Test various risk tiers
        for tier in ["low", "moderate", "high", "very_high", "positive_screen"]:
            display = format_score_display(
                config=cfg,
                scale_name="Test",
                raw_score=10.0,
                normalized_score=50.0,
                risk_tier=tier,
            )
            tier_label = display.get("tier", "")
            # Should never contain disease names
            self.assertNotIn("diabetes", tier_label.lower())
            self.assertNotIn("cancer", tier_label.lower())
            self.assertNotIn("disease", tier_label.lower())


class TestConfigSeparation(unittest.TestCase):
    """Test that Config B and Config C are strictly separated."""

    def test_no_cds_config_has_anifold(self):
        """No CDS config in the default registry should have AniFold."""
        reg = default_site_registry()
        for cfg in reg.list_by_type("cds"):
            self.assertFalse(
                cfg.anifold_enabled,
                f"CDS config {cfg.config_id} has AniFold enabled!"
            )

    def test_only_samd_has_anifold(self):
        """Only SaMD config should have AniFold enabled."""
        reg = default_site_registry()
        for cfg in reg.list_configs():
            if cfg.config_type != "samd":
                self.assertFalse(
                    cfg.anifold_enabled,
                    f"Non-SaMD config {cfg.config_id} has AniFold enabled!"
                )


if __name__ == "__main__":
    unittest.main()
