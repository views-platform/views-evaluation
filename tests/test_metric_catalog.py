"""
Tests for the MetricCatalog, named profiles, and Chain of Responsibility resolver.

Test color key (ADR-020):
    Green = happy path / correctness
    Red   = adversarial / must fail loudly
    Beige = realistic edge cases ("boring but dangerous")
"""

import pytest
from views_evaluation.evaluation.metric_catalog import (
    METRIC_CATALOG,
    METRIC_MEMBERSHIP,
    resolve_metric_params,
)
from views_evaluation.profiles import PROFILES
from views_evaluation.profiles.base import BASE_PROFILE
from views_evaluation.evaluation.native_metric_calculators import (
    calculate_mcr_native,
)
import numpy as np


# ---------------------------------------------------------------------------
# Green: happy path — resolve_metric_params
# ---------------------------------------------------------------------------

class TestResolveMetricParamsGreen:

    def test_empty_genome_resolves_to_empty_dict(self):
        """Metrics with no hyperparameters (e.g. MSE) resolve to {}."""
        result = resolve_metric_params("MSE", {}, BASE_PROFILE)
        assert result == {}

    def test_genome_resolves_from_profile(self):
        """twCRPS threshold resolves from the base profile."""
        result = resolve_metric_params("twCRPS", {}, BASE_PROFILE)
        assert result == {"threshold": BASE_PROFILE["twCRPS"]["threshold"]}

    def test_model_override_takes_precedence(self):
        """Model override beats profile value."""
        result = resolve_metric_params("twCRPS", {"threshold": 5.0}, BASE_PROFILE)
        assert result == {"threshold": 5.0}

    def test_partial_override_merges_with_profile(self):
        """QIS: override one param, get the other from profile."""
        result = resolve_metric_params(
            "QIS",
            {"lower_quantile": 0.1},
            BASE_PROFILE,
        )
        assert result["lower_quantile"] == 0.1
        assert result["upper_quantile"] == BASE_PROFILE["QIS"]["upper_quantile"]

    def test_multi_param_genome_resolves_fully(self):
        """Ignorance has 3 genome params — all must resolve."""
        result = resolve_metric_params("Ignorance", {}, BASE_PROFILE)
        assert set(result.keys()) == {"bins", "low_bin", "high_bin"}
        assert result["low_bin"] == 0
        assert result["high_bin"] == 10000

    def test_coverage_resolves_from_profile(self):
        """Coverage alpha resolves from base profile."""
        result = resolve_metric_params("Coverage", {}, BASE_PROFILE)
        assert result == {"alpha": BASE_PROFILE["Coverage"]["alpha"]}

    def test_mis_resolves_from_profile(self):
        """MIS alpha resolves from base profile."""
        result = resolve_metric_params("MIS", {}, BASE_PROFILE)
        assert result == {"alpha": BASE_PROFILE["MIS"]["alpha"]}

    def test_mtd_resolves_from_profile(self):
        """MTD power resolves from base profile."""
        result = resolve_metric_params("MTD", {}, BASE_PROFILE)
        assert result == {"power": BASE_PROFILE["MTD"]["power"]}


# ---------------------------------------------------------------------------
# Red: adversarial — must fail loud
# ---------------------------------------------------------------------------

class TestResolveMetricParamsRed:

    def test_unknown_metric_raises(self):
        """Unknown metric name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown metric"):
            resolve_metric_params("NONEXISTENT", {}, BASE_PROFILE)

    def test_unknown_override_param_raises(self):
        """Override with a param not in the genome raises ValueError."""
        with pytest.raises(ValueError, match="unknown parameters"):
            resolve_metric_params("twCRPS", {"bad_param": 1.0}, BASE_PROFILE)

    def test_missing_param_not_in_profile_raises(self):
        """Param required by genome but missing from both overrides and profile."""
        empty_profile = {}
        with pytest.raises(ValueError, match="requires parameter"):
            resolve_metric_params("twCRPS", {}, empty_profile)

    def test_none_value_raises(self):
        """Explicitly passing None for a genome param raises ValueError."""
        with pytest.raises(ValueError, match="None values"):
            resolve_metric_params("twCRPS", {"threshold": None}, BASE_PROFILE)

    def test_none_in_profile_raises(self):
        """None in profile (not overridden) raises ValueError."""
        bad_profile = {"twCRPS": {"threshold": None}}
        with pytest.raises(ValueError, match="None values"):
            resolve_metric_params("twCRPS", {}, bad_profile)

    def test_unimplemented_metric_raises(self):
        """Requesting an unimplemented metric raises ValueError."""
        with pytest.raises(ValueError, match="not yet implemented"):
            resolve_metric_params("SD", {}, BASE_PROFILE)

    def test_overrides_on_empty_genome_raises(self):
        """Passing overrides for a metric with no genome raises ValueError."""
        with pytest.raises(ValueError, match="has no hyperparameters"):
            resolve_metric_params("MSE", {"spurious": 1.0}, BASE_PROFILE)


# ---------------------------------------------------------------------------
# Red: bounds validation for hyperparameters (C-18)
# ---------------------------------------------------------------------------

class TestResolveMetricParamsBoundsRed:
    """Bounds validation for metric hyperparameters."""

    def test_alpha_above_one_raises(self):
        """Coverage alpha must be in (0, 1)."""
        with pytest.raises(ValueError, match="alpha"):
            resolve_metric_params("Coverage", {"alpha": 1.5}, BASE_PROFILE)

    def test_alpha_zero_raises(self):
        """Coverage alpha=0 would cause division by zero in MIS."""
        with pytest.raises(ValueError, match="alpha"):
            resolve_metric_params("MIS", {"alpha": 0.0}, BASE_PROFILE)

    def test_alpha_negative_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            resolve_metric_params("Coverage", {"alpha": -0.1}, BASE_PROFILE)

    def test_quantile_above_one_raises(self):
        """QS quantile must be in (0, 1)."""
        with pytest.raises(ValueError, match="quantile"):
            resolve_metric_params("QS_sample", {"quantile": 1.0}, BASE_PROFILE)

    def test_quantile_zero_raises(self):
        with pytest.raises(ValueError, match="quantile"):
            resolve_metric_params("QS_sample", {"quantile": 0.0}, BASE_PROFILE)

    def test_quantile_negative_raises(self):
        with pytest.raises(ValueError, match="quantile"):
            resolve_metric_params("QS_point", {"quantile": -0.5}, BASE_PROFILE)

    def test_qis_lower_quantile_above_upper_raises(self):
        """QIS lower_quantile must be < upper_quantile."""
        with pytest.raises(ValueError, match="quantile"):
            resolve_metric_params(
                "QIS", {"lower_quantile": 0.9, "upper_quantile": 0.1}, BASE_PROFILE
            )


# ---------------------------------------------------------------------------
# Beige: structural integrity
# ---------------------------------------------------------------------------

class TestCatalogStructuralIntegrity:

    def test_every_catalog_metric_in_at_least_one_membership(self):
        """Every metric in METRIC_CATALOG appears in at least one METRIC_MEMBERSHIP set."""
        all_members = set()
        for members in METRIC_MEMBERSHIP.values():
            all_members |= members
        for metric_name in METRIC_CATALOG:
            assert metric_name in all_members, (
                f"Metric '{metric_name}' is in METRIC_CATALOG but not in any METRIC_MEMBERSHIP set"
            )

    def test_every_membership_metric_in_catalog(self):
        """Every metric in METRIC_MEMBERSHIP exists in METRIC_CATALOG."""
        for key, members in METRIC_MEMBERSHIP.items():
            for metric_name in members:
                assert metric_name in METRIC_CATALOG, (
                    f"Metric '{metric_name}' in METRIC_MEMBERSHIP{key} but not in METRIC_CATALOG"
                )

    def test_base_profile_covers_all_implemented_genomes(self):
        """BASE_PROFILE must provide values for every genome param of every implemented metric."""
        for metric_name, spec in METRIC_CATALOG.items():
            if not spec.implemented or not spec.genome:
                continue
            assert metric_name in BASE_PROFILE, (
                f"Implemented metric '{metric_name}' with genome {spec.genome} "
                f"is missing from BASE_PROFILE"
            )
            for param in spec.genome:
                assert param in BASE_PROFILE[metric_name], (
                    f"BASE_PROFILE['{metric_name}'] is missing genome param '{param}'"
                )
                assert BASE_PROFILE[metric_name][param] is not None, (
                    f"BASE_PROFILE['{metric_name}']['{param}'] is None"
                )

    def test_metric_membership_matches_dataclass_fields(self):
        """Every metric in METRIC_MEMBERSHIP must have a field in its dataclass."""
        import dataclasses
        from views_evaluation.evaluation.metrics import (
            RegressionPointEvaluationMetrics,
            RegressionSampleEvaluationMetrics,
            ClassificationPointEvaluationMetrics,
            ClassificationSampleEvaluationMetrics,
        )
        cls_map = {
            ("regression", "point"): RegressionPointEvaluationMetrics,
            ("regression", "sample"): RegressionSampleEvaluationMetrics,
            ("classification", "point"): ClassificationPointEvaluationMetrics,
            ("classification", "sample"): ClassificationSampleEvaluationMetrics,
        }
        for (task, pred_type), metric_names in METRIC_MEMBERSHIP.items():
            cls = cls_map[(task, pred_type)]
            fields = {f.name for f in dataclasses.fields(cls)}
            missing = metric_names - fields
            assert not missing, (
                f"METRIC_MEMBERSHIP[({task!r}, {pred_type!r})] contains {sorted(missing)} "
                f"but {cls.__name__} has no corresponding fields"
            )

    def test_metric_spec_is_frozen(self):
        """MetricSpec instances must be immutable."""
        spec = METRIC_CATALOG["MSE"]
        with pytest.raises(AttributeError):
            spec.genome = ("bad",)


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

class TestProfiles:

    def test_base_profile_registered(self):
        """The 'base' profile must be available."""
        assert "base" in PROFILES

    def test_base_profile_is_base_profile(self):
        """PROFILES['base'] is the same object as BASE_PROFILE."""
        assert PROFILES["base"] is BASE_PROFILE

    def test_custom_profile_overrides_base(self):
        """A custom profile can override base values using dict spread."""
        custom = {
            **BASE_PROFILE,
            "twCRPS": {"threshold": 5.0},
        }
        result = resolve_metric_params("twCRPS", {}, custom)
        assert result == {"threshold": 5.0}

    def test_custom_profile_inherits_unmodified_base_values(self):
        """Custom profile inherits values it doesn't override."""
        custom = {
            **BASE_PROFILE,
            "twCRPS": {"threshold": 5.0},
        }
        result = resolve_metric_params("Coverage", {}, custom)
        assert result == {"alpha": BASE_PROFILE["Coverage"]["alpha"]}

    def test_hydranet_ucdp_profile_registered(self):
        """The 'hydranet_ucdp' profile must be available."""
        assert "hydranet_ucdp" in PROFILES

    def test_hydranet_ucdp_resolves_twcrps(self):
        """hydranet_ucdp profile resolves twCRPS threshold."""
        profile = PROFILES["hydranet_ucdp"]
        result = resolve_metric_params("twCRPS", {}, profile)
        assert "threshold" in result

    def test_hydranet_ucdp_resolves_mis(self):
        """hydranet_ucdp profile resolves MIS alpha."""
        profile = PROFILES["hydranet_ucdp"]
        result = resolve_metric_params("MIS", {}, profile)
        assert "alpha" in result

    def test_hydranet_ucdp_resolves_qis(self):
        """hydranet_ucdp profile resolves QIS quantile levels."""
        profile = PROFILES["hydranet_ucdp"]
        result = resolve_metric_params("QIS", {}, profile)
        assert "lower_quantile" in result
        assert "upper_quantile" in result

    def test_hydranet_ucdp_inherits_base_coverage(self):
        """hydranet_ucdp inherits Coverage from base (not explicitly overridden)."""
        profile = PROFILES["hydranet_ucdp"]
        result = resolve_metric_params("Coverage", {}, profile)
        assert result == {"alpha": BASE_PROFILE["Coverage"]["alpha"]}


# ---------------------------------------------------------------------------
# Green/Red: MCR (Magnitude Calibration Ratio)
# ---------------------------------------------------------------------------

class TestMCR:

    def test_perfect_calibration(self):
        """MCR = 1.0 when predictions match observations."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([[1.0], [2.0], [3.0]])
        assert calculate_mcr_native(y_true, y_pred) == 1.0

    def test_overprediction(self):
        """MCR > 1 when systematically overpredicting."""
        y_true = np.array([1.0, 1.0, 1.0])
        y_pred = np.array([[2.0], [2.0], [2.0]])
        assert calculate_mcr_native(y_true, y_pred) == 2.0

    def test_underprediction(self):
        """MCR < 1 when systematically underpredicting."""
        y_true = np.array([2.0, 2.0, 2.0])
        y_pred = np.array([[1.0], [1.0], [1.0]])
        assert calculate_mcr_native(y_true, y_pred) == 0.5

    def test_sample_predictions(self):
        """MCR averages over samples correctly."""
        y_true = np.array([1.0, 1.0])
        y_pred = np.array([[1.0, 3.0], [1.0, 3.0]])  # mean(y_pred) = 2.0
        assert calculate_mcr_native(y_true, y_pred) == 2.0

    def test_zero_true_positive_pred_returns_inf(self):
        """MCR returns inf when mean(y_true) == 0 and mean(y_pred) > 0."""
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([[1.0], [1.0], [1.0]])
        result = calculate_mcr_native(y_true, y_pred)
        assert result == np.inf

    def test_zero_both_returns_nan(self):
        """MCR returns nan when both means are 0."""
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([[0.0], [0.0]])
        result = calculate_mcr_native(y_true, y_pred)
        assert np.isnan(result)

    def test_mcr_point_in_catalog(self):
        """MCR_point is registered in METRIC_CATALOG."""
        assert "MCR_point" in METRIC_CATALOG

    def test_mcr_sample_in_catalog(self):
        """MCR_sample is registered in METRIC_CATALOG."""
        assert "MCR_sample" in METRIC_CATALOG


# ---------------------------------------------------------------------------
# Integration: NativeEvaluator with catalog
# ---------------------------------------------------------------------------

class TestNativeEvaluatorCatalogIntegration:

    def test_evaluator_accepts_base_profile(self):
        """NativeEvaluator initializes with the base profile."""
        from views_evaluation.evaluation.native_evaluator import NativeEvaluator
        config = {
            "regression_targets": ["test_target"],
            "regression_sample_metrics": ["CRPS"],
            "steps": [1],
        }
        evaluator = NativeEvaluator(config)
        assert evaluator.profile is BASE_PROFILE

    def test_evaluator_accepts_explicit_profile(self):
        """NativeEvaluator initializes with an explicitly named profile."""
        from views_evaluation.evaluation.native_evaluator import NativeEvaluator
        config = {
            "evaluation_profile": "base",
            "regression_targets": ["test_target"],
            "regression_sample_metrics": ["CRPS"],
            "steps": [1],
        }
        evaluator = NativeEvaluator(config)
        assert evaluator.profile is BASE_PROFILE

    def test_evaluator_rejects_unknown_profile(self):
        """NativeEvaluator raises on unknown profile."""
        from views_evaluation.evaluation.native_evaluator import NativeEvaluator
        config = {"evaluation_profile": "nonexistent"}
        with pytest.raises(ValueError, match="Unknown evaluation profile"):
            NativeEvaluator(config)

    def test_registry_snapshot_integrity(self):
        """Registries have expected sizes — catches accidental mutation or deletion."""
        assert len(METRIC_CATALOG) == 24
        assert len(METRIC_MEMBERSHIP) == 4
        assert len(PROFILES) >= 2

    def test_evaluator_metric_overrides_stored(self):
        """NativeEvaluator stores metric overrides from config."""
        from views_evaluation.evaluation.native_evaluator import NativeEvaluator
        config = {
            "regression_targets": ["test_target"],
            "regression_sample_metrics": ["twCRPS"],
            "steps": [1],
            "metric_hyperparameters": {
                "twCRPS": {"threshold": 3.0},
            },
        }
        evaluator = NativeEvaluator(config)
        assert evaluator.metric_overrides == {"twCRPS": {"threshold": 3.0}}
