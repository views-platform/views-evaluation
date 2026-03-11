"""
Metric Catalog — genome registry and Chain of Responsibility resolver.

This module implements the catalog pattern from views-r2darts2 for evaluation
metrics. Each metric declares its required hyperparameters (genome) but provides
NO default values. Values are supplied by named profiles and/or per-model overrides.

Architecture (ADR-042):
    MetricSpec declares requirements → profiles supply values → resolver validates

Chain of Responsibility for parameter resolution:
    model overrides → profile → fail loud
"""

from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Any

from views_evaluation.evaluation.native_metric_calculators import (
    calculate_mse_native,
    calculate_msle_native,
    calculate_rmsle_native,
    calculate_crps_native,
    calculate_twcrps_native,
    calculate_quantile_interval_score_native,
    calculate_ap_native,
    calculate_emd_native,
    calculate_pearson_native,
    calculate_mtd_native,
    calculate_mean_prediction_native,
    calculate_coverage_native,
    calculate_mean_interval_score_native,
    calculate_ignorance_score_native,
    calculate_sd_native,
    calculate_pEMDiv_native,
    calculate_variogram_native,
    calculate_brier_native,
    calculate_jeffreys_native,
)


@dataclass(frozen=True)
class MetricSpec:
    """
    Immutable specification for a single evaluation metric.

    Attributes:
        function: The callable that computes the metric.
        genome: Tuple of required hyperparameter names. Empty tuple means
                the metric needs no hyperparameters.
        implemented: False for placeholder metrics (SD, Variogram, etc.)
                     that are defined but raise ValueError when called.
    """
    function: Callable
    genome: Tuple[str, ...]
    implemented: bool = True


METRIC_CATALOG: Dict[str, MetricSpec] = {
    # ── Regression Point ──────────────────────────────────────────────────
    "MSE":       MetricSpec(function=calculate_mse_native,       genome=()),
    "MSLE":      MetricSpec(function=calculate_msle_native,      genome=()),
    "RMSLE":     MetricSpec(function=calculate_rmsle_native,     genome=()),
    "EMD":       MetricSpec(function=calculate_emd_native,       genome=()),
    "Pearson":   MetricSpec(function=calculate_pearson_native,   genome=()),
    "MTD":       MetricSpec(function=calculate_mtd_native,       genome=("power",)),
    "y_hat_bar": MetricSpec(function=calculate_mean_prediction_native, genome=()),
    "SD":        MetricSpec(function=calculate_sd_native,        genome=(), implemented=False),
    "pEMDiv":    MetricSpec(function=calculate_pEMDiv_native,    genome=(), implemented=False),
    "Variogram": MetricSpec(function=calculate_variogram_native, genome=(), implemented=False),

    # ── Regression Sample ─────────────────────────────────────────────────
    "CRPS":      MetricSpec(function=calculate_crps_native,      genome=()),
    "twCRPS":    MetricSpec(function=calculate_twcrps_native,    genome=("threshold",)),
    "MIS":       MetricSpec(function=calculate_mean_interval_score_native, genome=("alpha",)),
    "QIS":       MetricSpec(function=calculate_quantile_interval_score_native,
                            genome=("lower_quantile", "upper_quantile")),
    "Coverage":  MetricSpec(function=calculate_coverage_native,  genome=("alpha",)),
    "Ignorance": MetricSpec(function=calculate_ignorance_score_native,
                            genome=("bins", "low_bin", "high_bin")),

    # ── Classification ────────────────────────────────────────────────────
    "AP":        MetricSpec(function=calculate_ap_native,        genome=()),
    "Brier":     MetricSpec(function=calculate_brier_native,     genome=(), implemented=False),
    "Jeffreys":  MetricSpec(function=calculate_jeffreys_native,  genome=(), implemented=False),
}


METRIC_MEMBERSHIP: Dict[Tuple[str, str], set] = {
    ("regression", "point"):      {"MSE", "MSLE", "RMSLE", "EMD", "Pearson", "MTD",
                                    "y_hat_bar", "SD", "pEMDiv", "Variogram"},
    ("regression", "sample"):     {"CRPS", "twCRPS", "MIS", "QIS", "Coverage",
                                    "Ignorance", "y_hat_bar", "Brier", "Jeffreys"},
    ("classification", "point"):  {"AP"},
    ("classification", "sample"): {"CRPS", "twCRPS", "Brier", "Jeffreys"},
}


def resolve_metric_params(
    metric_name: str,
    model_overrides: Dict[str, Any],
    profile: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Resolve hyperparameters for a metric via Chain of Responsibility.

    Resolution order: model_overrides → profile → fail loud.

    Args:
        metric_name: Name of the metric (must exist in METRIC_CATALOG).
        model_overrides: Per-metric overrides from the model's config.
                         E.g. {"threshold": 2.0} for twCRPS.
        profile: Named evaluation profile providing values for all metrics.
                 E.g. BASE_PROFILE from views_evaluation.profiles.base.

    Returns:
        Dict of resolved hyperparameters ready to pass as **kwargs.
        Empty dict for metrics with no genome.

    Raises:
        ValueError: If metric_name is unknown, metric is not implemented,
                    a required parameter is missing from both overrides and
                    profile, a resolved parameter is None, or an unknown
                    parameter is provided in model_overrides.
    """
    if metric_name not in METRIC_CATALOG:
        raise ValueError(
            f"Unknown metric '{metric_name}'. "
            f"Available: {sorted(METRIC_CATALOG.keys())}"
        )

    spec = METRIC_CATALOG[metric_name]

    if not spec.implemented:
        raise ValueError(
            f"Metric '{metric_name}' is defined but not yet implemented. "
            f"Remove it from your config."
        )

    if not spec.genome:
        # No hyperparameters needed — reject any overrides as an error
        if model_overrides:
            raise ValueError(
                f"Metric '{metric_name}' has no hyperparameters, "
                f"but overrides were provided: {sorted(model_overrides.keys())}"
            )
        return {}

    # Reject unknown overrides
    unknown = set(model_overrides.keys()) - set(spec.genome)
    if unknown:
        raise ValueError(
            f"Metric '{metric_name}' received unknown parameters: {sorted(unknown)}. "
            f"Valid parameters: {sorted(spec.genome)}"
        )

    # Resolve each genome parameter via chain
    resolved = {}
    profile_params = profile.get(metric_name, {})

    for param in spec.genome:
        if param in model_overrides:
            resolved[param] = model_overrides[param]
        elif param in profile_params:
            resolved[param] = profile_params[param]
        else:
            raise ValueError(
                f"Metric '{metric_name}' requires parameter '{param}' but it was "
                f"not found in model overrides or the evaluation profile. "
                f"Required genome: {spec.genome}"
            )

    # Fail-loud: no None values (mirrors r2darts2 ReproducibilityGate)
    nones = [k for k, v in resolved.items() if v is None]
    if nones:
        raise ValueError(
            f"Metric '{metric_name}' has None values for parameters: {nones}. "
            f"All hyperparameters must be explicitly set."
        )

    return resolved
