"""
Evaluation profile for HydraNet models on UCDP targets.

Inherits all genome values from BASE_PROFILE (including Brier thresholds,
QS quantile, Coverage alpha, etc.) and overrides where domain needs differ.
"""

from views_evaluation.profiles.base import BASE_PROFILE

HYDRANET_UCDP_PROFILE = {
    **BASE_PROFILE,
    "twCRPS": {"threshold": 0.0},                                    # placeholder — discuss
    "MIS":    {"alpha": 0.05},                                       # placeholder — discuss
    "QIS":    {"lower_quantile": 0.025, "upper_quantile": 0.975},   # placeholder — discuss
}
