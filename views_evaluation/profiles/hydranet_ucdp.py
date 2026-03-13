"""
Evaluation profile for HydraNet models on UCDP targets.

Covers probabilistic evaluation metrics that require hyperparameters:
twCRPS, MIS, QIS. All other metrics (CRPS, AP, Brier, etc.) have
empty genomes and need no profile entries.

Values are placeholders inheriting from base — adjust per domain needs.
"""

from views_evaluation.profiles.base import BASE_PROFILE

HYDRANET_UCDP_PROFILE = {
    **BASE_PROFILE,
    "twCRPS": {"threshold": 0.0},                                    # placeholder — discuss
    "MIS":    {"alpha": 0.05},                                       # placeholder — discuss
    "QIS":    {"lower_quantile": 0.025, "upper_quantile": 0.975},   # placeholder — discuss
}
