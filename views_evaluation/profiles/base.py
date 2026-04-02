"""
Base evaluation profile — the system-wide standard evaluation protocol.

This profile provides hyperparameter values for all metrics that require them.
It ships with views-evaluation and serves as the baseline for all evaluations.

Researchers can create additional profiles for specific target+unit combinations
(e.g. sb_best_pgm.py) that override base values where domain knowledge dictates
different evaluation parameters.

Example researcher-added profile (sb_best_pgm.py):

    from views_evaluation.profiles.base import BASE_PROFILE

    SB_BEST_PGM_PROFILE = {
        **BASE_PROFILE,
        "twCRPS": {"threshold": 1.0},  # Higher for PRIO-GRID conflict counts
    }

Then register it in views_evaluation/profiles/__init__.py.
"""

BASE_PROFILE = {
    "MTD":           {"power": 1.5},
    "twCRPS":        {"threshold": 0.0},
    "MIS":           {"alpha": 0.05},
    "QIS":           {"lower_quantile": 0.025, "upper_quantile": 0.975},
    "QS_sample":     {"quantile": 0.99},
    "QS_point":      {"quantile": 0.99},
    "Brier_sample":  {"threshold": 1.0},
    "Brier_point":   {"threshold": 1.0},
    "Coverage":      {"alpha": 0.1},
    "Ignorance": {
        "bins": [0, 0.5, 2.5, 5.5, 10.5, 25.5, 50.5, 100.5, 250.5, 500.5, 1000.5],
        "low_bin": 0,
        "high_bin": 10000,
    },
}
