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
    "Brier_cls_point":  {"threshold": 0.0},  # hurdle event: any fatality (y > 0)
    "Brier_cls_sample": {"threshold": 0.0},  # hurdle event: any fatality (y > 0)
    "Brier_rgs_sample": {"threshold": 0.0},  # hurdle event: any fatality (y > 0)
    "Coverage":      {"alpha": 0.1},
    # `bins` — explicit half-open bin edges: an observation is in range for [0, 1000.5).
    #   Outside that, Ignorance raises (ADR-015 ruling 8). Widen THIS list to change the
    #   accepted domain — it is the only one of the three that has any effect.
    #
    # `low_bin` / `high_bin` — ⚠ RESERVED PLACEHOLDERS, CURRENTLY INERT.
    #   These are required by the Ignorance genome and must be present, but they change
    #   nothing: NumPy ignores `range=` whenever `bins` is a sequence. Widening
    #   `high_bin` to fix an out-of-range failure WILL NOT WORK — widen `bins` instead.
    #   Use-it-or-lose-it; activation spec is in calculate_ignorance_score_native's
    #   docstring. Risk register C-28(b).
    "Ignorance": {
        "bins": [0, 0.5, 2.5, 5.5, 10.5, 25.5, 50.5, 100.5, 250.5, 500.5, 1000.5],
        "low_bin": 0,       # inert — see note above
        "high_bin": 10000,  # inert — see note above
    },
}
