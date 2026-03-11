"""
Named evaluation profiles registry.

Each profile provides hyperparameter values for metrics that require them.
Models select a profile via the 'evaluation_profile' config key.

To add a new profile:
    1. Create a new file in this directory (e.g. sb_best_pgm.py)
    2. Define a profile dict (see base.py for the pattern)
    3. Import and register it in the PROFILES dict below
"""

from views_evaluation.profiles.base import BASE_PROFILE

PROFILES = {
    "base": BASE_PROFILE,
    # Researchers add target+unit specific profiles here:
    # "sb_best_pgm": SB_BEST_PGM_PROFILE,
    # "sb_best_cm": SB_BEST_CM_PROFILE,
}
