
import warnings

def raise_legacy_scale_msg() -> None:

    """
    Emit a highly visible warning banner for legacy scale-detection behavior
    that should eventually be removed, but does not currently break execution.
    """

    default_msg = """
Currently, the evaluation package infers target scaling (e.g. log, linear)
from the target variable name (lr_, ln_, lx_).

This is problematic because:

1) Target scaling is a MODEL parameter and must live with the model,
   not be inferred from target names.

2) Adding new scales would require updating a hard-coded list in the
   evaluation package, which is brittle and volatile.

3) Target prefixes (lr_, ln_, lx_) are not guarantees of scaling —
   at best they are hints, and can lead to silent errors.

As such, this behavior should be removed.
Targets should always be assumed unscaled.
"""

    banner = (
        "\n"
        + "#" * 78 + "\n"
        + "#{:^76}#\n".format("LEGACY SCALE DETECTION — SHOULD BE REMOVED")
        + "#" * 78 + "\n"
        + (default_msg).strip() + "\n"
        + "#" * 78
    )

    # Use UserWarning so it is always shown (DeprecationWarning is often suppressed)
    warnings.warn(banner, UserWarning, stacklevel=2)
