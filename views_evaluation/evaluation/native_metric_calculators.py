import warnings

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    mean_tweedie_deviance,
)
from scipy.stats import wasserstein_distance, pearsonr
from scipy.stats import ConstantInputWarning

def _guard_shapes(y_true: np.ndarray, y_pred: np.ndarray):
    """Internal guard to prevent broadcasting accidents.

    Assumes numeric NumPy arrays (guaranteed by EvaluationFrame._validate()).
    Validates shapes and normalises dimensions for metric functions.
    """
    # Shape validation (ADR-013)
    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1D, got shape {y_true.shape}")

    if y_pred.ndim == 1:
        # Reshape to (N, 1) for point forecasts passed as 1D
        y_pred = y_pred.reshape(-1, 1)

    if y_pred.ndim != 2:
        raise ValueError(f"y_pred must be 2D (N, S), got shape {y_pred.shape}")

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"Row mismatch: y_true={y_true.shape[0]}, y_pred={y_pred.shape[0]}")

    return y_true, y_pred

def calculate_mse_native(y_true: np.ndarray, y_pred: np.ndarray, target=None, **kwargs) -> float:
    y_true, y_pred = _guard_shapes(y_true, y_pred)
    return np.mean((y_true[:, np.newaxis] - y_pred) ** 2)

def calculate_msle_native(y_true: np.ndarray, y_pred: np.ndarray, target=None, **kwargs) -> float:
    y_true, y_pred = _guard_shapes(y_true, y_pred)
    if np.any(y_true < -1) or np.any(y_pred < -1):
        raise ValueError("MSLE input contains values < -1")
    return np.mean((np.log1p(y_true[:, np.newaxis]) - np.log1p(y_pred)) ** 2)

def calculate_rmsle_native(y_true: np.ndarray, y_pred: np.ndarray, target=None, **kwargs) -> float:
    return np.sqrt(calculate_msle_native(y_true, y_pred, target=target, **kwargs))

def _crps_ensemble_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Pure-numpy CRPS for finite ensembles using the sorted-array identity.

    CRPS(y, X) = (1/S) * sum|x_j - y| - (1/(2*S^2)) * sum|x_i - x_j|

    The pairwise spread term is computed efficiently via the sorted-ensemble
    identity: sum_{i<j}(x_j - x_i) = sum_j (2j - S + 1) * x_j  (0-indexed).

    Args:
        y_true: (N,) observations, already validated by _guard_shapes.
        y_pred: (N, S) ensemble members, already validated by _guard_shapes.

    Returns:
        (N,) per-observation CRPS values.
    """
    sorted_pred = np.sort(y_pred, axis=1)
    S = sorted_pred.shape[1]

    # Mean absolute error between each ensemble member and the observation
    mae_term = np.mean(np.abs(sorted_pred - y_true[:, np.newaxis]), axis=1)

    # Pairwise spread via sorted-array identity
    weights = (2 * np.arange(S) - S + 1).astype(float)  # shape (S,)
    spread_term = np.sum(weights[np.newaxis, :] * sorted_pred, axis=1) / (S * S)

    return mae_term - spread_term


def calculate_crps_native(y_true: np.ndarray, y_pred: np.ndarray, target=None, **kwargs) -> float:
    y_true, y_pred = _guard_shapes(y_true, y_pred)
    return float(np.mean(_crps_ensemble_numpy(y_true, y_pred)))

def calculate_ap_native(y_true: np.ndarray, y_pred: np.ndarray, target=None, **kwargs) -> float:
    y_true, y_pred = _guard_shapes(y_true, y_pred)
    return average_precision_score(
        np.repeat(y_true, y_pred.shape[1]), 
        y_pred.flatten()
    )

def calculate_emd_native(y_true: np.ndarray, y_pred: np.ndarray, target=None, **kwargs) -> float:
    y_true, y_pred = _guard_shapes(y_true, y_pred)
    emd_list = [wasserstein_distance(y_pred[i], [y_true[i]]) for i in range(len(y_true))]
    return np.mean(emd_list)

def calculate_pearson_native(y_true: np.ndarray, y_pred: np.ndarray, target=None, **kwargs) -> float:
    """
    Pearson linear correlation between observations and predictions.

    Returns np.nan if either series is constant — correlation requires variance in
    both, so no coefficient exists for that group.

    A constant series is a **property of the data**, not a broken invariant, so this
    is a documented sentinel rather than a failure (ADR-015 ruling 2). A constant
    prediction series means the model is a baseline (e.g. "predict zero everywhere"),
    which is a finding about the model, not an error; a constant truth series means the
    group had no variation to correlate against. Both are ordinary in conflict data,
    where most units are zero most of the time.

    Consumers should treat np.nan as "not computable for this group" and exclude it
    from aggregates — `EvaluationReport.to_metric_frame()` already does, via nanmean.

    See also `calculate_mcr_native`, which returns inf/nan on a zero-truth group for
    the same reason: the degenerate case is a fact about the data, not a fault.
    """
    y_true, y_pred = _guard_shapes(y_true, y_pred)

    observed = np.repeat(y_true, y_pred.shape[1])
    predicted = y_pred.flatten()

    # Suppress SciPy's ConstantInputWarning: the degenerate case is handled and
    # contracted above, so the warning is noise rather than signal. Scoped to this
    # call only — it must never mask a warning from anything else.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        correlation, _ = pearsonr(observed, predicted)
    return correlation

def calculate_mtd_native(y_true: np.ndarray, y_pred: np.ndarray, target=None, *, power: float, **kwargs) -> float:
    y_true, y_pred = _guard_shapes(y_true, y_pred)
    return mean_tweedie_deviance(
        np.repeat(y_true, y_pred.shape[1]), 
        y_pred.flatten(),
        power=power
    )

def calculate_mean_prediction_native(y_true: np.ndarray, y_pred: np.ndarray, target=None, **kwargs) -> float:
    # C-32: this kernel ignores y_true, but it must still guard like every sibling.
    # Without the call it silently returned a number for shape-mismatched input while
    # all ~20 neighbouring kernels raised.
    y_true, y_pred = _guard_shapes(y_true, y_pred)
    return np.mean(y_pred)

def calculate_mcr_native(y_true: np.ndarray, y_pred: np.ndarray, target=None, **kwargs) -> float:
    """
    Magnitude Calibration Ratio: mean(y_pred) / mean(y_true).

    MCR = 1 → perfect aggregate calibration.
    MCR < 1 → systematic underprediction.
    MCR > 1 → systematic overprediction.

    Returns np.inf if mean(y_true) == 0 and mean(y_pred) > 0, np.nan if both are 0.
    """
    y_true, y_pred = _guard_shapes(y_true, y_pred)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    if mean_true == 0.0:
        return float(np.inf) if mean_pred > 0.0 else float(np.nan)
    return float(mean_pred / mean_true)

def calculate_coverage_native(y_true: np.ndarray, y_pred: np.ndarray, target=None, *, alpha: float, **kwargs) -> float:
    y_true, y_pred = _guard_shapes(y_true, y_pred)
    lower = np.quantile(y_pred, alpha / 2, axis=1)
    upper = np.quantile(y_pred, 1 - alpha / 2, axis=1)
    covered = (y_true >= lower) & (y_true <= upper)
    return np.mean(covered)

def calculate_mean_interval_score_native(y_true: np.ndarray, y_pred: np.ndarray, target=None, *, alpha: float, **kwargs) -> float:
    y_true, y_pred = _guard_shapes(y_true, y_pred)
    lower = np.quantile(y_pred, q=alpha / 2, axis=1)
    upper = np.quantile(y_pred, q=1 - (alpha / 2), axis=1)
    
    interval_width = upper - lower
    lower_coverage = (2 / alpha) * (lower - y_true) * (y_true < lower)
    upper_coverage = (2 / alpha) * (y_true - upper) * (y_true > upper)
    interval_score = interval_width + lower_coverage + upper_coverage
    return np.mean(interval_score)

def calculate_ignorance_score_native(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target=None,
    *,
    bins,
    low_bin: int,
    high_bin: int,
    **kwargs
) -> float:
    """
    Ignorance score over a set of bin edges.

    Args:
        bins: A sequence of bin edges covering the target's domain, supplied by the
              evaluation profile. Half-open: an observation is in range for
              ``[bins[0], bins[-1])``.
        low_bin: **RESERVED — currently has no effect.** See the status block below.
        high_bin: **RESERVED — currently has no effect.** See the status block below.

    ┌──────────────────────────────────────────────────────────────────────────────┐
    │ STATUS OF ``low_bin`` / ``high_bin`` — RESERVED PLACEHOLDERS, NOT LIVE        │
    │ Recorded 2026-08-02. Risk register C-28(b).                                   │
    └──────────────────────────────────────────────────────────────────────────────┘

    **These two parameters do nothing.** They are passed to
    ``np.histogram_bin_edges(preds, bins=bins, range=(low_bin, high_bin))``, and NumPy
    **silently ignores ``range`` whenever ``bins`` is a sequence** — which it always is
    in every shipped profile. Verified 2026-08-02: computing edges with
    ``range=(0, 10000)`` and with ``range=(0, 1)`` returns byte-identical arrays.

    So the genome *requires* them, ``resolve_metric_params`` *fails loud* if a profile
    omits them, and changing them alters nothing. They are retained deliberately as
    placeholders for planned work, not because they function.

    **This is a use-it-or-lose-it parameter. It does not get to sit here indefinitely.**

    ── If ACTIVATED, they must conform to all of the following ──────────────────────

    1. **They must actually change the binning.** The only way ``range`` takes effect is
       if ``bins`` is an integer *count* rather than a sequence, in which case NumPy
       computes that many equal-width bins spanning ``(low_bin, high_bin)``. Supporting
       that means supporting two mutually exclusive bin-specification modes.
    2. **Contradictory configuration must fail loud, not be silently resolved**
       (ADR-013, ADR-015). If ``bins`` is a sequence *and* ``low_bin``/``high_bin`` are
       supplied, that is a contradiction — the caller has specified the domain twice, in
       two ways, which may disagree. Raise; do not pick a winner.
    3. **They must satisfy ``low_bin < high_bin``**, validated in
       ``resolve_metric_params`` alongside the existing bounds checks for ``alpha`` and
       the quantile parameters (``metric_catalog.py``, ``_UNIT_INTERVAL_EXCLUSIVE``).
    4. **They must agree with ADR-015 ruling 8.** An observation outside the bins raises.
       If ``low_bin``/``high_bin`` become the declared domain of the target, the
       out-of-range error message must cite *them*, not ``edges[0]``/``edges[-1]``, or
       the message will point the researcher at the wrong knob.
    5. **The semantics must be written down** in the profile docstring and in
       ``CICs/MetricCatalog.md`` — specifically whether they mean "the target's plausible
       domain" (a declaration about the data) or "the span to divide into bins" (an
       instruction about the algorithm). Those are different things and only the second
       is what NumPy's ``range`` does.

    ── If NOT activated ────────────────────────────────────────────────────────────

    Delete them from ``genome`` in ``metric_catalog.py``, from every profile, and from
    this signature. A required parameter that changes nothing is dead configuration
    presenting itself as a control, and it will mislead the next person who tries to
    fix an out-of-range failure by widening ``high_bin`` — which will do nothing at all.
    """
    y_true, y_pred = _guard_shapes(y_true, y_pred)
    
    def digitize_minus_one(x, edges):
        return np.digitize(x, edges, right=False) - 1

    scores = []
    for i in range(len(y_true)):
        preds = y_pred[i]
        truth = float(y_true[i])

        # NOTE: `range` is INERT here — NumPy ignores it whenever `bins` is a sequence,
        # which it always is in every shipped profile. Retained as a reserved
        # placeholder; see the status block in this function's docstring (C-28b).
        edges = np.histogram_bin_edges(preds, bins=bins, range=(low_bin, high_bin))
        binned_preds = digitize_minus_one(preds, edges)
        binned_obs = digitize_minus_one([truth], edges)[0]

        n_bins = len(edges) - 1

        # ADR-015 ruling 8: an observation outside the configured bins means the
        # profile's `bins` do not cover the target's domain. That is a configuration
        # error for the caller to fix — clamping it into an edge bin would silently
        # redefine the metric exactly at the tails, which is where conflict data
        # matters most.
        #
        # Without this guard both tails failed, differently and badly:
        #   above -> binned_obs == n_bins, an IndexError (unless a prediction happened
        #            to also be out of range, which silently made the index valid)
        #   below -> binned_obs == -1, which negative-indexes into the LAST bin and
        #            returned a plausible, wrong score with no error at all.
        if not 0 <= binned_obs < n_bins:
            raise ValueError(
                f"Ignorance: observation {truth} at row {i} lies outside the configured "
                f"bin range [{edges[0]}, {edges[-1]}). Widen 'bins' in the evaluation "
                f"profile so it covers the target's observed domain."
            )

        bin_counts = np.bincount(binned_preds, minlength=n_bins)
        smoothed_counts = bin_counts + 1

        prob = smoothed_counts[binned_obs] / np.sum(smoothed_counts)
        scores.append(-np.log2(prob))

    return np.mean(scores)

def calculate_twcrps_native(y_true: np.ndarray, y_pred: np.ndarray, target=None, *, threshold: float, **kwargs) -> float:
    """
    Threshold-weighted CRPS using the chaining representation.

    twCRPS(y, X; tau) = CRPS(max(y, tau), max(X, tau))

    This focuses evaluation on values above the threshold, which is useful for
    conflict data where non-zero counts are the decision-critical regime.

    Args:
        threshold: Values below this are clamped. Must be provided explicitly
                   via evaluation profile or model config.
    """
    y_true, y_pred = _guard_shapes(y_true, y_pred)
    y_true_tw = np.maximum(y_true, threshold)
    y_pred_tw = np.maximum(y_pred, threshold)
    return float(np.mean(_crps_ensemble_numpy(y_true_tw, y_pred_tw)))


def calculate_quantile_interval_score_native(
    y_true: np.ndarray, y_pred: np.ndarray, target=None,
    *, lower_quantile: float, upper_quantile: float, **kwargs
) -> float:
    """
    Quantile Interval Score for asymmetric quantile levels.

    Generalises the symmetric Interval Score (MIS) to allow different lower
    and upper quantile levels.  When lower_quantile == alpha/2 and
    upper_quantile == 1 - alpha/2, the result is identical to MIS with that alpha.

    Args:
        lower_quantile: Lower quantile level (0 < lower < upper < 1).
        upper_quantile: Upper quantile level.
    """
    y_true, y_pred = _guard_shapes(y_true, y_pred)
    lower = np.quantile(y_pred, lower_quantile, axis=1)
    upper = np.quantile(y_pred, upper_quantile, axis=1)

    interval_width = upper - lower
    lower_penalty = (2 / lower_quantile) * (lower - y_true) * (y_true < lower)
    upper_penalty = (2 / (1 - upper_quantile)) * (y_true - upper) * (y_true > upper)
    qis = interval_width + lower_penalty + upper_penalty
    return float(np.mean(qis))


# ── Brier Score ───────────────────────────────────────────────────────────────
#
# Three explicit variants for the 2×2 evaluation matrix:
#   Brier_cls_point  — classification point: y_pred is a probability
#   Brier_cls_sample — classification sample: y_pred are probability samples (MC Dropout)
#   Brier_rgs_sample — regression sample: y_pred are count/magnitude samples
#
# Brier_rgs_point is intentionally omitted: a regression point estimate
# is not a probability, so calling the result a Brier score is misleading.

def calculate_brier_cls_point_native(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target=None,
    *,
    threshold: float,
    **kwargs,
) -> float:
    """
    Brier Score for classification point (probability) predictions.

    Binarises truth at the threshold, uses the point prediction
    directly as the predicted probability.

    Brier = mean((y_pred - y_binary)^2)

    y_pred values should be in [0, 1] for meaningful results.
    For point predictions, y_pred is (N, 1) after _guard_shapes.

    Args:
        threshold: Onset threshold for binarising y_true.
    """
    y_true, y_pred = _guard_shapes(y_true, y_pred)
    y_binary = (y_true > threshold).astype(float)
    p_hat = y_pred[:, 0]
    return float(np.mean((p_hat - y_binary) ** 2))


def calculate_brier_cls_sample_native(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target=None,
    *,
    threshold: float,
    **kwargs,
) -> float:
    """
    Brier Score for classification probability samples (e.g. MC Dropout).

    Each sample in y_pred is a probability in [0, 1]. The posterior mean
    probability is used as the point estimate:

    Brier = mean((mean(y_pred, axis=1) - y_binary)^2)

    where y_binary = (y_true > threshold).

    This is the correct formulation for probability samples — averaging
    probabilities preserves calibration information. Binarising probability
    samples at a threshold (as Brier_rgs_sample does for count data) would
    destroy discrimination.

    Args:
        threshold: Onset threshold for binarising y_true.
    """
    y_true, y_pred = _guard_shapes(y_true, y_pred)
    y_binary = (y_true > threshold).astype(float)
    p_hat = np.mean(y_pred, axis=1)
    return float(np.mean((p_hat - y_binary) ** 2))


def calculate_brier_rgs_sample_native(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target=None,
    *,
    threshold: float,
    **kwargs,
) -> float:
    """
    Brier Score for regression (count/magnitude) samples.

    Binarises both truth and each sample at the threshold, then
    estimates the event probability from the fraction of ensemble
    members exceeding the threshold.

    Brier = mean((p_hat - y_binary)^2)

    where p_hat = mean(y_pred > threshold, axis=1) and
    y_binary = (y_true > threshold).

    Args:
        threshold: Onset threshold for binarisation of both y_true
                   and y_pred. Must be provided explicitly via
                   evaluation profile or model config.
    """
    y_true, y_pred = _guard_shapes(y_true, y_pred)
    y_binary = (y_true > threshold).astype(float)
    p_hat = np.mean(y_pred > threshold, axis=1)
    return float(np.mean((p_hat - y_binary) ** 2))


# ── Quantile Score (Pinball Loss) ─────────────────────────────────────────────

def calculate_qs_sample_native(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target=None,
    *,
    quantile: float,
    **kwargs,
) -> float:
    """
    Quantile Score (pinball loss) for sample-based predictions.

    Extracts the specified quantile from the forecast ensemble, then
    computes the asymmetric pinball loss.

    QS = mean(max(alpha * (y - q), (1 - alpha) * (q - y)))

    where q = np.quantile(y_pred, quantile, axis=1).

    Args:
        quantile: Quantile level in (0, 1). E.g. 0.99 for QS99.
    """
    y_true, y_pred = _guard_shapes(y_true, y_pred)
    q = np.quantile(y_pred, quantile, axis=1)
    diff = y_true - q
    scores = np.where(
        diff >= 0,
        diff * quantile,
        -diff * (1 - quantile),
    )
    return float(np.mean(scores))


def calculate_qs_point_native(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target=None,
    *,
    quantile: float,
    **kwargs,
) -> float:
    """
    Quantile Score (pinball loss) for point predictions.

    The point prediction is treated as the quantile estimate directly.
    Computes the asymmetric pinball loss.

    QS = mean(max(alpha * (y - y_hat), (1 - alpha) * (y_hat - y)))

    For point predictions, y_pred is (N, 1) after _guard_shapes.

    Args:
        quantile: Quantile level in (0, 1). E.g. 0.99 for QS99.
    """
    y_true, y_pred = _guard_shapes(y_true, y_pred)
    q = y_pred[:, 0]
    diff = y_true - q
    scores = np.where(
        diff >= 0,
        diff * quantile,
        -diff * (1 - quantile),
    )
    return float(np.mean(scores))


# Placeholder functions for metrics that are planned but not yet implemented.
# ADR-013: Raise ValueError (not NotImplementedError) so callers get a
# consistent, user-facing message rather than a bare exception type.
def calculate_sd_native(*args, **kwargs):
    raise ValueError(
        "Metric 'SD' is defined but not yet implemented."
        " Remove it from your config."
    )
def calculate_pEMDiv_native(*args, **kwargs):
    raise ValueError(
        "Metric 'pEMDiv' is defined but not yet implemented."
        " Remove it from your config."
    )
def calculate_variogram_native(*args, **kwargs):
    raise ValueError(
        "Metric 'Variogram' is defined but not yet implemented."
        " Remove it from your config."
    )
def calculate_jeffreys_native(*args, **kwargs):
    raise ValueError(
        "Metric 'Jeffreys' is defined but not yet implemented."
        " Remove it from your config."
    )

