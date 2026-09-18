import warnings

import numpy as np
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


# ── Pure-numpy AP and Tweedie deviance (register C-05; epic #66, stories #70 and #64) ──
#
# These replaced the scikit-learn kernels on 2026-09-18. Each is a transcription of the
# scikit-learn 1.7.2 algorithm, so the switch was numerically invisible for float64
# input with one ruled exception — AP on a group with no positive truth returns `nan`
# where scikit-learn returned 0.0 (ADR-015 R9) — and the parity suite in
# tests/test_metric_calculators.py still runs them against scikit-learn, now a dev-only
# dependency, as the oracle, permanently. Level 0 imports
# numpy and scipy only (ADR-011), so `import views_evaluation` no longer loads pandas
# through scikit-learn.
#
# Input validation is the kernels' own. On the NativeEvaluator path EvaluationFrame
# already rejects NaN, inf and object dtype, but the public kernels and
# METRIC_CATALOG[...].function are importable directly, and scikit-learn's check_array
# rejected some of what the frame lets through (complex, empty, mismatched length).
# Dropping the oracle must not turn those raises into numbers. The validation here is
# stricter than scikit-learn's in a few places, each named in the CHANGELOG as an input
# that now fails: string and timedelta arrays (scikit-learn cast strings to numbers), a
# bool `power`, and a single truth label outside {-1, 0, 1}.


def _kernel_vector(name: str, values) -> np.ndarray:
    """A validated 1-D real, finite, non-empty vector for a kernel — what scikit-learn's
    `check_array(ensure_2d=False, ensure_min_samples=1)` plus `assert_all_finite`
    guaranteed. Returned in its own dtype (integer or floating); callers cast where the
    arithmetic needs float64, and not where it does not (AP sorts scores natively, so
    int64 scores above 2**53 stay distinct as they do in scikit-learn)."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} is empty; at least one sample is required")
    if arr.dtype == bool:
        arr = arr.astype(np.int64)
    # kind, not issubdtype: np.timedelta64 subclasses np.signedinteger and would pass.
    if arr.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be real-numeric, got dtype {arr.dtype}")
    if arr.dtype.kind == "f" and not np.all(np.isfinite(arr)):
        bad = int(np.flatnonzero(~np.isfinite(arr))[0])
        # .item(): a Python float, so the message reads `nan`, not `np.float64(nan)` under numpy 2
        raise ValueError(f"{name} contains {arr[bad].item()!r} at index {bad}; values must be finite")
    return arr


def _average_precision_numpy(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision as scikit-learn computes it, in numpy.

    Reproduces ``sklearn.metrics.average_precision_score`` for binary ``y_true``:
    a descending-score threshold sweep with tied scores collapsed to one operating
    point (``_binary_clf_curve``), recall against the total positives, and the
    step-wise sum ``Σ (R_n − R_{n−1}) P_n`` over the reversed curve with the ``(1, 0)``
    endpoint appended. Not the trapezoidal area under the curve.

    **A group with no positive label returns `nan`** (ADR-015 ruling 9, decided
    2026-09-18; register D-01). Average precision is undefined without a positive to
    recall; scikit-learn's convention on that input — set recall to 1 everywhere and
    return `0.0` with a `UserWarning` — is a convention, not a score, and would mark a
    model that correctly predicted "nothing here" as the worst possible. An empty
    group is a property of conflict data, not a fault, so it is ruled as `MCR` and
    `Pearson` are: a documented sentinel, warning suppressed, excluded from the
    `to_metric_frame()` mean row by `nanmean`.

    Accepted ``y_true``: integral values, at most two distinct, and if two then ``1``
    among them; the positive class is ``y_true == 1``. That is scikit-learn's accepted
    set (``type_of_target`` in {"binary"} with ``pos_label=1``), so nothing that
    computes today starts failing, but the rejection is a plain ``ValueError`` naming
    the observed labels rather than scikit-learn's "Expected 2D array" (multiclass
    path) or "continuous format is not supported".

    Args:
        y_true: (M,) labels, already repeated per sample by the caller.
        y_score: (M,) scores, already flattened.
    """
    y_true = _kernel_vector("y_true", y_true)
    y_score = _kernel_vector("y_score", y_score)
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError(
            f"y_true and y_score have different lengths: {y_true.shape[0]} vs {y_score.shape[0]}"
        )
    labels = np.unique(y_true)
    if (
        len(labels) > 2
        or (len(labels) == 2 and 1 not in labels)
        or (len(labels) == 1 and labels[0] not in (-1, 0, 1))
        or np.any(labels != np.floor(labels))
    ):
        # The single-label clause is stricter than scikit-learn, which computed 0.0 for a
        # uniform 2 or 5: under R9 that input would become a silent `nan`,
        # indistinguishable from a legitimate all-zero group, and a uniformly mis-coded
        # truth column is a fault the caller must see.
        raise ValueError(
            f"AP requires binary y_true with 1 as the positive label; observed labels "
            f"{labels.tolist()[:8]}{'...' if len(labels) > 8 else ''}"
        )
    positive = y_true == 1
    if not positive.any():
        return float("nan")  # ADR-015 R9: undefined for this group, not "worst possible"

    # Stable ascending sort, then reversed — scikit-learn's exact permutation, which
    # matters only for which tied element comes first and is then collapsed anyway.
    # Sorted in the scores' own dtype: casting int64 to float64 first would merge
    # values above 2**53 into false ties.
    order = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[order]
    positive = positive[order]

    # One operating point per distinct score: the LAST index of each run of ties.
    distinct = np.where(np.diff(y_score) != 0)[0]
    thresholds = np.r_[distinct, positive.size - 1]
    tps = np.cumsum(positive, dtype=np.float64)[thresholds]
    fps = 1 + thresholds - tps

    precision = tps / (tps + fps)  # never zero: each threshold has >= 1 prediction
    recall = tps / tps[-1]

    # Reverse so recall is decreasing, append the (precision=1, recall=0) endpoint,
    # then the negated step integral; clip a numerical -0.0.
    precision = np.hstack((precision[::-1], 1.0))
    recall = np.hstack((recall[::-1], 0.0))
    return float(max(0.0, -np.sum(np.diff(recall) * precision[:-1])))


def _tweedie_deviance_numpy(y_true: np.ndarray, y_pred: np.ndarray, power: float) -> float:
    """Mean Tweedie deviance as scikit-learn computes it, in numpy, always in float64.

    Reproduces ``sklearn.metrics.mean_tweedie_deviance`` branch for branch: p < 0
    (extreme stable; the first term clamps y at 0), p == 0 (Gaussian, squared error),
    p == 1 (Poisson, with x·log(x/μ) taken as 0 at x == 0), p == 2 (Gamma), and the
    general form for every other admissible power (1 < p < 2 compound Poisson–Gamma,
    p > 2). Each domain raise starts with scikit-learn's sentence verbatim and appends
    the offending value and its index, which ADR-015 requires and scikit-learn omits.

    Deviations, all documented: ``power`` in the open interval (0, 1), non-finite, or
    not a real number is rejected with a plain ``ValueError`` (scikit-learn: its own
    ``InvalidParameterError``, a ``ValueError`` subclass, with different text). And the
    arithmetic is always float64. scikit-learn computes in the highest-precision
    floating dtype among its inputs, so float32 input is computed in float32 there;
    the difference from this kernel's float64 answer is ~1e-8 relative at p = 1.5 but
    up to ~2e-3 near p = 1 and p = 2 (measured over 200 seeds), where the closed form
    cancels. This kernel's answer is the more accurate one; the parity suite asserts it
    against scikit-learn's float64 result, and records the float32 gap.

    Args:
        y_true: (M,) observations, already repeated per sample by the caller.
        y_pred: (M,) predictions, already flattened.
        power: the Tweedie power p, a finite real number outside (0, 1).
    """
    import numbers

    if isinstance(power, bool) or not isinstance(power, numbers.Real) or not np.isfinite(float(power)):
        raise ValueError(
            f"Mean Tweedie deviance requires a finite real power; got {power!r} "
            f"({type(power).__name__})"
        )
    p = float(power)
    if 0.0 < p < 1.0:
        raise ValueError(
            f"Mean Tweedie deviance is not defined for power in (0, 1); got {power}"
        )
    y = _kernel_vector("y_true", y_true).astype(np.float64)
    mu = _kernel_vector("y_pred", y_pred).astype(np.float64)
    if y.shape[0] != mu.shape[0]:
        raise ValueError(
            f"y_true and y_pred have different lengths: {y.shape[0]} vs {mu.shape[0]}"
        )

    message = f"Mean Tweedie deviance error with power={power} can only be used on "

    def offending(mask, name, arr):
        i = int(np.flatnonzero(mask)[0])
        return f" Offending {name}: {arr[i].item()!r} at index {i}."

    def generic(y_first_term):
        # The closed form for every power outside {0, 1, 2}. p < 0 passes a clamped y
        # for the first term only — scikit-learn's `where(y > 0, y, 0)` — and the
        # signed y for the second.
        return 2 * (
            np.power(y_first_term, 2 - p) / ((1 - p) * (2 - p))
            - y * np.power(mu, 1 - p) / (1 - p)
            + np.power(mu, 2 - p) / (2 - p)
        )

    if p < 0:
        if np.any(mu <= 0):
            raise ValueError(message + "strictly positive y_pred." + offending(mu <= 0, "y_pred", mu))
        dev = generic(np.where(y > 0, y, 0.0))
    elif p == 0:
        dev = (y - mu) ** 2
    elif 1 <= p < 2:
        if np.any(y < 0):
            raise ValueError(message + "non-negative y and strictly positive y_pred." + offending(y < 0, "y", y))
        if np.any(mu <= 0):
            raise ValueError(message + "non-negative y and strictly positive y_pred." + offending(mu <= 0, "y_pred", mu))
        if p == 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                xlogy = np.where(y == 0.0, 0.0, y * np.log(y / mu))
            dev = 2 * (xlogy - y + mu)
        else:
            dev = generic(y)
    elif p >= 2:
        if np.any(y <= 0):
            raise ValueError(message + "strictly positive y and y_pred." + offending(y <= 0, "y", y))
        if np.any(mu <= 0):
            raise ValueError(message + "strictly positive y and y_pred." + offending(mu <= 0, "y_pred", mu))
        if p == 2:
            dev = 2 * (np.log(mu / y) + y / mu - 1)
        else:
            dev = generic(y)
    else:  # unreachable: every finite real p outside (0, 1) matched above
        raise ValueError(f"unhandled Tweedie power {power!r}")
    return float(np.mean(dev))


def calculate_crps_native(y_true: np.ndarray, y_pred: np.ndarray, target=None, **kwargs) -> float:
    y_true, y_pred = _guard_shapes(y_true, y_pred)
    return float(np.mean(_crps_ensemble_numpy(y_true, y_pred)))

def calculate_ap_native(y_true: np.ndarray, y_pred: np.ndarray, target=None, **kwargs) -> float:
    """Average precision; `nan` for a group with no positive truth (ADR-015 R9)."""
    y_true, y_pred = _guard_shapes(y_true, y_pred)
    return _average_precision_numpy(np.repeat(y_true, y_pred.shape[1]), y_pred.flatten())

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
    """Mean Tweedie deviance, computed in float64 for every power outside (0, 1)."""
    y_true, y_pred = _guard_shapes(y_true, y_pred)
    return _tweedie_deviance_numpy(np.repeat(y_true, y_pred.shape[1]), y_pred.flatten(), power)

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

