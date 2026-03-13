import numpy as np
from sklearn.metrics import (
    average_precision_score,
    mean_tweedie_deviance,
)
from scipy.stats import wasserstein_distance, pearsonr

def _guard_shapes(y_true: np.ndarray, y_pred: np.ndarray):
    """Internal guard to prevent broadcasting accidents. Handles conversion from legacy pandas."""
    if hasattr(y_true, "values"):
        # Extract values from Series/DataFrame
        y_true = y_true.values
    if hasattr(y_pred, "values"):
        y_pred = y_pred.values

    # Handle lists-in-cells (legacy structure)
    def ensure_array(x):
        if isinstance(x, (list, np.ndarray)):
            if len(x) > 0 and isinstance(x[0], (list, np.ndarray)):
                return np.array([ensure_array(i) for i in x])
            return np.array(x)
        return np.array([x])

    if y_true.dtype == object:
        y_true = np.array([x[0] if isinstance(x, (list, np.ndarray)) else x for x in y_true])
    if y_pred.dtype == object:
        y_pred = np.array([ensure_array(x).flatten() for x in y_pred])

    # Shape validation — unconditional, runs for all dtypes (ADR-013)
    if y_true.ndim == 2 and y_true.shape[1] == 1:
        y_true = y_true.flatten()
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
    y_true, y_pred = _guard_shapes(y_true, y_pred)
    correlation, _ = pearsonr(
        np.repeat(y_true, y_pred.shape[1]), 
        y_pred.flatten()
    )
    return correlation

def calculate_mtd_native(y_true: np.ndarray, y_pred: np.ndarray, target=None, *, power: float, **kwargs) -> float:
    y_true, y_pred = _guard_shapes(y_true, y_pred)
    return mean_tweedie_deviance(
        np.repeat(y_true, y_pred.shape[1]), 
        y_pred.flatten(),
        power=power
    )

def calculate_mean_prediction_native(y_true: np.ndarray, y_pred: np.ndarray, target=None, **kwargs) -> float:
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
    y_true, y_pred = _guard_shapes(y_true, y_pred)
    
    def digitize_minus_one(x, edges):
        return np.digitize(x, edges, right=False) - 1

    scores = []
    for i in range(len(y_true)):
        preds = y_pred[i]
        truth = float(y_true[i])

        edges = np.histogram_bin_edges(preds, bins=bins, range=(low_bin, high_bin))
        binned_preds = digitize_minus_one(preds, edges)
        binned_obs = digitize_minus_one([truth], edges)[0]

        n_bins = len(edges) - 1
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


# Placeholder functions for metrics that are planned but not yet implemented.
# ADR-013: Raise ValueError (not NotImplementedError) so callers get a consistent,
# user-facing message rather than a bare exception type.
def calculate_sd_native(*args, **kwargs):
    raise ValueError("Metric 'SD' is defined but not yet implemented. Remove it from your config.")
def calculate_pEMDiv_native(*args, **kwargs):
    raise ValueError("Metric 'pEMDiv' is defined but not yet implemented. Remove it from your config.")
def calculate_variogram_native(*args, **kwargs):
    raise ValueError("Metric 'Variogram' is defined but not yet implemented. Remove it from your config.")
def calculate_brier_native(*args, **kwargs):
    raise ValueError("Metric 'Brier' is defined but not yet implemented. Remove it from your config.")
def calculate_jeffreys_native(*args, **kwargs):
    raise ValueError("Metric 'Jeffreys' is defined but not yet implemented. Remove it from your config.")

# Legacy aliases for backward compatibility with tests
calculate_mse = calculate_mse_native
calculate_msle = calculate_msle_native
calculate_rmsle = calculate_rmsle_native
calculate_crps = calculate_crps_native
calculate_ap = calculate_ap_native
calculate_emd = calculate_emd_native
calculate_pearson = calculate_pearson_native
calculate_mtd = calculate_mtd_native
calculate_coverage = calculate_coverage_native
calculate_mean_interval_score = calculate_mean_interval_score_native
calculate_ignorance_score = calculate_ignorance_score_native
calculate_sd = calculate_sd_native
calculate_pEMDiv = calculate_pEMDiv_native
calculate_variogram = calculate_variogram_native
calculate_twcrps = calculate_twcrps_native
calculate_quantile_interval_score = calculate_quantile_interval_score_native
calculate_brier = calculate_brier_native
calculate_jeffreys = calculate_jeffreys_native

# Dispatch dicts (Framework Agnostic)
REGRESSION_POINT_NATIVE = {
    "MSE":       calculate_mse_native,
    "MSLE":      calculate_msle_native,
    "RMSLE":     calculate_rmsle_native,
    "EMD":       calculate_emd_native,
    "Pearson":   calculate_pearson_native,
    "MTD":       calculate_mtd_native,
    "y_hat_bar": calculate_mean_prediction_native,
    "MCR_point": calculate_mcr_native,
    "SD":        calculate_sd_native,
    "pEMDiv":    calculate_pEMDiv_native,
    "Variogram": calculate_variogram_native,
}




REGRESSION_SAMPLE_NATIVE = {
    "CRPS":      calculate_crps_native,
    "twCRPS":    calculate_twcrps_native,
    "MIS":       calculate_mean_interval_score_native,
    "QIS":       calculate_quantile_interval_score_native,
    "Coverage":  calculate_coverage_native,
    "Ignorance": calculate_ignorance_score_native,
    "y_hat_bar": calculate_mean_prediction_native,
    "MCR_sample": calculate_mcr_native,
    "Brier":     calculate_brier_native,
    "Jeffreys":  calculate_jeffreys_native,
}

CLASSIFICATION_POINT_NATIVE = {
    "AP": calculate_ap_native,
}

CLASSIFICATION_SAMPLE_NATIVE = {
    "CRPS":      calculate_crps_native,
    "twCRPS":    calculate_twcrps_native,
    "Brier":     calculate_brier_native,
    "Jeffreys":  calculate_jeffreys_native,
}
