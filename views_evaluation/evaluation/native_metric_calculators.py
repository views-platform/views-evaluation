import numpy as np
import properscoring as ps
from sklearn.metrics import (
    root_mean_squared_log_error,
    mean_squared_error,
    mean_squared_log_error,
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

def calculate_crps_native(y_true: np.ndarray, y_pred: np.ndarray, target=None, **kwargs) -> float:
    y_true, y_pred = _guard_shapes(y_true, y_pred)
    return np.mean(ps.crps_ensemble(y_true, y_pred, axis=1))

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

def calculate_mtd_native(y_true: np.ndarray, y_pred: np.ndarray, target=None, power: float = 1.5, **kwargs) -> float:
    y_true, y_pred = _guard_shapes(y_true, y_pred)
    return mean_tweedie_deviance(
        np.repeat(y_true, y_pred.shape[1]), 
        y_pred.flatten(),
        power=power
    )

def calculate_mean_prediction_native(y_true: np.ndarray, y_pred: np.ndarray, target=None, **kwargs) -> float:
    return np.mean(y_pred)

def calculate_coverage_native(y_true: np.ndarray, y_pred: np.ndarray, target=None, alpha=0.1, **kwargs) -> float:
    y_true, y_pred = _guard_shapes(y_true, y_pred)
    lower = np.quantile(y_pred, alpha / 2, axis=1)
    upper = np.quantile(y_pred, 1 - alpha / 2, axis=1)
    covered = (y_true >= lower) & (y_true <= upper)
    return np.mean(covered)

def calculate_mean_interval_score_native(y_true: np.ndarray, y_pred: np.ndarray, target=None, alpha=0.05, **kwargs) -> float:
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
    bins=[0, 0.5, 2.5, 5.5, 10.5, 25.5, 50.5, 100.5, 250.5, 500.5, 1000.5],
    low_bin=0,
    high_bin=10000,
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

# Placeholder functions for unimplemented metrics to satisfy tests
def calculate_sd_native(*args, **kwargs): raise NotImplementedError()
def calculate_pEMDiv_native(*args, **kwargs): raise NotImplementedError()
def calculate_variogram_native(*args, **kwargs): raise NotImplementedError()
def calculate_brier_native(*args, **kwargs): raise NotImplementedError()
def calculate_jeffreys_native(*args, **kwargs): raise NotImplementedError()

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
    "SD":        calculate_sd_native,
    "pEMDiv":    calculate_pEMDiv_native,
    "Variogram": calculate_variogram_native,
}




REGRESSION_SAMPLE_NATIVE = {
    "CRPS":      calculate_crps_native,
    "MIS":       calculate_mean_interval_score_native,
    "Coverage":  calculate_coverage_native,
    "Ignorance": calculate_ignorance_score_native,
    "y_hat_bar": calculate_mean_prediction_native,
    "Brier":     calculate_brier_native,
    "Jeffreys":  calculate_jeffreys_native,
}

CLASSIFICATION_POINT_NATIVE = {
    "AP": calculate_ap_native,
}

CLASSIFICATION_SAMPLE_NATIVE = {
    "CRPS":      calculate_crps_native,
    "Brier":     calculate_brier_native,
    "Jeffreys":  calculate_jeffreys_native,
}
