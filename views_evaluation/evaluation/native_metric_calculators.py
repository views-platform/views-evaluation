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

def calculate_mse_native(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Vectorized: (N, 1) - (N, S) broadcasts to (N, S)
    # This is mathematically equivalent to expanded MSE but 14x faster.
    return np.mean((y_true[:, np.newaxis] - y_pred) ** 2)

def calculate_msle_native(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Vectorized MSLE
    return np.mean((np.log1p(y_true[:, np.newaxis]) - np.log1p(y_pred)) ** 2)

def calculate_rmsle_native(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(calculate_msle_native(y_true, y_pred))


def calculate_crps_native(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Legacy uses a loop. We use vectorized for better performance if possible,
    # but for Phase 1 parity we might need to be careful.
    # The benchmark showed parity between loop and axis=1.
    return np.mean(ps.crps_ensemble(y_true, y_pred, axis=1))

def calculate_ap_native(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # classification: actuals expanded to match samples
    return average_precision_score(
        np.repeat(y_true, y_pred.shape[1]), 
        y_pred.flatten()
    )

def calculate_emd_native(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Legacy loop
    emd_list = [wasserstein_distance(y_pred[i], [y_true[i]]) for i in range(len(y_true))]
    return np.mean(emd_list)

def calculate_pearson_native(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    correlation, _ = pearsonr(
        np.repeat(y_true, y_pred.shape[1]), 
        y_pred.flatten()
    )
    return correlation

def calculate_mtd_native(y_true: np.ndarray, y_pred: np.ndarray, power: float = 1.5) -> float:
    return mean_tweedie_deviance(
        np.repeat(y_true, y_pred.shape[1]), 
        y_pred.flatten(),
        power=power
    )

def calculate_mean_prediction_native(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(y_pred)

def calculate_coverage_native(y_true: np.ndarray, y_pred: np.ndarray, alpha=0.1) -> float:
    lower = np.quantile(y_pred, alpha / 2, axis=1)
    upper = np.quantile(y_pred, 1 - alpha / 2, axis=1)
    covered = (y_true >= lower) & (y_true <= upper)
    return np.mean(covered)

def calculate_mean_interval_score_native(y_true: np.ndarray, y_pred: np.ndarray, alpha=0.05) -> float:
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
    bins=[0, 0.5, 2.5, 5.5, 10.5, 25.5, 50.5, 100.5, 250.5, 500.5, 1000.5],
    low_bin=0,
    high_bin=10000,
) -> float:
    
    def digitize_minus_one(x, edges):
        return np.digitize(x, edges, right=False) - 1

    scores = []
    for i in range(len(y_true)):
        preds = y_pred[i]
        truth = float(y_true[i])

        edges = np.histogram_bin_edges(preds, bins=bins, range=(low_bin, high_bin))
        
        # Predicted bins
        binned_preds = digitize_minus_one(preds, edges)
        # Observed bin
        binned_obs = digitize_minus_one([truth], edges)[0]

        # Laplace smoothing (add 1 to each bin)
        n_bins = len(edges) - 1
        bin_counts = np.bincount(binned_preds, minlength=n_bins)
        smoothed_counts = bin_counts + 1
        
        prob = smoothed_counts[binned_obs] / np.sum(smoothed_counts)
        scores.append(-np.log2(prob))

    return np.mean(scores)

# Dispatch dicts
REGRESSION_POINT_NATIVE = {
    "MSE":       calculate_mse_native,
    "MSLE":      calculate_msle_native,
    "RMSLE":     calculate_rmsle_native,
    "EMD":       calculate_emd_native,
    "Pearson":   calculate_pearson_native,
    "MTD":       calculate_mtd_native,
    "y_hat_bar": calculate_mean_prediction_native,
}

REGRESSION_SAMPLE_NATIVE = {
    "CRPS":      calculate_crps_native,
    "MIS":       calculate_mean_interval_score_native,
    "Coverage":  calculate_coverage_native,
    "Ignorance": calculate_ignorance_score_native,
    "y_hat_bar": calculate_mean_prediction_native,
}

CLASSIFICATION_POINT_NATIVE = {
    "AP": calculate_ap_native,
}

CLASSIFICATION_SAMPLE_NATIVE = {
    "CRPS": calculate_crps_native,
}
