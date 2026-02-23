from collections import Counter
import pandas as pd
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


def calculate_mse(
    matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str
) -> float:
    """
    Calculate Mean Square Error for each prediction.

    Args:
        matched_actual (pd.DataFrame): DataFrame containing actual values
        matched_pred (pd.DataFrame): DataFrame containing predictions
        target (str): The target column name

    Returns:
        float: Average MSE score
    """
    actual_values = np.concatenate(matched_actual[target].values)
    pred_values = np.concatenate(matched_pred[f"pred_{target}"].values)

    actual_expanded = np.repeat(
        actual_values, [len(x) for x in matched_pred[f"pred_{target}"]]
    )

    return mean_squared_error(actual_expanded, pred_values)


def calculate_msle(
    matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str
) -> float:
    """
    Calculate Mean Squared Logarithmic Error (MSLE) for each prediction.

    Args:
        matched_actual (pd.DataFrame): DataFrame containing actual values
        matched_pred (pd.DataFrame): DataFrame containing predictions
        target (str): The target column name

    Returns:
        float: Average MSLE score
    """
    actual_values = np.concatenate(matched_actual[target].values)
    pred_values = np.concatenate(matched_pred[f"pred_{target}"].values)
    actual_expanded = np.repeat(
        actual_values, [len(x) for x in matched_pred[f"pred_{target}"]]
    )
    return mean_squared_log_error(actual_expanded, pred_values)


def calculate_rmsle(
    matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str
) -> float:
    """
    Calculate Root Mean Squared Logarithmic Error (RMSLE) for each prediction.

    Args:
        matched_actual (pd.DataFrame): DataFrame containing actual values
        matched_pred (pd.DataFrame): DataFrame containing predictions
        target (str): The target column name

    Returns:
        float: Average RMSLE score
    """
    actual_values = np.concatenate(matched_actual[target].values)
    pred_values = np.concatenate(matched_pred[f"pred_{target}"].values)

    actual_expanded = np.repeat(
        actual_values, [len(x) for x in matched_pred[f"pred_{target}"]]
    )

    return root_mean_squared_log_error(actual_expanded, pred_values)


def calculate_crps(
    matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str
) -> float:
    """
    Calculate Continuous Ranked Probability Score (CRPS) for each prediction.

    Args:
        matched_actual (pd.DataFrame): DataFrame containing actual values
        matched_pred (pd.DataFrame): DataFrame containing predictions
        target (str): The target column name

    Returns:
        float: Average CRPS score
    """
    return np.mean(
        [
            ps.crps_ensemble(actual[0], np.array(pred))
            for actual, pred in zip(
                matched_actual[target], matched_pred[f"pred_{target}"]
            )
        ]
    )


def calculate_ap(
    matched_actual: pd.DataFrame,
    matched_pred: pd.DataFrame,
    target: str,
) -> float:
    """
    Calculate Average Precision (AP) for classification targets.

    Actuals must be pre-binarised (0/1) by the model pipeline before reaching
    the evaluator. Predictions should be probability scores in [0, 1].
    No thresholding is applied here — that is the model pipeline's responsibility.

    Args:
        matched_actual (pd.DataFrame): DataFrame with binary actual values (0/1)
        matched_pred (pd.DataFrame): DataFrame with prediction probability scores
        target (str): The target column name

    Returns:
        float: Average Precision score
    """
    actual_values = np.concatenate(matched_actual[target].values)
    pred_values = np.concatenate(matched_pred[f"pred_{target}"].values)

    actual_expanded = np.repeat(
        actual_values, [len(x) for x in matched_pred[f"pred_{target}"]]
    )

    return average_precision_score(actual_expanded, pred_values)


def calculate_emd(
    matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str
) -> float:
    """
    Calculate Earth Mover's Distance (EMD) between predicted and actual distributions.
    EMD measures the minimum amount of work needed to transform one distribution into another.

    Args:
        matched_actual (pd.DataFrame): DataFrame containing actual values
        matched_pred (pd.DataFrame): DataFrame containing predictions
        target (str): The target column name

    Returns:
        float: Average EMD score
    """
    emd_list = []
    for actual, preds in zip(matched_actual[target], matched_pred[f"pred_{target}"]):
        actual_val = np.asarray(actual)
        preds_arr = np.asarray(preds)
        emd_list.append(wasserstein_distance(preds_arr, actual_val))
    return np.mean(emd_list)


def calculate_sd(
    matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str
) -> float:
    """
    Calculate Sinkhorn Distance between predicted and actual distributions.

    Sinkhorn Distance is a regularized version of the Earth Mover's Distance
    that is computationally more efficient.

    Args:
        matched_actual (pd.DataFrame): DataFrame containing actual values
        matched_pred (pd.DataFrame): DataFrame containing predictions
        target (str): The target column name

    Returns:
        float: Sinkhorn Distance score
    """
    raise NotImplementedError("Sinkhorn Distance calculation not yet implemented")


def calculate_pEMDiv(
    matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str
) -> float:
    """
    Calculate pseudo-Earth Mover Divergence between predicted and actual distributions.

    pEMDiv is a computationally efficient approximation of the Earth Mover's Distance.

    Args:
        matched_actual (pd.DataFrame): DataFrame containing actual values
        matched_pred (pd.DataFrame): DataFrame containing predictions
        target (str): The target column name

    Returns:
        float: pEMDiv score
    """
    raise NotImplementedError("pEMDiv calculation not yet implemented")


def calculate_pearson(
    matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str
) -> float:
    """
    Calculate Pearson correlation coefficient between actual and predicted values.
    This measures the linear correlation between predictions and actual values.

    Args:
        matched_actual (pd.DataFrame): DataFrame containing actual values
        matched_pred (pd.DataFrame): DataFrame containing predictions
        target (str): The target column name

    Returns:
        float: Pearson correlation coefficient
    """
    actual_values = np.concatenate(matched_actual[target].values)
    pred_values = np.concatenate(matched_pred[f"pred_{target}"].values)

    actual_expanded = np.repeat(
        actual_values, [len(x) for x in matched_pred[f"pred_{target}"]]
    )

    correlation, _ = pearsonr(actual_expanded, pred_values)
    return correlation


def calculate_variogram(
    matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str
) -> float:
    """
    !! How to accountr for time and location?
    Calculate the variogram score between actual and predicted values.
    This measures the spatial/temporal correlation structure.

    Args:
        matched_actual (pd.DataFrame): DataFrame containing actual values
        matched_pred (pd.DataFrame): DataFrame containing predictions
        target (str): The target column name

    Returns:
        float: Variogram score
    """
    raise NotImplementedError("Variogram calculation not yet implemented")


def calculate_brier(
    matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str
) -> float:
    """
    Calculate Brier Score for probabilistic predictions.

    The Brier Score measures the accuracy of probabilistic predictions.
    Lower values indicate better predictions.

    Args:
        matched_actual (pd.DataFrame): DataFrame containing actual values
        matched_pred (pd.DataFrame): DataFrame containing predictions
        target (str): The target column name

    Returns:
        float: Brier Score
    """
    raise NotImplementedError("Brier Score calculation not yet implemented")


def calculate_jeffreys(
    matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str
) -> float:
    """
    Calculate Jeffreys Divergence between predicted and actual distributions.

    Jeffreys Divergence is a symmetric measure of the difference between
    two probability distributions.

    Args:
        matched_actual (pd.DataFrame): DataFrame containing actual values
        matched_pred (pd.DataFrame): DataFrame containing predictions
        target (str): The target column name

    Returns:
        float: Jeffreys Divergence score
    """
    raise NotImplementedError("Jeffreys Divergence calculation not yet implemented")


def calculate_coverage(
    matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str, alpha=0.1
) -> float:
    """
    Calculate Coverage (Histograms) for probabilistic predictions.

    Coverage measures how well the predicted distribution covers the actual values.

    Args:
        matched_actual (pd.DataFrame): DataFrame containing actual values
        matched_pred (pd.DataFrame): DataFrame containing predictions
        target (str): The target column name
        alpha (float): Significance level for the interval (default: 0.1)
    Returns:
        float: Coverage score
    """
    y_true = matched_actual[target].values
    y_pred_samples = matched_pred[f"pred_{target}"].values
    
    lower_q = alpha / 2
    upper_q = 1 - alpha / 2
    
    covered = []
    for yt, pred_list in zip(y_true, y_pred_samples):
        lower = np.quantile(pred_list, lower_q)
        upper = np.quantile(pred_list, upper_q)
        covered.append(lower <= yt <= upper)
        
    return np.mean(covered)


def calculate_mean_interval_score(
    matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str, alpha=0.05
):
    """
    Calculate the Mean Interval Score (MIS) for probabilistic predictions.

    The Mean Interval Score measures the average width of prediction intervals
    and the coverage of the actual values.

    Args:
        matched_actual (pd.DataFrame): DataFrame containing actual values
        matched_pred (pd.DataFrame): DataFrame containing predictions
        target (str): The target column name
        alpha (float): Significance level for the interval (default: 0.05)

    Returns:
        float: Mean Interval Score
    """
    lower = np.array(
        [np.quantile(row, q=alpha / 2) for row in matched_pred[f"pred_{target}"]]
    )
    upper = np.array(
        [np.quantile(row, q=1 - (alpha / 2)) for row in matched_pred[f"pred_{target}"]]
    )
    actuals = np. array(
        [
            row[0] if isinstance(row, (np.ndarray, list)) else row
            for row in matched_actual[target]
        ]
    )

    interval_width = upper - lower
    lower_coverage = (2 / alpha) * (lower - actuals) * (actuals < lower)
    upper_coverage = (2 / alpha) * (actuals - upper) * (actuals > upper)
    interval_score = interval_width + lower_coverage + upper_coverage

    return np.mean(interval_score)


def calculate_ignorance_score(
    matched_actual: pd.DataFrame,
    matched_pred: pd.DataFrame,
    target: str,
    bins=[0, 0.5, 2.5, 5.5, 10.5, 25.5, 50.5, 100.5, 250.5, 500.5, 1000.5],
    low_bin=0,
    high_bin=10000,
):
    """
    !!Note unfinished. Bins need to be fixed bacause in competition we evaluate over log values but not here.
    This is an adapted version from https://github.com/prio-data/prediction_competition_2023/tree/main
    Compute Binned Ignorance Score for predictions and observations.

    Parameters:
        matched_actual (pd.DataFrame): DataFrame containing actual values
        matched_pred (pd.DataFrame): DataFrame containing predictions
        target (str): The target column name
        bins (list): List of bins for the histogram
        low_bin (float): The lower bound of the bins
        high_bin (float): The upper bound of the bins

    Returns:
        float: Mean ignorance score.
    """

    def digitize_minus_one(x, edges):
        return np.digitize(x, edges, right=False) - 1

    def _calculate_ignorance_score(predictions, observed, n, all_bins):
        # Initialize each bin with 1 (Laplace smoothing)
        c = Counter({bin_idx: 1 for bin_idx in all_bins})
        c.update(predictions)
        prob = c[observed] / sum(c.values())
        return -np.log2(prob)

    scores = []
    for row_p, row_o in zip(matched_pred[f"pred_{target}"], matched_actual[target]):
        preds = np.asarray(row_p)
        truth = float(np.asarray(row_o).squeeze())

        edges = np.histogram_bin_edges(preds, bins=bins, range=(low_bin, high_bin))

        binned_preds = digitize_minus_one(preds, edges)
        binned_obs = digitize_minus_one([truth], edges)[0]

        synthetic = np.arange(len(edges) - 1)
        binned_preds = np.concatenate([binned_preds, synthetic])

        n = len(binned_preds)
        score = _calculate_ignorance_score(binned_preds, binned_obs, n, synthetic)
        scores.append(score)

    return np.mean(scores)


def calculate_mtd(
    matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str, power: float = 1.5
) -> float:
    """
    Calculate Mean Tweedie Deviance (MTD) between actual and predicted values.

    The Tweedie deviance is a family of loss functions parameterized by a power parameter `p`.
    It generalizes several common loss functions:
        - p = 0: Equivalent to Mean Squared Error (Gaussian distribution)
        - p = 1: Equivalent to Poisson deviance (count data)
        - p = 2: Equivalent to Gamma deviance (positive continuous data)
        - 1 < p < 2: Compound Poisson-Gamma distribution (zero-inflated positive continuous data)

    With the default power of 1.5 (compound Poisson-Gamma), this metric is particularly
    well-suited for conflict forecasting data which typically exhibits:
        - Right-skewness (many small values, few large values)
        - Zero-inflation (many observations with zero fatalities)
        - Non-negative continuous outcomes

    The Tweedie deviance for a single observation is defined as:
        d(y, μ) = 2 * (y^(2-p)/((1-p)*(2-p)) - y*μ^(1-p)/(1-p) + μ^(2-p)/(2-p))
    where y is the actual value and μ is the predicted value.

    Lower values indicate better model performance.

    Args:
        matched_actual (pd.DataFrame): DataFrame containing actual values with the target column.
            The target column should contain numpy arrays or lists of actual observations.
        matched_pred (pd.DataFrame): DataFrame containing predictions with the `pred_{target}` column.
            The prediction column should contain numpy arrays or lists of predicted values.
        target (str): The target column name (without the 'pred_' prefix).
        power (float): The power parameter for the Tweedie distribution. Must be in range
            [0, 1) or >= 1. Default is 1.5, which corresponds to the compound Poisson-Gamma
            distribution, ideal for zero-inflated positive continuous data.

    Returns:
        float: The Mean Tweedie Deviance score. Lower values indicate better predictions.

    Raises:
        ValueError: If predictions contain negative values when power > 0, or if
            actual values are negative when power >= 1.

    Example:
        >>> mtd_score = calculate_mtd(actual_df, pred_df, "ln_sb_best")
        >>> print(f"Mean Tweedie Deviance: {mtd_score:.4f}")

    See Also:
        - sklearn.metrics.mean_tweedie_deviance: The underlying implementation.
        - calculate_mse: Mean Squared Error (equivalent to MTD with power=0).
    """
    actual_values = np.concatenate(matched_actual[target].values)
    pred_values = np.concatenate(matched_pred[f"pred_{target}"].values)

    actual_expanded = np.repeat(
        actual_values, [len(x) for x in matched_pred[f"pred_{target}"]]
    )

    return mean_tweedie_deviance(actual_expanded, pred_values, power=power)


def calculate_mean_prediction(
    matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str
) -> float:
    """
    Calculate the mean prediction.
    """
    all_preds = np.concatenate([np.asarray(v).flatten() for v in matched_pred[f"pred_{target}"]])
    return np.mean(all_preds)

REGRESSION_POINT_METRIC_FUNCTIONS = {
    "MSE":       calculate_mse,
    "MSLE":      calculate_msle,
    "RMSLE":     calculate_rmsle,
    "EMD":       calculate_emd,
    "SD":        calculate_sd,         # raises NotImplementedError
    "pEMDiv":    calculate_pEMDiv,     # raises NotImplementedError
    "Pearson":   calculate_pearson,
    "Variogram": calculate_variogram,  # raises NotImplementedError
    "MTD":       calculate_mtd,
    "y_hat_bar": calculate_mean_prediction,
}

REGRESSION_UNCERTAINTY_METRIC_FUNCTIONS = {
    "CRPS":      calculate_crps,
    "MIS":       calculate_mean_interval_score,
    "Coverage":  calculate_coverage,
    "Ignorance": calculate_ignorance_score,
    "y_hat_bar": calculate_mean_prediction,
}

CLASSIFICATION_POINT_METRIC_FUNCTIONS = {
    "AP": calculate_ap,
}

CLASSIFICATION_UNCERTAINTY_METRIC_FUNCTIONS = {
    "CRPS":     calculate_crps,
    "Brier":    calculate_brier,    # raises NotImplementedError
    "Jeffreys": calculate_jeffreys, # raises NotImplementedError
}

# DEPRECATED — will be removed in a future minor version once all callers
# have migrated to the four task-specific dicts above.
POINT_METRIC_FUNCTIONS = {
    **REGRESSION_POINT_METRIC_FUNCTIONS,
    **CLASSIFICATION_POINT_METRIC_FUNCTIONS,
}
UNCERTAINTY_METRIC_FUNCTIONS = {
    **REGRESSION_UNCERTAINTY_METRIC_FUNCTIONS,
    **CLASSIFICATION_UNCERTAINTY_METRIC_FUNCTIONS,
}
