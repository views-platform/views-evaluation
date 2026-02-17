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
    threshold=25,
) -> float:
    """
    Calculate Average Precision (AP) for binary predictions with a threshold.

    Args:
        matched_actual (pd.DataFrame): DataFrame containing actual values
        matched_pred (pd.DataFrame): DataFrame containing predictions
        target (str): The target column name
        threshold (float): Threshold to convert predictions to binary values

    Returns:
        float: Average Precision score
    """
    actual_values = np.concatenate(matched_actual[target].values)
    pred_values = np.concatenate(matched_pred[f"pred_{target}"].values)

    actual_expanded = np.repeat(
        actual_values, [len(x) for x in matched_pred[f"pred_{target}"]]
    )

    actual_binary = (actual_expanded > threshold).astype(int)
    pred_binary = (pred_values >= threshold).astype(int)

    return average_precision_score(actual_binary, pred_binary)


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

    # Clip predictions to avoid zero values (MTD with power > 1 requires positive predictions)
    pred_values = np.clip(pred_values, 1e-4, None)

    return mean_tweedie_deviance(actual_expanded, pred_values, power=power)


def calculate_mean_prediction(
    matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str
) -> float:
    """
    Calculate the mean prediction.
    """
    all_preds = np.concatenate([np.asarray(v).flatten() for v in matched_pred[f"pred_{target}"]])
    return np.mean(all_preds)


def calculate_level_ratio(
    matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str
) -> float:
    """
    Calculate the Global Level Ratio between predictions and actuals.

    The level ratio is defined as:
        R_level = mean(predictions) / mean(actuals)

    This metric detects systematic under-prediction, where a model achieves apparent
    improvements in MSLE by shrinking forecasts toward a low, near-constant level.
    In sparse, zero-inflated settings, such shrinkage can reduce average error without
    reflecting meaningful information about underlying risk.

    A model is considered disqualified if R_level < 0.90, indicating excessive shrinkage.

    Args:
        matched_actual (pd.DataFrame): DataFrame containing actual values with the target column.
        matched_pred (pd.DataFrame): DataFrame containing predictions with the `pred_{target}` column.
        target (str): The target column name (without the 'pred_' prefix).

    Returns:
        float: The level ratio. Values < 0.90 indicate disqualification-level under-prediction.
            Values close to 1.0 indicate good level consistency.

    Reference:
        Colaresi, M. (2026). Level consistency constraint for conflict forecasting evaluation.
    """
    actual_values = np.concatenate(matched_actual[target].values)
    pred_values = np.concatenate(matched_pred[f"pred_{target}"].values)

    actual_expanded = np.repeat(
        actual_values, [len(x) for x in matched_pred[f"pred_{target}"]]
    )

    y_bar = np.mean(actual_expanded)
    y_hat_bar = np.mean(pred_values)

    # Avoid division by zero
    if y_bar == 0:
        return np.inf if y_hat_bar > 0 else 1.0

    return y_hat_bar / y_bar


def calculate_baseline_deviation(
    matched_actual: pd.DataFrame,
    matched_pred: pd.DataFrame,
    target: str,
    baseline: pd.DataFrame = None,
    epsilon: float = 1e-8,
) -> float:
    """
    Calculate the Normalized Deviation from No-Change Baseline.

    This metric measures how different a model's predictions are from a no-change
    baseline. Models that produce forecasts indistinguishable from baseline are
    considered baseline-equivalent and not eligible for production use.

    The deviation is defined as:
        D(M) = mean(|y_hat_M - y_hat_NC|) / (mean(|y_hat_NC|) + epsilon)

    where y_hat_M are model predictions and y_hat_NC are no-change baseline predictions.

    A model is disqualified if D(M) < tau (threshold), indicating it extracts no
    meaningful temporal information beyond persistence.

    Args:
        matched_actual (pd.DataFrame): DataFrame containing actual values (unused but
            required for API consistency).
        matched_pred (pd.DataFrame): DataFrame containing model predictions with
            the `pred_{target}` column.
        target (str): The target column name (without the 'pred_' prefix).
        baseline (pd.DataFrame): DataFrame containing no-change baseline predictions
            with the `pred_{target}` column. If None, returns NaN.
        epsilon (float): Small constant for numerical stability. Default 1e-8.

    Returns:
        float: The normalized deviation from baseline. Values close to 0 indicate
            the model is baseline-equivalent. Higher values indicate more deviation.
            Returns NaN if baseline is not provided.

    Reference:
        Colaresi, M. (2026). Baseline equivalence constraint for conflict forecasting.
    """
    if baseline is None:
        return np.nan

    pred_values = np.concatenate(matched_pred[f"pred_{target}"].values)
    baseline_values = np.concatenate(baseline[f"pred_{target}"].values)

    # Ensure same length
    if len(pred_values) != len(baseline_values):
        raise ValueError(
            f"Model predictions ({len(pred_values)}) and baseline predictions "
            f"({len(baseline_values)}) must have the same length."
        )

    # Calculate normalized deviation
    mean_abs_diff = np.mean(np.abs(pred_values - baseline_values))
    mean_abs_baseline = np.mean(np.abs(baseline_values))

    return mean_abs_diff / (mean_abs_baseline + epsilon)


def calculate_bcd(
    matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str, power: float = 1.5
) -> float:
    """
    Calculate Balanced Conflict Deviation (BCD) between actual and predicted values.

    BCD is defined as the geometric mean of Mean Tweedie Deviance (MTD),
    Mean Squared Logarithmic Error (MSLE), and log(1 + MSE):
        BCD = (MTD * MSLE * log(1 + MSE))^(1/3)

    This metric combines the strengths of MTD, MSLE, and MSE:
        - MTD (with power=1.5) is well-suited for zero-inflated positive continuous data
          typical in conflict forecasting
        - MSLE penalizes underestimates more than overestimates and is scale-independent
        - log(1 + MSE) incorporates absolute error magnitude in a bounded, scale-friendly way

    The geometric mean ensures that all components contribute equally on a multiplicative
    scale, making BCD robust to cases where one metric might dominate the others.

    Lower values indicate better model performance.

    Args:
        matched_actual (pd.DataFrame): DataFrame containing actual values with the target column.
        matched_pred (pd.DataFrame): DataFrame containing predictions with the `pred_{target}` column.
        target (str): The target column name (without the 'pred_' prefix).
        power (float): The power parameter for the Tweedie distribution used in MTD calculation.
            Default is 1.5 (compound Poisson-Gamma distribution).

    Returns:
        float: The Balanced Conflict Deviation score. Lower values indicate better predictions.

    See Also:
        - calculate_mtd: Mean Tweedie Deviance component.
        - calculate_msle: Mean Squared Logarithmic Error component.
        - calculate_mse: Mean Squared Error component (used as log(1 + MSE)).
    """
    mtd = calculate_mtd(matched_actual, matched_pred, target, power=power)
    msle = calculate_msle(matched_actual, matched_pred, target)
    mse = calculate_mse(matched_actual, matched_pred, target)
    return np.cbrt(mtd * msle * np.log(1 + mse))


POINT_METRIC_FUNCTIONS = {
    "MSE": calculate_mse,
    "MSLE": calculate_msle,
    "RMSLE": calculate_rmsle,
    "CRPS": calculate_crps,
    "AP": calculate_ap,
    "EMD": calculate_emd,
    "SD": calculate_sd,
    "pEMDiv": calculate_pEMDiv,
    "Pearson": calculate_pearson,
    "Variogram": calculate_variogram,
    "MTD": calculate_mtd,
    "BCD": calculate_bcd,
    "LevelRatio": calculate_level_ratio,
    "BaselineDeviation": calculate_baseline_deviation,
    "y_hat_bar": calculate_mean_prediction,
}

UNCERTAINTY_METRIC_FUNCTIONS = {
    "CRPS": calculate_crps,
    "MIS": calculate_mean_interval_score,
    "Ignorance": calculate_ignorance_score,
    "Brier": calculate_brier,
    "Jeffreys": calculate_jeffreys,
    "Coverage": calculate_coverage,
    "pEMDiv": calculate_pEMDiv,
    "y_hat_bar": calculate_mean_prediction,
}
