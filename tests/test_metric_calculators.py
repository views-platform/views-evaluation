import pytest
import pandas as pd
import numpy as np
import properscoring as ps
from sklearn.metrics import mean_squared_error, root_mean_squared_log_error, average_precision_score
from scipy.stats import wasserstein_distance, pearsonr

# Import all required functions
from views_evaluation.evaluation.metric_calculators import (
    calculate_mse,
    calculate_rmsle,
    calculate_crps,
    calculate_ap,
    calculate_emd,
    calculate_pearson,
    calculate_coverage,
    calculate_ignorance_score,
    calculate_mean_interval_score,
    POINT_METRIC_FUNCTIONS,
    UNCERTAINTY_METRIC_FUNCTIONS,
)

# Set tolerance for floating point comparisons
TOLERANCE = 1e-6


@pytest.fixture
def sample_data():
    """Create sample point data for testing (arrays of length 1)."""
    actual = pd.DataFrame({
        'target': [[1.0], [2.0], [3.0], [4.0]]
    })
    pred = pd.DataFrame({
        'pred_target': [[1.1], [1.9], [3.1], [3.9]]
    })
    return actual, pred


@pytest.fixture
def sample_uncertainty_data():
    """Create sample uncertainty data for testing (arrays of length > 1)."""
    actual = pd.DataFrame({
        'target': [[1.0], [2.0], [3.0], [4.0]]
    })
    pred = pd.DataFrame({
        'pred_target': [
            [1.0, 1.1, 1.2],    # Mean = 1.1
            [1.8, 2.0, 2.2],    # Mean = 2.0
            [2.9, 3.0, 3.1],    # Mean = 3.0
            [3.8, 4.0, 4.2]     # Mean = 4.0
        ]
    })
    return actual, pred


@pytest.fixture
def sample_ap_data():
    """Create data tailored for AP thresholding logic (threshold=30)."""
    actual = pd.DataFrame({
        'target': [[40], [20], [35], [25]]
    })
    pred = pd.DataFrame({
        'pred_target': [[35], [30], [20], [15]]
    })
    return actual, pred


# --- NUMERICAL VALUE TESTS ---

def test_calculate_mse(sample_data):
    """Test MSE calculation against sklearn."""
    actual, pred = sample_data
    result = calculate_mse(actual, pred, 'target')
    
    # Expected: (0.1^2 + (-0.1)^2 + 0.1^2 + (-0.1)^2) / 4 = 0.04 / 4 = 0.01
    actual_flat = [1.0, 2.0, 3.0, 4.0]
    pred_flat = [1.1, 1.9, 3.1, 3.9]
    expected = mean_squared_error(actual_flat, pred_flat)

    assert np.allclose(result, expected, atol=TOLERANCE)


def test_calculate_rmsle_point(sample_data):
    """Test RMSLE calculation against sklearn."""
    actual, pred = sample_data
    result = calculate_rmsle(actual, pred, 'target')
    
    actual_flat = [1.0, 2.0, 3.0, 4.0]
    pred_flat = [1.1, 1.9, 3.1, 3.9]
    expected = root_mean_squared_log_error(actual_flat, pred_flat)

    assert np.allclose(result, expected, atol=TOLERANCE)


def test_calculate_crps_point(sample_data):
    """Test CRPS calculation (treats scalar prediction as a single ensemble member)."""
    actual, pred = sample_data
    result = calculate_crps(actual, pred, 'target')
    
    actual_flat = [1.0, 2.0, 3.0, 4.0]
    pred_flat = [[1.1], [1.9], [3.1], [3.9]] # Must pass as list of lists to ps.crps_ensemble
    expected = np.mean([ps.crps_ensemble(a, np.array(p)) for a, p in zip(actual_flat, pred_flat)])

    assert np.allclose(result, expected, atol=TOLERANCE)


def test_calculate_crps_uncertainty(sample_uncertainty_data):
    """Test CRPS calculation with proper ensembles."""
    actual, pred = sample_uncertainty_data
    result = calculate_crps(actual, pred, 'target')

    # Expected calculation uses the ensemble
    actual_flat = [1.0, 2.0, 3.0, 4.0]
    pred_ensembles = pred['pred_target'].tolist()
    expected = np.mean([ps.crps_ensemble(a, np.array(p)) for a, p in zip(actual_flat, pred_ensembles)])

    assert np.allclose(result, expected, atol=TOLERANCE)


def test_calculate_ap_point_predictions():
    actual_data = {'target': [[40], [20], [35], [25]]}
    pred_data = {'pred_target': [[35], [30], [20], [15]]}
    threshold=30
    
    matched_actual = pd.DataFrame(actual_data)
    matched_pred = pd.DataFrame(pred_data)
    
    from views_evaluation.evaluation.metric_calculators import calculate_ap
    ap_score = calculate_ap(matched_actual, matched_pred, 'target', threshold)
    
    actual_binary = [1, 0, 1, 0]  # 40>30, 20<30, 35>30, 25<30
    from sklearn.metrics import average_precision_score
    expected_ap = average_precision_score(actual_binary, pred_data['pred_target'])
    
    assert abs(ap_score - expected_ap) < 0.01


def test_calculate_ap_uncertainty_predictions():
    actual_data = {'target': [[40], [20], [35], [25]]}
    pred_data = {
        'pred_target': [
            [35, 40, 45],
            [30, 35, 40],
            [20, 25, 30],
            [15, 20, 25]
        ]
    }
    threshold=30
    matched_actual = pd.DataFrame(actual_data)
    matched_pred = pd.DataFrame(pred_data)
    
    from views_evaluation.evaluation.metric_calculators import calculate_ap
    ap_score = calculate_ap(matched_actual, matched_pred, 'target', threshold)
    
    pred_values = [35, 40, 45, 30, 35, 40, 20, 25, 30, 15, 20, 25]
    actual_values = [40, 40, 40, 20, 20, 20, 35, 35, 35, 25, 25, 25]
    actual_binary = [1 if x > threshold else 0 for x in actual_values]

    from sklearn.metrics import average_precision_score
    expected_ap = average_precision_score(actual_binary, pred_values)
    
    assert abs(ap_score - expected_ap) < 0.01


def test_calculate_emd(sample_uncertainty_data):
    """Test Earth Mover's Distance calculation against scipy's wasserstein_distance."""
    actual, pred = sample_uncertainty_data
    result = calculate_emd(actual, pred, 'target')

    # EMD is calculated per row and averaged
    actuals = [1.0, 2.0, 3.0, 4.0]
    preds = pred['pred_target'].tolist()
    
    # We must treat the actual scalar as a distribution (a delta function) for EMD.
    # The actual implementation of EMD uses a simpler comparison between the prediction
    # distribution and the actual scalar value, which simplifies to:
    expected_list = [
        wasserstein_distance([1.0, 1.1, 1.2], [1.0]),
        wasserstein_distance([1.8, 2.0, 2.2], [2.0]),
        wasserstein_distance([2.9, 3.0, 3.1], [3.0]),
        wasserstein_distance([3.8, 4.0, 4.2], [4.0]),
    ]
    expected = np.mean(expected_list)
    
    assert np.allclose(result, expected, atol=TOLERANCE)


def test_calculate_pearson(sample_data):
    """Test Pearson correlation calculation against scipy."""
    actual, pred = sample_data
    result = calculate_pearson(actual, pred, 'target')

    actual_flat = [1.0, 2.0, 3.0, 4.0]
    pred_flat = [1.1, 1.9, 3.1, 3.9]
    expected, _ = pearsonr(actual_flat, pred_flat)

    assert np.allclose(result, expected, atol=TOLERANCE)


def test_calculate_coverage_uncertainty(sample_uncertainty_data):
    """Test Coverage calculation (using default alpha=0.1 -> 90% interval)."""
    actual, pred = sample_uncertainty_data
    result = calculate_coverage(actual, pred, 'target', alpha=0.1)

    # For 3 samples, the 5th and 95th percentiles (alpha=0.1) are not directly available.
    # The quantiles used are Q_0.05 and Q_0.95.
    # Q_0.05: Lower bound (e.g., 1.0)
    # Q_0.95: Upper bound (e.g., 1.2)
    # For 3 samples, quantiles often fall on the min and max.
    
    actuals = [1.0, 2.0, 3.0, 4.0]
    preds = pred['pred_target'].tolist()

    covered = []
    for a, p in zip(actuals, preds):
        lower = np.quantile(p, 0.05)
        upper = np.quantile(p, 0.95)
        covered.append(lower <= a <= upper)
    
    expected = np.mean(covered)
    assert np.allclose(result, expected, atol=TOLERANCE)


def test_calculate_mis_uncertainty(sample_uncertainty_data):
    """Test Mean Interval Score calculation (using default alpha=0.05 -> 95% interval)."""
    actual, pred = sample_uncertainty_data
    result = calculate_mean_interval_score(actual, pred, 'target', alpha=0.05)
    
    alpha = 0.05
    actuals = np.array([1.0, 2.0, 3.0, 4.0])
    preds = pred['pred_target'].tolist()
    
    lower = np.array([np.quantile(row, q=alpha / 2) for row in preds]) # Q_0.025
    upper = np.array([np.quantile(row, q=1 - (alpha / 2)) for row in preds]) # Q_0.975

    interval_width = upper - lower
    lower_coverage = (2 / alpha) * (lower - actuals) * (actuals < lower)
    upper_coverage = (2 / alpha) * (actuals - upper) * (actuals > upper)
    
    expected = np.mean(interval_width + lower_coverage + upper_coverage)

    assert np.allclose(result, expected, atol=TOLERANCE)


def test_calculate_ignorance_score_uncertainty(sample_uncertainty_data):
    """Test Ignorance Score calculation (checks execution, value depends heavily on binning)."""
    actual, pred = sample_uncertainty_data
    result = calculate_ignorance_score(actual, pred, 'target')
    
    # We assert the output format and type, as asserting the exact value without 
    # copying the complex binning logic is prone to error.
    assert isinstance(result, float)
    assert result >= 0
    # A regression check (asserting against a known historical result) would be better here.


# --- STRUCTURE/INTEGRATION TESTS ---

def test_point_metric_functions_completeness():
    """Test that all expected point metric functions are present and callable."""
    # Note: SD, pEMDiv, Variogram are expected placeholders.
    expected_metrics = [
        "MSE", "MSLE", "RMSLE", "CRPS", "AP", "EMD", "SD", "pEMDiv", "Pearson", "Variogram", "y_hat_bar"
    ]
    
    for metric in expected_metrics:
        assert metric in POINT_METRIC_FUNCTIONS
        assert callable(POINT_METRIC_FUNCTIONS[metric])


def test_uncertainty_metric_functions_completeness():
    """Test that all expected uncertainty metric functions are present and callable."""
    expected_metrics = ["CRPS", "MIS", "Ignorance", "Brier", "Jeffreys", "Coverage", "pEMDiv", "y_hat_bar"]
    
    for metric in expected_metrics:
        assert metric in UNCERTAINTY_METRIC_FUNCTIONS
        assert callable(UNCERTAINTY_METRIC_FUNCTIONS[metric])


def test_not_implemented_metrics():
    """Test that unimplemented metrics correctly raise NotImplementedError."""
    actual = pd.DataFrame({'target': [[1.0]]})
    pred = pd.DataFrame({'pred_target': [[1.0]]})
    
    from views_evaluation.evaluation.metric_calculators import (
        calculate_brier,
        calculate_jeffreys,
        calculate_sd,
        calculate_pEMDiv,
        calculate_variogram,
    )
    
    unimplemented_functions = [
        calculate_brier,
        calculate_jeffreys,
        calculate_sd,
        calculate_pEMDiv,
        calculate_variogram,
    ]
    
    for func in unimplemented_functions:
        with pytest.raises(NotImplementedError):
            func(actual, pred, 'target')