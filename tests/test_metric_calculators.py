import pytest
import pandas as pd
from views_evaluation.evaluation.native_metric_calculators import (
    calculate_mse_native,
    calculate_rmsle_native,
    calculate_crps_native,
    calculate_ap_native,
    calculate_emd_native,
    calculate_pearson_native,
    calculate_coverage_native,
    calculate_ignorance_score_native,
    calculate_mean_interval_score_native,
    calculate_mtd_native,
    REGRESSION_POINT_NATIVE,
    REGRESSION_SAMPLE_NATIVE,
    REGRESSION_POINT_NATIVE,
    REGRESSION_SAMPLE_NATIVE,
    CLASSIFICATION_POINT_NATIVE,
    CLASSIFICATION_SAMPLE_NATIVE,
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    actual = pd.DataFrame({
        'target': [[1.0], [2.0], [3.0], [4.0]]
    })
    pred = pd.DataFrame({
        'pred_target': [[1.1], [1.9], [3.1], [3.9]]
    })
    return actual, pred


@pytest.fixture
def sample_sample_data():
    """Create sample sample data for testing."""
    actual = pd.DataFrame({
        'target': [[1.0], [2.0], [3.0], [4.0]]
    })
    pred = pd.DataFrame({
        'pred_target': [[1.0, 1.1, 1.2], [1.8, 2.0, 2.2], [2.9, 3.0, 3.1], [3.8, 4.0, 4.2]]
    })
    return actual, pred


def test_calculate_mse_native(sample_data):
    """Test MSE calculation."""
    actual, pred = sample_data
    result = calculate_mse_native(actual, pred, 'target')
    assert isinstance(result, float)
    assert result >= 0

def test_calculate_rmsle_native_point(sample_data):
    """Test RMSLE calculation."""
    actual, pred = sample_data
    result = calculate_rmsle_native(actual, pred, 'target')
    assert isinstance(result, float)
    assert result >= 0

def test_calculate_crps_native_point(sample_data):
    """Test CRPS calculation."""
    actual, pred = sample_data
    result = calculate_crps_native(actual, pred, 'target')
    assert isinstance(result, float)
    assert result >= 0


def test_calculate_crps_native_sample(sample_sample_data):
    """Test CRPS calculation."""
    actual, pred = sample_sample_data
    result = calculate_crps_native(actual, pred, 'target')
    assert isinstance(result, float)
    assert result >= 0


def test_calculate_ap_native():
    """Test Average Precision calculation with pre-binarised actuals and probability scores."""
    # Binary actuals (0/1) and probability scores as predictions
    actual = pd.DataFrame({'target': [[1], [0], [1], [0]]})
    pred = pd.DataFrame({'pred_target': [[0.9], [0.4], [0.3], [0.1]]})
    result = calculate_ap_native(actual, pred, 'target')
    assert isinstance(result, float)
    assert 0 <= result <= 1


def test_calculate_emd_native(sample_data):
    """Test Earth Mover's Distance calculation."""
    actual, pred = sample_data
    result = calculate_emd_native(actual, pred, 'target')
    assert isinstance(result, float)
    assert result >= 0


def test_calculate_pearson_native(sample_data):
    """Test Pearson correlation calculation."""
    actual, pred = sample_data
    result = calculate_pearson_native(actual, pred, 'target')
    assert isinstance(result, float)
    assert -1 <= result <= 1


def test_calculate_mtd_native(sample_data):
    """Test Mean Tweedie Deviance calculation."""
    actual, pred = sample_data
    result = calculate_mtd_native(actual, pred, 'target')
    assert isinstance(result, float)
    assert result >= 0


def test_calculate_mtd_native_with_power(sample_data):
    """Test Mean Tweedie Deviance calculation with different power values."""
    actual, pred = sample_data
    # Test with power=1.5 (compound Poisson-Gamma)
    result_15 = calculate_mtd_native(actual, pred, 'target', power=1.5)
    assert isinstance(result_15, float)
    assert result_15 >= 0

    # Test with power=2 (Gamma)
    result_2 = calculate_mtd_native(actual, pred, 'target', power=2.0)
    assert isinstance(result_2, float)
    assert result_2 >= 0


def test_calculate_coverage_native_sample(sample_sample_data):
    """Test Coverage calculation."""
    actual, pred = sample_sample_data
    result = calculate_coverage_native(actual, pred, 'target')
    assert isinstance(result, float)
    assert 0 <= result <= 1


def test_calculate_ignorance_score_native_sample(sample_sample_data):
    """Test Ignorance Score calculation."""
    actual, pred = sample_sample_data
    result = calculate_ignorance_score_native(actual, pred, 'target')
    assert isinstance(result, float)
    assert result >= 0


def test_calculate_mis_sample(sample_sample_data):
    """Test Mean Interval Score calculation."""
    actual, pred = sample_sample_data
    result = calculate_mean_interval_score_native(actual, pred, 'target')
    assert isinstance(result, float)
    assert result >= 0


def test_point_metric_functions():
    """Test that all point metric functions are available in the deprecated REGRESSION_POINT_NATIVE."""
    expected_metrics = [
        "MSE", "MSLE", "RMSLE", "EMD", "SD", "pEMDiv", "Pearson", "Variogram", "MTD", "y_hat_bar"
    ]


    for metric in expected_metrics:
        assert metric in REGRESSION_POINT_NATIVE
        assert callable(REGRESSION_POINT_NATIVE[metric])


def test_sample_metric_functions():
    """Test that all sample metric functions are available in the deprecated REGRESSION_SAMPLE_NATIVE."""
    expected_metrics = ["CRPS", "MIS", "Ignorance", "Brier", "Jeffreys", "Coverage"]

    for metric in expected_metrics:
        assert metric in REGRESSION_SAMPLE_NATIVE
        assert callable(REGRESSION_SAMPLE_NATIVE[metric])


def test_regression_point_metric_functions():
    """Test that all regression point metric functions are available in REGRESSION_POINT_NATIVE."""
    expected_metrics = ["MSE", "MSLE", "RMSLE", "EMD", "SD", "pEMDiv", "Pearson", "Variogram", "MTD", "y_hat_bar"]

    for metric in expected_metrics:
        assert metric in REGRESSION_POINT_NATIVE
        assert callable(REGRESSION_POINT_NATIVE[metric])

    # AP must NOT be in regression point functions
    assert "AP" not in REGRESSION_POINT_NATIVE
    # CRPS must NOT be in regression point functions
    assert "CRPS" not in REGRESSION_POINT_NATIVE


def test_regression_sample_metric_functions():
    """Test that all regression sample metric functions are available."""
    expected_metrics = ["CRPS", "MIS", "Coverage", "Ignorance", "y_hat_bar"]

    for metric in expected_metrics:
        assert metric in REGRESSION_SAMPLE_NATIVE
        assert callable(REGRESSION_SAMPLE_NATIVE[metric])

    # AP must NOT be in regression sample functions
    assert "AP" not in REGRESSION_SAMPLE_NATIVE


def test_classification_point_metric_functions():
    """Test that AP is in CLASSIFICATION_POINT_NATIVE."""
    assert "AP" in CLASSIFICATION_POINT_NATIVE
    assert callable(CLASSIFICATION_POINT_NATIVE["AP"])

    # RMSLE must NOT be in classification point functions
    assert "RMSLE" not in CLASSIFICATION_POINT_NATIVE


def test_classification_sample_metric_functions():
    """Test that classification sample metric functions are available."""
    expected_metrics = ["CRPS", "Brier", "Jeffreys"]

    for metric in expected_metrics:
        assert metric in CLASSIFICATION_SAMPLE_NATIVE
        assert callable(CLASSIFICATION_SAMPLE_NATIVE[metric])

    # RMSLE must NOT be in classification sample functions
    assert "RMSLE" not in CLASSIFICATION_SAMPLE_NATIVE


def test_not_implemented_metrics():
    """Test that unimplemented metrics raise NotImplementedError."""
    actual = pd.DataFrame({'target': [[1.0]]})
    pred = pd.DataFrame({'pred_target': [[1.0]]})

    from views_evaluation.evaluation.native_metric_calculators import (
        calculate_brier_native,
        calculate_jeffreys_native,
        calculate_sd_native,
        calculate_pEMDiv_native,
        calculate_variogram_native,
    )

    unimplemented_functions = [
        calculate_brier_native,
        calculate_jeffreys_native,
        calculate_sd_native,
        calculate_pEMDiv_native,
        calculate_variogram_native,
    ]

    for func in unimplemented_functions:
        with pytest.raises(NotImplementedError):
            func(actual, pred, 'target')
