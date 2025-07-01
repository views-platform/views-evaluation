import pytest
import pandas as pd
import numpy as np
from views_evaluation.evaluation.metric_calculators import (
    calculate_rmsle,
    calculate_crps,
    calculate_ap,
    calculate_emd,
    calculate_pearson,
    calculate_variogram,
    calculate_ignorance_score,
    calculate_mean_interval_score,
    POINT_METRIC_FUNCTIONS,
    UNCERTAINTY_METRIC_FUNCTIONS,
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
def sample_uncertainty_data():
    """Create sample uncertainty data for testing."""
    actual = pd.DataFrame({
        'target': [[1.0], [2.0], [3.0], [4.0]]
    })
    pred = pd.DataFrame({
        'pred_target': [[1.0, 1.1, 1.2], [1.8, 2.0, 2.2], [2.9, 3.0, 3.1], [3.8, 4.0, 4.2]]
    })
    return actual, pred


def test_calculate_rmsle(sample_data):
    """Test RMSLE calculation."""
    actual, pred = sample_data
    result = calculate_rmsle(actual, pred, 'target')
    assert isinstance(result, float)
    assert result >= 0


def test_calculate_crps(sample_uncertainty_data):
    """Test CRPS calculation."""
    actual, pred = sample_uncertainty_data
    result = calculate_crps(actual, pred, 'target')
    assert isinstance(result, float)
    assert result >= 0


def test_calculate_ap(sample_data):
    """Test Average Precision calculation."""
    actual, pred = sample_data
    result = calculate_ap(actual, pred, 'target', threshold=2.5)
    assert isinstance(result, float)
    assert 0 <= result <= 1


def test_calculate_emd(sample_data):
    """Test Earth Mover's Distance calculation."""
    actual, pred = sample_data
    result = calculate_emd(actual, pred, 'target')
    assert isinstance(result, float)
    assert result >= 0


def test_calculate_pearson(sample_data):
    """Test Pearson correlation calculation."""
    actual, pred = sample_data
    result = calculate_pearson(actual, pred, 'target')
    assert isinstance(result, float)
    assert -1 <= result <= 1


def test_calculate_variogram(sample_data):
    """Test Variogram calculation."""
    actual, pred = sample_data
    result = calculate_variogram(actual, pred, 'target')
    assert isinstance(result, float)
    assert result >= 0


def test_calculate_ignorance_score(sample_uncertainty_data):
    """Test Ignorance Score calculation."""
    actual, pred = sample_uncertainty_data
    result = calculate_ignorance_score(actual, pred, 'target')
    assert isinstance(result, float)
    assert result >= 0


def test_calculate_mis(sample_uncertainty_data):
    """Test Mean Interval Score calculation."""
    actual, pred = sample_uncertainty_data
    result = calculate_mean_interval_score(actual, pred, 'target')
    assert isinstance(result, float)
    assert result >= 0


def test_point_metric_functions():
    """Test that all point metric functions are available."""
    expected_metrics = [
        "RMSLE", "CRPS", "AP", "Brier", "Jeffreys", 
        "Coverage", "EMD", "SD", "pEMDiv", "Pearson", "Variogram"
    ]
    
    for metric in expected_metrics:
        assert metric in POINT_METRIC_FUNCTIONS
        assert callable(POINT_METRIC_FUNCTIONS[metric])


def test_uncertainty_metric_functions():
    """Test that all uncertainty metric functions are available."""
    expected_metrics = ["CRPS"]
    
    for metric in expected_metrics:
        assert metric in UNCERTAINTY_METRIC_FUNCTIONS
        assert callable(UNCERTAINTY_METRIC_FUNCTIONS[metric])


def test_not_implemented_metrics():
    """Test that unimplemented metrics raise NotImplementedError."""
    actual = pd.DataFrame({'target': [[1.0]]})
    pred = pd.DataFrame({'pred_target': [[1.0]]})
    
    from views_evaluation.evaluation.metric_calculators import (
        calculate_brier,
        calculate_jeffreys,
        calculate_coverage,
        calculate_sd,
        calculate_pEMDiv,
    )
    
    unimplemented_functions = [
        calculate_brier,
        calculate_jeffreys,
        calculate_coverage,
        calculate_sd,
        calculate_pEMDiv,
    ]
    
    for func in unimplemented_functions:
        with pytest.raises(NotImplementedError):
            func(actual, pred, 'target') 