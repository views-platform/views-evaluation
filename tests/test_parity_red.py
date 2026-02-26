import pytest
import pandas as pd
import numpy as np
from views_evaluation.evaluation.evaluation_manager import EvaluationManager
from views_evaluation.adapters.pandas import PandasAdapter
from views_evaluation.evaluation.native_evaluator import NativeEvaluator
from tests.test_parity_green import assert_parity

@pytest.fixture
def red_data_unordered():
    """Predictions are mis-ordered in the DataFrame."""
    index = pd.MultiIndex.from_product([[100, 101], [1, 2]], names=['month', 'unit'])
    actual = pd.DataFrame({'target': [0.1, 0.2, 0.3, 0.4]}, index=index)
    
    # Sequence 0: SHUFFLED rows
    shuffled_idx = index[[3, 0, 2, 1]]
    pred_0 = pd.DataFrame({'pred_target': [0.41, 0.11, 0.31, 0.21]}, index=shuffled_idx)
    
    config = {
        'steps': [1, 2],
        'regression_targets': ['target'],
        'regression_point_metrics': ['MSE']
    }
    
    return actual, [pred_0], "target", config

@pytest.fixture
def red_data_coordinates():
    """Mismatched coordinates (extra units/months)."""
    index_actual = pd.MultiIndex.from_product([[100, 101], [1, 2]], names=['month', 'unit'])
    actual = pd.DataFrame({'target': [0.1, 0.2, 0.3, 0.4]}, index=index_actual)
    
    # Extra unit 3 (not in actuals)
    index_pred = pd.MultiIndex.from_product([[100, 101], [1, 2, 3]], names=['month', 'unit'])
    pred_0 = pd.DataFrame({'pred_target': [0.11, 0.21, 0.99, 0.31, 0.41, 0.99]}, index=index_pred)
    
    config = {
        'steps': [1, 2],
        'regression_targets': ['target'],
        'regression_point_metrics': ['MSE']
    }
    
    return actual, [pred_0], "target", config

@pytest.fixture
def red_data_inconsistent_samples():
    """Ragged sample lengths (e.g. some rows have 2 samples, some have 3)."""
    index = pd.MultiIndex.from_product([[100], [1, 2]], names=['month', 'unit'])
    actual = pd.DataFrame({'target': [0.1, 0.2]}, index=index)
    
    # Row 1 has 2 samples, Row 2 has 3 samples
    pred_0 = pd.DataFrame({
        'pred_target': [[0.1, 0.12], [0.2, 0.22, 0.24]]
    }, index=index)
    
    config = {
        'steps': [1],
        'regression_targets': ['target'],
        'regression_point_metrics': [],
        'regression_sample_metrics': ['CRPS']
    }
    
    return actual, [pred_0], "target", config

@pytest.fixture
def red_data_nan_index():
    """NaNs in index levels."""
    # Note: Use object dtype for index to allow NaN mixed with ints
    idx_actual = pd.MultiIndex.from_tuples([(100, 1), (101, 1), (np.nan, 2)], names=['month', 'unit'])
    actual = pd.DataFrame({'target': [0.1, 0.2, 0.3]}, index=idx_actual)
    
    idx_pred = pd.MultiIndex.from_tuples([(100, 1), (101, 1), (np.nan, 2)], names=['month', 'unit'])
    pred_0 = pd.DataFrame({'pred_target': [0.11, 0.21, 0.31]}, index=idx_pred)
    
    config = {
        'steps': [1, 2],
        'regression_targets': ['target'],
        'regression_point_metrics': ['MSE']
    }
    
    return actual, [pred_0], "target", config

def test_parity_red_unordered(red_data_unordered):
    actual, predictions, target, config = red_data_unordered
    manager = EvaluationManager()
    legacy_results = manager.evaluate(actual, predictions, target, config)
    ef = PandasAdapter.from_dataframes(actual, predictions, target)
    native_evaluator = NativeEvaluator(config)
    native_results = native_evaluator.evaluate(ef)
    assert_parity(legacy_results, native_results)

def test_parity_red_coordinates(red_data_coordinates):
    actual, predictions, target, config = red_data_coordinates
    manager = EvaluationManager()
    legacy_results = manager.evaluate(actual, predictions, target, config)
    ef = PandasAdapter.from_dataframes(actual, predictions, target)
    native_evaluator = NativeEvaluator(config)
    native_results = native_evaluator.evaluate(ef)
    assert_parity(legacy_results, native_results)

def test_fail_loud_inconsistent_samples(red_data_inconsistent_samples):
    actual, predictions, target, config = red_data_inconsistent_samples
    manager = EvaluationManager()
    # The new implementation raises a more descriptive error from the adapter
    with pytest.raises(ValueError, match="Inconsistent list lengths"):
        manager.evaluate(actual, predictions, target, config)


def test_fail_loud_nan_index(red_data_nan_index):
    actual, predictions, target, config = red_data_nan_index
    manager = EvaluationManager()
    # The new implementation fails early in the adapter if NaNs are detected
    with pytest.raises(ValueError, match="NaN detected in 'time' index level"):
        manager.evaluate(actual, predictions, target, config)

