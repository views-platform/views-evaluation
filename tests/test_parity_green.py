import pytest
import pandas as pd
import numpy as np
from views_evaluation.evaluation.evaluation_manager import EvaluationManager
from views_evaluation.evaluation.adapters import PandasAdapter
from views_evaluation.evaluation.evaluation_frame import EvaluationFrame
from views_evaluation.evaluation.native_evaluator import NativeEvaluator

def assert_parity(legacy_results, native_results, tolerance=1e-9):
    """
    Asserts bit-wise (or within tolerance) parity between legacy and native results.
    legacy_results: output of EvaluationManager.evaluate()
    native_results: output of NativeEvaluator.evaluate()
    """
    for schema in ["month", "time_series", "step"]:
        legacy_df = legacy_results[schema][1]
        native_df = native_results[schema][1]
        
        # Check index parity
        pd.testing.assert_index_equal(legacy_df.index, native_df.index)
        
        # Check column parity (might be slight differences in names if not careful)
        pd.testing.assert_index_equal(legacy_df.columns, native_df.columns)
        
        # Check value parity
        pd.testing.assert_frame_equal(legacy_df, native_df, atol=tolerance)

@pytest.fixture
def green_data():
    """Clean, overlapping rolling origin data."""
    index = pd.MultiIndex.from_product([[100, 101, 102], [1, 2]], names=['month', 'unit'])
    actual = pd.DataFrame({'target': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}, index=index)
    
    # Sequence 0: Steps 1-2 for all units
    pred_0 = pd.DataFrame({'pred_target': [0.11, 0.21, 0.31, 0.41]}, index=index[:4])
    # Sequence 1: Steps 1-2 for all units, starting from month 101
    pred_1 = pd.DataFrame({'pred_target': [0.32, 0.42, 0.52, 0.62]}, index=index[2:])
    
    config = {
        'steps': [1, 2],
        'regression_targets': ['target'],
        'regression_point_metrics': ['MSE']
    }
    
    return actual, [pred_0, pred_1], "target", config

@pytest.fixture
def green_data_samples():
    """Clean, overlapping rolling origin data with samples."""
    index = pd.MultiIndex.from_product([[100, 101, 102], [1, 2]], names=['month', 'unit'])
    actual = pd.DataFrame({'target': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}, index=index)
    
    # 3 samples per prediction
    pred_0 = pd.DataFrame({
        'pred_target': [[0.1, 0.12, 0.08], [0.2, 0.22, 0.18], [0.3, 0.32, 0.28], [0.4, 0.42, 0.38]]
    }, index=index[:4])
    pred_1 = pd.DataFrame({
        'pred_target': [[0.31, 0.33, 0.29], [0.41, 0.43, 0.39], [0.51, 0.53, 0.49], [0.61, 0.63, 0.59]]
    }, index=index[2:])
    
    config = {
        'steps': [1, 2],
        'regression_targets': ['target'],
        'regression_point_metrics': [],
        'regression_sample_metrics': ['CRPS']
    }
    
    return actual, [pred_0, pred_1], "target", config

def test_parity_green_happy_path(green_data):
    actual, predictions, target, config = green_data
    
    # 1. Run Legacy
    manager = EvaluationManager()
    legacy_results = manager.evaluate(actual, predictions, target, config)
    
    # 2. Run Native (New Path)
    ef = PandasAdapter.from_dataframes(actual, predictions, target)
    native_evaluator = NativeEvaluator(config)
    native_results = native_evaluator.evaluate(ef)
    
    # 3. Assert Parity
    assert_parity(legacy_results, native_results)

def test_parity_green_samples(green_data_samples):
    actual, predictions, target, config = green_data_samples
    
    # 1. Run Legacy
    manager = EvaluationManager()
    legacy_results = manager.evaluate(actual, predictions, target, config)
    
    # 2. Run Native (New Path)
    ef = PandasAdapter.from_dataframes(actual, predictions, target)
    native_evaluator = NativeEvaluator(config)
    native_results = native_evaluator.evaluate(ef)
    
    # 3. Assert Parity
    assert_parity(legacy_results, native_results)
