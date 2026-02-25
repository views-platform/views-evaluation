import pytest
import pandas as pd
import numpy as np
from views_evaluation.evaluation.evaluation_manager import EvaluationManager
from views_evaluation.evaluation.adapters import PandasAdapter
from views_evaluation.evaluation.native_evaluator import NativeEvaluator
from tests.test_parity_green import assert_parity

@pytest.fixture
def beige_data_ragged():
    """Ragged sequences and missing months."""
    index = pd.MultiIndex.from_product([[100, 101, 102, 103], [1, 2]], names=['month', 'unit'])
    actual = pd.DataFrame({'target': np.random.rand(8)}, index=index)
    
    # Sequence 0: Month 100 and 101 (complete)
    pred_0 = pd.DataFrame({'pred_target': np.random.rand(4)}, index=index[:4])
    
    # Sequence 1: Month 101 and 102, but Month 101 Unit 2 is MISSING
    idx_1 = index[2:6].drop((101, 2))
    pred_1 = pd.DataFrame({'pred_target': np.random.rand(3)}, index=idx_1)
    
    # Sequence 2: Only Month 103
    pred_2 = pd.DataFrame({'pred_target': np.random.rand(2)}, index=index[6:])
    
    config = {
        'steps': [1, 2],
        'regression_targets': ['target'],
        'regression_point_metrics': ['MSE']
    }
    
    return actual, [pred_0, pred_1, pred_2], "target", config

def test_parity_beige_ragged(beige_data_ragged):
    actual, predictions, target, config = beige_data_ragged
    
    # 1. Run Legacy
    manager = EvaluationManager()
    legacy_results = manager.evaluate(actual, predictions, target, config)
    
    # 2. Run Native
    ef = PandasAdapter.from_dataframes(actual, predictions, target)
    native_evaluator = NativeEvaluator(config)
    native_results = native_evaluator.evaluate(ef)
    
    # 3. Assert Parity
    assert_parity(legacy_results, native_results)
