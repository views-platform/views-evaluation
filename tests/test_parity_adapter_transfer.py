"""
PHASE-3-DELETE: Tests parity between internal and external PandasAdapter adaptation.
Will be deleted when Phase 3 of the orchestrator migration is complete.
See reports/2026-02-25_evaluation_frame_refactor/10_orchestrator_migration_plan.md
"""
import pytest
import pandas as pd
from views_evaluation import EvaluationManager, PandasAdapter, NativeEvaluator

def test_parity_internal_vs_external_adaptation():
    """
    PROVING PARITY FOR UPSTREAMING:
    This test verifies that adapting a DataFrame to an EvaluationFrame 
    OUTSIDE of the EvaluationManager produces identical results to 
    letting the Manager handle it internally.
    """
    # 1. Setup Data
    index = pd.MultiIndex.from_product([[100, 101], [1, 2]], names=['month', 'unit'])
    actuals = pd.DataFrame({'target': [0, 1, 0, 1]}, index=index)
    preds = [pd.DataFrame({'pred_target': [0.1, 0.8, 0.15, 0.7]}, index=index)]
    
    config = {
        'steps': [1],
        'regression_targets': ['target'],
        'regression_point_metrics': ['MSE']
    }
    
    manager = EvaluationManager()
    
    # 2. PATH A: Internal Adaptation (The status quo)
    # Manager receives DataFrames, adapts internally.
    results_internal = manager.evaluate(actuals, preds, "target", config)
    
    # 3. PATH B: External Adaptation (The future)
    # Simulation of Orchestrator running the adapter.
    ef_external = PandasAdapter.from_dataframes(actuals, preds, "target")
    
    # We use the NativeEvaluator directly to simulate the final state
    evaluator = NativeEvaluator(config)
    report_external = evaluator.evaluate(ef_external)
    
    # 4. Bit-wise Parity Check
    for schema in ["month", "time_series", "step"]:
        df_internal = results_internal[schema][1]
        df_external = report_external.to_dataframe(schema)
        
        pd.testing.assert_frame_equal(df_internal, df_external, 
                                      obj=f"Divergence in schema: {schema}")

    print("Parity Proven: External adaptation matches internal adaptation 100%.")

def test_shadow_verification_mode():
    """Verifies that verify_parity=True catches mismatches and allows matches."""
    index = pd.MultiIndex.from_product([[100], [1]], names=['month', 'unit'])
    actuals = pd.DataFrame({'target': [1]}, index=index)
    preds = [pd.DataFrame({'pred_target': [0.9]}, index=index)]
    config = {'steps': [1], 'regression_targets': ['target'], 'regression_point_metrics': ['MSE']}
    
    manager = EvaluationManager()
    ef_external = PandasAdapter.from_dataframes(actuals, preds, "target")
    
    # 1. Matching case - should pass silently
    manager.evaluate(actuals, preds, "target", config, ef=ef_external, verify_parity=True)
    
    # 2. Mismatching case - should raise ValueError
    ef_corrupted = PandasAdapter.from_dataframes(actuals, preds, "target")
    ef_corrupted.y_true = ef_corrupted.y_true * 2 # Corrupt the data
    
    with pytest.raises(ValueError, match="Parity Failure"):
        manager.evaluate(actuals, preds, "target", config, ef=ef_corrupted, verify_parity=True)

