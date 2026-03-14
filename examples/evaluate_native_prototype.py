import numpy as np
import pandas as pd
from views_evaluation.evaluation.adapters import PandasAdapter
from views_evaluation.evaluation.evaluation_frame import EvaluationFrame

def mock_metrics_mse(ef: EvaluationFrame) -> float:
    """A 'native' metric that uses broadcasting."""
    # y_true (N,) broadcasts to (N, S)
    # y_pred (N, S)
    # result (N, S) -> mean(axis=1) -> (N,) -> mean() -> scalar
    errors = (ef.y_true[:, np.newaxis] - ef.y_pred) ** 2
    return np.mean(errors)

def run_parity_demo():
    # 1. Create dummy data mimicking the current structure
    index = pd.MultiIndex.from_product([[100, 101, 102], [1, 2]], names=['month', 'unit'])
    actual = pd.DataFrame({'target': np.random.rand(6)}, index=index)
    
    # Two sequences (overlapping)
    pred_0 = pd.DataFrame({'pred_target': [[x, x+0.1] for x in np.random.rand(4)]}, 
                          index=index[:4])
    pred_1 = pd.DataFrame({'pred_target': [[x, x+0.1] for x in np.random.rand(4)]}, 
                          index=index[2:])
    
    print("--- 1. Adapter Phase ---")
    ef = PandasAdapter.from_dataframes(actual, [pred_0, pred_1], "target")
    print(ef)

    print("\n--- 2. Schema Preservation ---")
    
    # Month-wise
    print("\nMonth-wise Groups:")
    month_groups = ef.get_group_indices('time')
    for month, idx in month_groups.items():
        sub_ef = ef.select_indices(idx)
        mse = mock_metrics_mse(sub_ef)
        print(f"  Month {month}: {sub_ef.n_rows} rows, MSE={mse:.4f}")

    # Step-wise
    print("\nStep-wise Groups:")
    step_groups = ef.get_group_indices('step')
    for step, idx in step_groups.items():
        sub_ef = ef.select_indices(idx)
        mse = mock_metrics_mse(sub_ef)
        print(f"  Step {step}: {sub_ef.n_rows} rows, MSE={mse:.4f}")

    # Sequence-wise (Origin-wise)
    print("\nSequence-wise Groups:")
    origin_groups = ef.get_group_indices('origin')
    for origin, idx in origin_groups.items():
        sub_ef = ef.select_indices(idx)
        mse = mock_metrics_mse(sub_ef)
        print(f"  Sequence {origin}: {sub_ef.n_rows} rows, MSE={mse:.4f}")

if __name__ == "__main__":
    run_parity_demo()
