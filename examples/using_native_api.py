"""
Example: Using the Native Evaluation API

This script demonstrates the modern, performant way to evaluate forecasts 
using the NativeEvaluator and EvaluationFrame. This path is up to 14x 
faster for probabilistic forecasts.
"""
import pandas as pd
from views_evaluation import PandasAdapter, NativeEvaluator

# 1. Prepare dummy data (The legacy format)
index = pd.MultiIndex.from_product([[100, 101], [1, 2]], names=['month', 'unit'])
actuals = pd.DataFrame({'target': [0, 1, 0, 1]}, index=index)
# 3-sample ensemble predictions
preds = [
    pd.DataFrame({'pred_target': [[0.1, 0.2, 0.05], [0.8, 0.9, 0.7]]}, index=index[:2]),
    pd.DataFrame({'pred_target': [[0.1, 0.15, 0.2], [0.7, 0.8, 0.9]]}, index=index[2:])
]

# 2. Configure metrics
config = {
    'steps': [1],
    'regression_targets': ['target'],
    'regression_sample_metrics': ['CRPS', 'Ignorance']
}

print("--- Step 1: Adapt Data ---")
# The adapter performs alignment, truth-duplication, and list-extraction.
# This step can be moved to the Orchestration layer (Pipeline Core) in the future.
ef = PandasAdapter.from_dataframes(actuals, preds, "target")
print(f"Adapted data: {ef.n_rows} rows, {ef.n_samples} samples")

print("")
print("--- Step 2: Evaluate ---")
# The evaluator is stateless and pure math (no pandas).
evaluator = NativeEvaluator(config)
report = evaluator.evaluate(ef)
print("Evaluation complete.")

print("")
print("--- Step 3: Export Results ---")
# Convert to DataFrames only when needed for reporting
month_df = report.to_dataframe(schema="month")
print("Month-wise results:")
print(month_df)

# Or export to pure dict for JSON serialization
json_friendly_dict = report.to_dict()
print("")
print("JSON Export (Sample):")
print(f"Target: {json_friendly_dict['target']}")
print(f"Schemas found: {list(json_friendly_dict['schemas'].keys())}")
