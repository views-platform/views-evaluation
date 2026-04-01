"""
Example: Using the Native Evaluation API

This script demonstrates how to evaluate forecasts using the
NativeEvaluator and EvaluationFrame with pure NumPy arrays.
"""
import numpy as np
from views_evaluation import EvaluationFrame, NativeEvaluator

# 1. Prepare data as NumPy arrays
n = 4
y_true = np.array([0.0, 1.0, 0.0, 1.0])
y_pred = np.array([
    [0.1, 0.2, 0.05],
    [0.8, 0.9, 0.7],
    [0.1, 0.15, 0.2],
    [0.7, 0.8, 0.9],
])  # shape (4, 3) — 3-sample ensemble

identifiers = {
    'time':   np.array([100, 100, 101, 101]),
    'unit':   np.array([1, 2, 1, 2]),
    'origin': np.array([0, 0, 1, 1]),
    'step':   np.array([1, 1, 1, 1]),
}

# 2. Construct EvaluationFrame (validates shapes, NaN, identifiers)
ef = EvaluationFrame(y_true, y_pred, identifiers, metadata={'target': 'target'})
print(f"EvaluationFrame: {ef.n_rows} rows, {ef.n_samples} samples")

# 3. Configure and evaluate
config = {
    'steps': [1],
    'regression_targets': ['target'],
    'regression_sample_metrics': ['CRPS', 'Ignorance'],
}
evaluator = NativeEvaluator(config)
report = evaluator.evaluate(ef)
print("Evaluation complete.")

# 4. Export results
print("\nMonth-wise results (dict):")
print(report.to_dict()['schemas']['month'])

print("\nFull export:")
d = report.to_dict()
print(f"Target: {d['target']}, Schemas: {list(d['schemas'].keys())}")
