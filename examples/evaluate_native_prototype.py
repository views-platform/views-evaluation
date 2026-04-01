"""
Example: EvaluationFrame grouping and metric computation

Demonstrates how EvaluationFrame provides month-wise, step-wise,
and origin-wise grouping for evaluation schemas.
"""
import numpy as np
from views_evaluation.evaluation.evaluation_frame import EvaluationFrame


def mock_metrics_mse(ef: EvaluationFrame) -> float:
    """A native metric using broadcasting: mean((y_true - y_pred)^2)."""
    errors = (ef.y_true[:, np.newaxis] - ef.y_pred) ** 2
    return np.mean(errors)


def run_demo():
    rng = np.random.default_rng(42)

    # Build 2 overlapping sequences, 3 months each, 2 units
    rows, y_true_list, y_pred_list = [], [], []
    for origin in range(2):
        for step_idx, month in enumerate(range(100 + origin, 103 + origin)):
            for unit in [1, 2]:
                rows.append((month, unit, origin, step_idx + 1))
                y_true_list.append(rng.random())
                y_pred_list.append([rng.random(), rng.random()])  # 2-sample ensemble

    n = len(rows)
    ef = EvaluationFrame(
        y_true=np.array(y_true_list),
        y_pred=np.array(y_pred_list),
        identifiers={
            'time':   np.array([r[0] for r in rows]),
            'unit':   np.array([r[1] for r in rows]),
            'origin': np.array([r[2] for r in rows]),
            'step':   np.array([r[3] for r in rows]),
        },
        metadata={'target': 'target'},
    )
    print(ef)

    for group_key in ['time', 'step', 'origin']:
        print(f"\n{group_key.title()}-wise Groups:")
        for val, idx in ef.get_group_indices(group_key).items():
            sub = ef.select_indices(idx)
            print(f"  {group_key}={val}: {sub.n_rows} rows, MSE={mock_metrics_mse(sub):.4f}")


if __name__ == "__main__":
    run_demo()
