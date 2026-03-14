"""
PHASE-3-DELETE: Tests data contract enforcement for the legacy EvaluationManager path.
Will be deleted when Phase 3 of the orchestrator migration is complete.
See reports/2026-02-25_evaluation_frame_refactor/10_orchestrator_migration_plan.md
"""
import pandas as pd
import pytest
from views_evaluation.evaluation.evaluation_manager import EvaluationManager

@pytest.fixture
def mock_data():
    target = "lr_target"
    index = pd.MultiIndex.from_tuples([(100, 1), (101, 1)], names=["month", "id"])
    actual = pd.DataFrame({target: [10, 20]}, index=index)
    config = {
        "steps": [1, 2],
        "regression_targets": [target],
        "regression_point_metrics": ["MSE"],
    }
    return actual, target, config, index

def test_missing_pred_column(mock_data):
    actual, target, config, index = mock_data
    # Column name is wrong
    pred_df = pd.DataFrame({"wrong_name": [[10.5], [19.5]]}, index=index)
    manager = EvaluationManager()

    with pytest.raises(ValueError, match=f"must contain the column named 'pred_{target}'"):
        manager.evaluate(actual, [pred_df], target, config)

def test_extra_columns_raises_error(mock_data):
    """Verify that extra columns now raise a ValueError per the documentation."""
    actual, target, config, index = mock_data
    pred_df = pd.DataFrame({
        f"pred_{target}": [[10.5], [19.5]],
        "extra_garbage": [1, 2]
    }, index=index)
    manager = EvaluationManager()

    with pytest.raises(ValueError, match="must contain exactly one column"):
        manager.evaluate(actual, [pred_df], target, config)

def test_duplicate_pred_columns_raises_error(mock_data):
    """Verify that duplicate target columns cause a failure (currently a crash)."""
    actual, target, config, index = mock_data
    df1 = pd.DataFrame({f"pred_{target}": [[10.5], [19.5]]}, index=index)
    df2 = pd.DataFrame({f"pred_{target}": [[11.0], [20.0]]}, index=index)
    pred_df = pd.concat([df1, df2], axis=1)

    manager = EvaluationManager()

    # We expect a failure. Note: Ideally we want a custom ValueError from our validator.
    # Currently it raises a numpy/pandas ValueError during calculation.
    with pytest.raises(ValueError):
        manager.evaluate(actual, [pred_df], target, config)

def test_zero_index_overlap_graceful_failure(mock_data):
    """Verify behavior when actuals and predictions have no common months."""
    actual, target, config, _ = mock_data
    # Preds are for months 200, 201 (no overlap with 100, 101)
    index_no_overlap = pd.MultiIndex.from_tuples([(200, 1), (201, 1)], names=["month", "id"])
    pred_df = pd.DataFrame({f"pred_{target}": [[10.5], [19.5]]}, index=index_no_overlap)

    manager = EvaluationManager()

    # Currently, this crashes in np.concatenate inside the metric calculator.
    # We want it to either raise a clear error or return NaNs.
    with pytest.raises((ValueError, KeyError)):
        manager.evaluate(actual, [pred_df], target, config)

def test_mixed_point_and_sample_types(mock_data):
    actual, target, config, index = mock_data
    # First is point, second is sample
    pred1 = pd.DataFrame({f"pred_{target}": [[10.5], [19.5]]}, index=index)
    pred2 = pd.DataFrame({f"pred_{target}": [[10, 11, 12], [19, 20, 21]]}, index=index)

    manager = EvaluationManager()

    with pytest.raises(ValueError, match="Mix of evaluation types detected"):
        manager.evaluate(actual, [pred1, pred2], target, config)
