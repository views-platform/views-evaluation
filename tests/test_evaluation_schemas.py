
"""
This test suite rigorously verifies the grouping logic of the three evaluation
schemas (step-wise, time-series-wise, and month-wise) as described in the
core project documentation.
"""
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from views_evaluation.evaluation.evaluation_manager import EvaluationManager
from views_evaluation.evaluation.metrics import RegressionPointEvaluationMetrics

@pytest.fixture
def schema_test_data():
    """
    Generates a predictable, non-random "predictive parallelogram" for testing.

    - 3 sequences (t0, t1, t2)
    - 4 steps per sequence (s1, s2, s3, s4)
    - 2 locations (l0, l1)
    - Start month: 100

    Parallelogram structure (value is month_id):
            l0   l1  (Sequence 0)
    t0_s1:  100  100
    t0_s2:  101  101
    t0_s3:  102  102
    t0_s4:  103  103
            ...  ... (Sequence 1)
    t1_s1:  101  101
    t1_s2:  102  102
    t1_s3:  103  103
    t1_s4:  104  104
            ...  ... (Sequence 2)
    t2_s1:  102  102
    t2_s2:  103  103
    t2_s3:  104  104
    t2_s4:  105  105
    """
    target_name = "lr_test_target"
    pred_col_name = f"pred_{target_name}"
    loc_id_name = "location_id"
    num_sequences = 3
    num_steps = 4
    num_locations = 2
    start_month = 100

    # 1. Actuals DataFrame (covering all possible months)
    actuals_index = pd.MultiIndex.from_product(
        [range(start_month, start_month + num_sequences + num_steps), range(num_locations)],
        names=['month_id', loc_id_name]
    )
    # Use month_id as the value for easy checking
    actuals_values = [idx[0] for idx in actuals_index]
    actuals = pd.DataFrame({target_name: actuals_values}, index=actuals_index)

    # 2. Predictions List
    predictions_list = []
    for i in range(num_sequences):
        preds_index = pd.MultiIndex.from_product(
            [range(start_month + i, start_month + i + num_steps), range(num_locations)],
            names=['month_id', loc_id_name]
        )
        # Use month_id as the prediction value for easy checking. Wrap in a list.
        pred_values = [[idx[0]] for idx in preds_index]
        preds = pd.DataFrame({pred_col_name: pred_values}, index=preds_index)
        predictions_list.append(preds)

    # 3. Config
    config = {'steps': list(range(1, num_steps + 1))}

    return actuals, predictions_list, target_name, config


def get_months_from_mock_call(call):
    """Helper to extract unique month_ids from a mock call's DataFrame argument."""
    df = call[0][1]  # call[0] is args, [1] is the matched_pred dataframe
    return sorted(df.index.get_level_values('month_id').unique().tolist())


def test_step_wise_schema_grouping(schema_test_data):
    """
    Verify that step-wise evaluation groups data by forecast horizon (diagonals).
    """
    actuals, preds, target, config = schema_test_data
    manager = EvaluationManager()
    mock_metric_func = MagicMock()

    with patch.dict(manager.regression_point_functions, {"RMSLE": mock_metric_func}):
        actuals, preds = manager._process_data(actuals, preds, target)
        manager.step_wise_evaluation(
            actuals, preds, target, config["steps"],
            metrics_list=["RMSLE"],
            metric_functions=manager.regression_point_functions,
            metrics_cls=RegressionPointEvaluationMetrics,
        )

    # Expected groupings for steps (diagonals of the parallelogram)
    expected_step_months = {
        # step 1: (t0_s1, t1_s1, t2_s1) -> months (100, 101, 102)
        0: [100, 101, 102],
        # step 2: (t0_s2, t1_s2, t2_s2) -> months (101, 102, 103)
        1: [101, 102, 103],
        # step 3: (t0_s3, t1_s3, t2_s3) -> months (102, 103, 104)
        2: [102, 103, 104],
        # step 4: (t0_s4, t1_s4, t2_s4) -> months (103, 104, 105)
        3: [103, 104, 105],
    }

    assert mock_metric_func.call_count == len(expected_step_months)

    for i, expected_months in expected_step_months.items():
        call = mock_metric_func.call_args_list[i]
        observed_months = get_months_from_mock_call(call)
        assert observed_months == expected_months, f"Mismatch on step {i+1}"


def test_time_series_wise_schema_grouping(schema_test_data):
    """
    Verify that time-series-wise evaluation groups data by forecast run (columns).
    """
    actuals, preds, target, config = schema_test_data
    manager = EvaluationManager()
    mock_metric_func = MagicMock()

    with patch.dict(manager.regression_point_functions, {"RMSLE": mock_metric_func}):
        actuals, preds = manager._process_data(actuals, preds, target)
        manager.time_series_wise_evaluation(
            actuals, preds, target,
            metrics_list=["RMSLE"],
            metric_functions=manager.regression_point_functions,
            metrics_cls=RegressionPointEvaluationMetrics,
        )

    # Expected groupings for time-series (columns of the parallelogram)
    expected_ts_months = {
        # sequence 0: months 100, 101, 102, 103
        0: [100, 101, 102, 103],
        # sequence 1: months 101, 102, 103, 104
        1: [101, 102, 103, 104],
        # sequence 2: months 102, 103, 104, 105
        2: [102, 103, 104, 105],
    }

    assert mock_metric_func.call_count == len(expected_ts_months)

    for i, expected_months in expected_ts_months.items():
        call = mock_metric_func.call_args_list[i]
        observed_months = get_months_from_mock_call(call)
        assert observed_months == expected_months, f"Mismatch on time-series {i}"


def test_month_wise_schema_grouping(schema_test_data):
    """
    Verify that month-wise evaluation groups data by calendar month (rows).
    """
    actuals, preds, target, config = schema_test_data
    manager = EvaluationManager()
    mock_metric_func = MagicMock()

    with patch.dict(manager.regression_point_functions, {"RMSLE": mock_metric_func}):
        actuals, preds = manager._process_data(actuals, preds, target)
        manager.month_wise_evaluation(
            actuals, preds, target,
            metrics_list=["RMSLE"],
            metric_functions=manager.regression_point_functions,
            metrics_cls=RegressionPointEvaluationMetrics,
        )

    # For month-wise, each call corresponds to one month.
    # We check that each month was called and that the data in the call is correct.
    observed_calls = {}
    for call in mock_metric_func.call_args_list:
        df_pred = call[0][1]
        month = get_months_from_mock_call(call)[0]
        # Check that dataframe only contains data for its specified month
        assert all(m == month for m in get_months_from_mock_call(call))
        observed_calls[month] = df_pred

    # Expected months in the full parallelogram
    expected_months = [100, 101, 102, 103, 104, 105]
    assert sorted(observed_calls.keys()) == expected_months

    # Check the number of predictions for a few key months
    # Month 100: Only from sequence 0 (2 locations)
    assert len(observed_calls[100]) == 2
    # Month 101: From sequence 0 and 1 (2 locs * 2 seqs = 4)
    assert len(observed_calls[101]) == 4
    # Month 102: From sequence 0, 1, and 2 (2 locs * 3 seqs = 6)
    assert len(observed_calls[102]) == 6
    # Month 105: Only from sequence 2 (2 locations)
    assert len(observed_calls[105]) == 2
