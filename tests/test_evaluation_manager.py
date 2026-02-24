import logging
import pandas as pd
import numpy as np
import pytest
from sklearn.metrics import root_mean_squared_log_error, average_precision_score
import properscoring as ps
from views_evaluation.evaluation.evaluation_manager import EvaluationManager
from views_evaluation.evaluation.metric_calculators import (
    REGRESSION_POINT_METRIC_FUNCTIONS,
    REGRESSION_SAMPLE_METRIC_FUNCTIONS,
)
from views_evaluation.evaluation.metrics import (
    RegressionPointEvaluationMetrics,
    RegressionSampleEvaluationMetrics,
)


@pytest.fixture
def mock_index():
    index_0 = pd.MultiIndex.from_tuples(
        [
            (100, 1),
            (100, 2),
            (101, 1),
            (101, 2),
            (102, 1),
            (102, 2),
        ],
        names=["month", "country"],
    )
    index_1 = pd.MultiIndex.from_tuples(
        [
            (101, 1),
            (101, 2),
            (102, 1),
            (102, 2),
            (103, 1),
            (103, 2),
        ],
        names=["month", "country"],
    )
    return [index_0, index_1]


@pytest.fixture
def mock_actual():
    index = pd.MultiIndex.from_tuples(
        [
            (99, 1),
            (99, 2),
            (100, 1),
            (100, 2),
            (101, 1),
            (101, 2),
            (102, 1),
            (102, 2),
            (103, 1),
            (103, 2),
            (104, 1),
            (104, 2),
        ],
        names=["month", "country"],
    )
    df = pd.DataFrame(
        {
            "target": [0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0],
            "covariate_1": [3.0, 2.0, 4.0, 5.0, 2.0, 6.0, 8.0, 5.0, 3.0, 2.0, 9.0, 4.0],
        },
        index=index,
    )
    return EvaluationManager.convert_to_array(df, "target")


@pytest.fixture
def mock_point_predictions(mock_index):
    df1 = pd.DataFrame({"pred_target": [1.0, 3.0, 5.0, 7.0, 9.0, 7.0]}, index=mock_index[0])
    df2 = pd.DataFrame({"pred_target": [2.0, 4.0, 6.0, 8.0, 10.0, 8.0]}, index=mock_index[1])
    return [EvaluationManager.convert_to_array(df1, "pred_target"), EvaluationManager.convert_to_array(df2, "pred_target")]


@pytest.fixture
def mock_sample_predictions(mock_index):
    df1 = pd.DataFrame(
        {
            "pred_target": [
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
                [3.0, 4.0, 5.0],
                [4.0, 5.0, 6.0],
                [5.0, 6.0, 7.0],
                [6.0, 7.0, 8.0],
            ]
        },
        index=mock_index[0],
    )
    df2 = pd.DataFrame(
        {
            "pred_target": [
                [4.0, 6.0, 8.0],
                [5.0, 7.0, 9.0],
                [6.0, 8.0, 10.0],
                [7.0, 9.0, 11.0],
                [8.0, 10.0, 12.0],
                [9.0, 11.0, 13.0],
            ]
        },
        index=mock_index[1],
    )
    return [EvaluationManager.convert_to_array(df1, "pred_target"), EvaluationManager.convert_to_array(df2, "pred_target")]


def test_validate_dataframes_valid_type(mock_point_predictions):
    with pytest.raises(TypeError):
        EvaluationManager.validate_predictions(
            mock_point_predictions[0], "target"
        )


def test_validate_dataframes_valid_columns(mock_point_predictions):
    with pytest.raises(ValueError):
        EvaluationManager.validate_predictions(
            mock_point_predictions, "y"
        )

def test_get_evaluation_type():
    # Test case 1: All DataFrames for sample evaluation
    predictions_sample = [
        pd.DataFrame({'pred_target': [[1.0, 2.0], [3.0, 4.0]]}),
        pd.DataFrame({'pred_target': [[5.0, 6.0], [7.0, 8.0]]}),
    ]
    assert EvaluationManager.get_evaluation_type(predictions_sample, "pred_target") is True

    # Test case 2: All DataFrames for point evaluation
    predictions_point = [
        pd.DataFrame({'pred_target': [[1.0], [2.0]]}),
        pd.DataFrame({'pred_target': [[3.0], [4.0]]}),
    ]
    assert EvaluationManager.get_evaluation_type(predictions_point, "pred_target") is False

    # Test case 3: Mixed evaluation types
    predictions_mixed = [
        pd.DataFrame({'pred_target': [[1.0, 2.0], [3.0, 4.0]]}),
        pd.DataFrame({'pred_target': [[5.0], [6.0]]}),
    ]
    with pytest.raises(ValueError):
        EvaluationManager.get_evaluation_type(predictions_mixed, "pred_target")

    # Test case 4: Single element lists
    predictions_single_element = [
        pd.DataFrame({'pred_target': [[1.0], [2.0]]}),
        pd.DataFrame({'pred_target': [[3.0], [4.0]]}),
    ]
    assert EvaluationManager.get_evaluation_type(predictions_single_element, "pred_target") is False


def test_match_actual_pred_point(
    mock_actual, mock_point_predictions, mock_sample_predictions, mock_index
):
    df_matched = [
        pd.DataFrame({"target": [[1.0], [2.0], [2.0], [3.0], [3.0], [4.0]]}, index=mock_index[0]),
        pd.DataFrame({"target": [[2.0], [3.0], [3.0], [4.0], [4.0], [5.0]]}, index=mock_index[1]),
    ]
    for i in range(len(df_matched)):
        df_matched_actual_point, df_matched_point = (
            EvaluationManager._match_actual_pred(
                mock_actual, mock_point_predictions[i], "target"
            )
        )
        df_matched_actual_sample, df_matched_sample = (
            EvaluationManager._match_actual_pred(
                mock_actual, mock_sample_predictions[i], "target"
            )
        )
        assert df_matched[i].equals(df_matched_actual_point)
        assert df_matched_point.equals(mock_point_predictions[i])
        assert df_matched[i].equals(df_matched_actual_sample)
        assert df_matched_sample.equals(mock_sample_predictions[i])


def test_split_dfs_by_step(mock_point_predictions, mock_sample_predictions):
    df_splitted_point = [
        EvaluationManager.convert_to_array(pd.DataFrame(
            {"pred_target": [[1.0], [3.0], [2.0], [4.0]]},
            index=pd.MultiIndex.from_tuples(
                [(100, 1), (100, 2), (101, 1), (101, 2)], names=["month", "country"]
            ),
        ), "pred_target"),
        EvaluationManager.convert_to_array(pd.DataFrame(
            {"pred_target": [[5.0], [7.0], [6.0], [8.0]]},
            index=pd.MultiIndex.from_tuples(
                [(101, 1), (101, 2), (102, 1), (102, 2)], names=["month", "country"]
            ),
        ), "pred_target"),
        EvaluationManager.convert_to_array(pd.DataFrame(
            {"pred_target": [[9.0], [7.0], [10.0], [8.0]]},
            index=pd.MultiIndex.from_tuples(
                [(102, 1), (102, 2), (103, 1), (103, 2)], names=["month", "country"]
            ),
        ), "pred_target"),
    ]
    df_splitted_sample = [
        EvaluationManager.convert_to_array(pd.DataFrame(
            {"pred_target": [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [4.0, 6.0, 8.0], [5.0, 7.0, 9.0]]},
            index=pd.MultiIndex.from_tuples(
                [(100, 1), (100, 2), (101, 1), (101, 2)], names=["month", "country"]
            ),
        ), "pred_target"),
        EvaluationManager.convert_to_array(pd.DataFrame(
            {"pred_target": [[3.0, 4.0, 5.0], [4.0, 5.0, 6.0], [6.0, 8.0, 10.0], [7.0, 9.0, 11.0]]},
            index=pd.MultiIndex.from_tuples(
                [(101, 1), (101, 2), (102, 1), (102, 2)], names=["month", "country"]
            ),
        ), "pred_target"),
        EvaluationManager.convert_to_array(pd.DataFrame(
            {"pred_target": [[5.0, 6.0, 7.0], [6.0, 7.0, 8.0], [8.0, 10.0, 12.0], [9.0, 11.0, 13.0]]},
            index=pd.MultiIndex.from_tuples(
                [(102, 1), (102, 2), (103, 1), (103, 2)], names=["month", "country"]
            ),
        ), "pred_target"),
    ]
    df_splitted_point_test = EvaluationManager._split_dfs_by_step(
        mock_point_predictions
    )
    df_splitted_sample_test = EvaluationManager._split_dfs_by_step(
        mock_sample_predictions
    )
    for df1, df2 in zip(df_splitted_point, df_splitted_point_test):
        assert df1.equals(df2)
    for df1, df2 in zip(df_splitted_sample, df_splitted_sample_test):
        assert df1.equals(df2)


def test_step_wise_evaluation_point(mock_actual, mock_point_predictions):
    manager = EvaluationManager()
    evaluation_dict, df_evaluation = manager.step_wise_evaluation(
        mock_actual, mock_point_predictions, "target", [1, 2, 3],
        metrics_list=["RMSLE"],
        metric_functions=REGRESSION_POINT_METRIC_FUNCTIONS,
        metrics_cls=RegressionPointEvaluationMetrics,
    )

    actuals = [[1, 2, 2, 3], [2, 3, 3, 4], [3, 4, 4, 5]]
    preds = [[1, 3, 2, 4], [5, 7, 6, 8], [9, 7, 10, 8]]
    df_evaluation_test = pd.DataFrame(
        {
            "RMSLE": [
                root_mean_squared_log_error(actual, pred)
                for (actual, pred) in zip(actuals, preds)
            ],
        },
        index=["step01", "step02", "step03"],
    )

    assert ["step01", "step02", "step03"] == list(evaluation_dict.keys())
    assert np.allclose(df_evaluation, df_evaluation_test, atol=0.000001)


def test_step_wise_evaluation_sample(mock_actual, mock_sample_predictions):
    manager = EvaluationManager()
    evaluation_dict, df_evaluation = manager.step_wise_evaluation(
        mock_actual, mock_sample_predictions, "target", [1, 2, 3],
        metrics_list=["CRPS"],
        metric_functions=REGRESSION_SAMPLE_METRIC_FUNCTIONS,
        metrics_cls=RegressionSampleEvaluationMetrics,
    )
    actuals = [[1, 2, 2, 3], [2, 3, 3, 4], [3, 4, 4, 5]]
    preds = [
        [[1, 2, 3], [2, 3, 4], [4, 6, 8], [5, 7, 9]],
        [[3, 4, 5], [4, 5, 6], [6, 8, 10], [7, 9, 11]],
        [[5, 6, 7], [6, 7, 8], [8, 10, 12], [9, 11, 13]],
    ]
    df_evaluation_test = pd.DataFrame(
        {
            "CRPS": [
                ps.crps_ensemble(actual, pred).mean()
                for (actual, pred) in zip(actuals, preds)
            ],
        },
        index=["step01", "step02", "step03"],
    )

    assert ["step01", "step02", "step03"] == list(evaluation_dict.keys())
    assert np.allclose(df_evaluation, df_evaluation_test, atol=0.000001)


def test_time_series_wise_evaluation_point(mock_actual, mock_point_predictions):
    manager = EvaluationManager()
    evaluation_dict, df_evaluation = manager.time_series_wise_evaluation(
        mock_actual, mock_point_predictions, "target",
        metrics_list=["RMSLE"],
        metric_functions=REGRESSION_POINT_METRIC_FUNCTIONS,
        metrics_cls=RegressionPointEvaluationMetrics,
    )

    actuals = [[1, 2, 2, 3, 3, 4], [2, 3, 3, 4, 4, 5]]
    preds = [1, 3, 5, 7, 9, 7], [2, 4, 6, 8, 10, 8]
    df_evaluation_test = pd.DataFrame(
        {
            "RMSLE": [
                root_mean_squared_log_error(actual, pred)
                for (actual, pred) in zip(actuals, preds)
            ],
        },
        index=["ts00", "ts01"],
    )

    assert ["ts00", "ts01"] == list(evaluation_dict.keys())
    assert np.allclose(df_evaluation, df_evaluation_test, atol=0.000001)


def test_time_series_wise_evaluation_sample(mock_actual, mock_sample_predictions):
    manager = EvaluationManager()
    evaluation_dict, df_evaluation = manager.time_series_wise_evaluation(
        mock_actual, mock_sample_predictions, "target",
        metrics_list=["CRPS"],
        metric_functions=REGRESSION_SAMPLE_METRIC_FUNCTIONS,
        metrics_cls=RegressionSampleEvaluationMetrics,
    )

    actuals = [[1, 2, 2, 3, 3, 4], [2, 3, 3, 4, 4, 5]]
    preds = [
        [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]],
        [[4, 6, 8], [5, 7, 9], [6, 8, 10], [7, 9, 11], [8, 10, 12], [9, 11, 13]],
    ]
    df_evaluation_test = pd.DataFrame(
        {
            "CRPS": [
                ps.crps_ensemble(actual, pred).mean()
                for (actual, pred) in zip(actuals, preds)
            ],
        },
        index=["ts00", "ts01"],
    )

    assert ["ts00", "ts01"] == list(evaluation_dict.keys())
    assert np.allclose(df_evaluation, df_evaluation_test, atol=0.000001)


def test_month_wise_evaluation_point(mock_actual, mock_point_predictions):
    manager = EvaluationManager()
    evaluation_dict, df_evaluation = manager.month_wise_evaluation(
        mock_actual, mock_point_predictions, "target",
        metrics_list=["RMSLE"],
        metric_functions=REGRESSION_POINT_METRIC_FUNCTIONS,
        metrics_cls=RegressionPointEvaluationMetrics,
    )

    actuals = [[1, 2], [2, 3, 2, 3], [3, 4, 3, 4], [4, 5]]
    preds = [[1, 3], [5, 7, 2, 4], [9, 7, 6, 8], [10, 8]]
    df_evaluation_test = pd.DataFrame({
            "RMSLE": [
                root_mean_squared_log_error(actual, pred)
                for (actual, pred) in zip(actuals, preds)
            ],
        },
        index=["month100", "month101", "month102", "month103"],
    )

    assert ["month100", "month101", "month102", "month103"] == list(
        evaluation_dict.keys()
    )
    assert np.allclose(df_evaluation, df_evaluation_test, atol=0.000001)


def test_month_wise_evaluation_sample(mock_actual, mock_sample_predictions):
    manager = EvaluationManager()
    evaluation_dict, df_evaluation = manager.month_wise_evaluation(
        mock_actual, mock_sample_predictions, "target",
        metrics_list=["CRPS"],
        metric_functions=REGRESSION_SAMPLE_METRIC_FUNCTIONS,
        metrics_cls=RegressionSampleEvaluationMetrics,
    )

    actuals = [[1, 2], [2, 3, 2, 3], [3, 4, 3, 4], [4, 5]]
    preds = [
        [[1, 2, 3], [2, 3, 4]],
        [[3, 4, 5], [4, 5, 6], [4, 6, 8], [5, 7, 9]],
        [[5, 6, 7], [6, 7, 8], [6, 8, 10], [7, 9, 11]],
        [[8, 10, 12], [9, 11, 13]],
    ]
    df_evaluation_test = pd.DataFrame(
        {
            "CRPS": [
                ps.crps_ensemble(actual, pred).mean()
                for (actual, pred) in zip(actuals, preds)
            ],
        },
        index=["month100", "month101", "month102", "month103"],
    )

    assert ["month100", "month101", "month102", "month103"] == list(
        evaluation_dict.keys()
    )
    assert np.allclose(df_evaluation, df_evaluation_test, atol=0.000001)


def test_calculate_ap_point_predictions():
    """
    Test calculate_ap with pre-binarised actuals (0/1) and probability scores as predictions.
    """
    # Binary actuals: 1 = positive class, 0 = negative class
    actual_binary = [1, 0, 1, 0]
    # Probability scores for the positive class
    pred_scores = [0.9, 0.4, 0.3, 0.1]

    matched_actual = pd.DataFrame({'target': [[v] for v in actual_binary]})
    matched_pred = pd.DataFrame({'pred_target': [[v] for v in pred_scores]})

    from views_evaluation.evaluation.metric_calculators import calculate_ap
    ap_score = calculate_ap(matched_actual, matched_pred, 'target')

    expected_ap = average_precision_score(actual_binary, pred_scores)

    assert abs(ap_score - expected_ap) < 0.01


def test_calculate_ap_sample_predictions():
    """
    Test calculate_ap with pre-binarised actuals and distributional probability scores.
    Each prediction is a list of probability samples; actuals are 0/1.
    """
    # Binary actuals: 1 = positive, 0 = negative
    actual_binary = [1, 0, 1, 0]
    # Distributional probability predictions (multiple samples per observation)
    pred_scores = [
        [0.8, 0.9, 0.95],
        [0.3, 0.4, 0.45],
        [0.2, 0.25, 0.35],
        [0.05, 0.1, 0.15],
    ]

    matched_actual = pd.DataFrame({'target': [[v] for v in actual_binary]})
    matched_pred = pd.DataFrame({'pred_target': pred_scores})

    from views_evaluation.evaluation.metric_calculators import calculate_ap
    ap_score = calculate_ap(matched_actual, matched_pred, 'target')

    # Expected: actuals expanded to match samples, predictions are the raw samples
    actual_expanded = np.repeat(actual_binary, [len(p) for p in pred_scores])
    pred_flat = np.concatenate(pred_scores)
    expected_ap = average_precision_score(actual_expanded, pred_flat)

    assert abs(ap_score - expected_ap) < 0.01


# ---------------------------------------------------------------------------
# New tests for config normalisation and validation
# ---------------------------------------------------------------------------

def test_normalise_config_legacy_targets_key(caplog):
    """Legacy 'targets' key should be translated to 'regression_targets' with a warning."""
    config = {'steps': [1], 'targets': ['my_target'], 'regression_point_metrics': ['MSE']}
    with caplog.at_level(logging.WARNING):
        normalised = EvaluationManager._normalise_config(config)
    assert 'regression_targets' in normalised
    assert 'targets' not in normalised
    assert any('DEPRECATED' in r.message for r in caplog.records)


def test_normalise_config_legacy_metrics_key(caplog):
    """Legacy 'metrics' key should be translated to 'regression_point_metrics' with a warning."""
    config = {'steps': [1], 'regression_targets': ['t'], 'metrics': ['MSE']}
    with caplog.at_level(logging.WARNING):
        normalised = EvaluationManager._normalise_config(config)
    assert 'regression_point_metrics' in normalised
    assert 'metrics' not in normalised
    assert any('DEPRECATED' in r.message for r in caplog.records)


def test_validate_config_missing_steps():
    with pytest.raises(KeyError, match="steps"):
        EvaluationManager._validate_config({'regression_targets': ['t'], 'regression_point_metrics': ['MSE']})


def test_validate_config_missing_all_targets():
    with pytest.raises(KeyError):
        EvaluationManager._validate_config({'steps': [1]})


def test_validate_config_regression_targets_without_metrics():
    with pytest.raises(KeyError, match="regression_point_metrics"):
        EvaluationManager._validate_config({'steps': [1], 'regression_targets': ['t']})


def test_validate_config_classification_targets_without_metrics():
    with pytest.raises(KeyError, match="classification_point_metrics"):
        EvaluationManager._validate_config({'steps': [1], 'classification_targets': ['t']})


def test_evaluate_target_not_in_config(mock_actual, mock_point_predictions):
    manager = EvaluationManager()
    config = {
        'steps': [1, 2, 3],
        'regression_targets': ['some_other_target'],
        'regression_point_metrics': ['RMSLE'],
    }
    with pytest.raises(ValueError, match="not declared in config"):
        manager.evaluate(mock_actual, mock_point_predictions, 'target', config)


def test_evaluate_invalid_metric_for_task_type(mock_actual, mock_point_predictions):
    """AP is a classification metric — declaring it under regression_point_metrics should raise."""
    manager = EvaluationManager()
    config = {
        'steps': [1, 2, 3],
        'regression_targets': ['target'],
        'regression_point_metrics': ['AP'],  # AP is not a regression metric
    }
    with pytest.raises(ValueError, match="not valid for"):
        manager.evaluate(mock_actual, mock_point_predictions, 'target', config)
