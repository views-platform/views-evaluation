import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, mock_open
from sklearn.metrics import root_mean_squared_log_error
import properscoring as ps
from views_evaluation.evaluation.evaluation_manager import EvaluationManager
from views_evaluation.evaluation.metrics import UncertaintyEvaluationMetrics, PointEvaluationMetrics


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
def mock_uncertainty_predictions(mock_index):
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


def test_step_wise_evaluation_point(mock_actual, mock_point_predictions):
    manager = EvaluationManager(metrics_list=["RMSLE", "CRPS", "ABCD"])
    evaluation_dict = manager.step_wise_evaluation(
        mock_actual, mock_point_predictions, "target", [1, 2, 3], False
    )

    actuals = [[1, 2, 2, 3], [2, 3, 3, 4], [3, 4, 4, 5]]
    preds = [[1, 3, 2, 4], [5, 7, 6, 8], [9, 7, 10, 8]]
    df_evaluation_test = pd.DataFrame(
        {
            "RMSLE": [
                root_mean_squared_log_error(actual, pred)
                for (actual, pred) in zip(actuals, preds)
            ],
            "CRPS": [
                ps.crps_ensemble(actual, pred).mean()
                for (actual, pred) in zip(actuals, preds)
            ],
        },
        index=["step01", "step02", "step03"],
    )

    assert ["step01", "step02", "step03"] == list(evaluation_dict.keys())
    assert np.allclose(PointEvaluationMetrics.evaluation_dict_to_dataframe(evaluation_dict), df_evaluation_test, atol=0.000001)


def test_step_wise_evaluation_uncertainty(mock_actual, mock_uncertainty_predictions):
    manager = EvaluationManager(metrics_list=["RMSLE", "CRPS", "ABCD"])
    evaluation_dict = manager.step_wise_evaluation(
        mock_actual, mock_uncertainty_predictions, "target", [1, 2, 3], True
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
    assert np.allclose(UncertaintyEvaluationMetrics.evaluation_dict_to_dataframe(evaluation_dict), df_evaluation_test, atol=0.000001)


def test_time_series_wise_evaluation_point(mock_actual, mock_point_predictions):
    manager = EvaluationManager(metrics_list=["RMSLE", "CRPS", "Diversity"])
    evaluation_dict = manager.time_series_wise_evaluation(
        mock_actual, mock_point_predictions, "target", False
    )

    actuals = [[1, 2, 2, 3, 3, 4], [2, 3, 3, 4, 4, 5]]
    preds = [1, 3, 5, 7, 9, 7], [2, 4, 6, 8, 10, 8]
    evaluation_test = {
        "ts00": {
            "RMSLE": root_mean_squared_log_error(actuals[0], preds[0]),

            "CRPS": ps.crps_ensemble(actuals[0], preds[0]).mean(),
            "Diversity": {
                1: {"total_actual": 6.0, "total_pred": 15.0, "ratio": 2.5}, 
                2: {"total_actual": 9.0, "total_pred": 17.0, "ratio": 17/9}
            }
        },
        "ts01": {
            "RMSLE": root_mean_squared_log_error(actuals[1], preds[1]),
            "CRPS": ps.crps_ensemble(actuals[1], preds[1]).mean(),
            "Diversity": {
                1: {"total_actual": 9.0, "total_pred": 18.0, "ratio": 2}, 
                2: {"total_actual": 12.0, "total_pred": 20.0, "ratio": 20/12}
            }
        }
    }


    assert ["ts00", "ts01"] == list(evaluation_dict.keys())
    for ts_key, data in evaluation_test.items():
        for loc_id, res in data["Diversity"].items():
            actual_res = evaluation_dict[ts_key].Diversity[loc_id]
            assert actual_res["total_actual"] == res["total_actual"]
            assert actual_res["total_pred"] == res["total_pred"]
            assert np.allclose(actual_res["ratio"], res["ratio"], atol=0.000001)


def test_time_series_wise_evaluation_uncertainty(mock_actual, mock_uncertainty_predictions):
    manager = EvaluationManager(metrics_list=["RMSLE", "CRPS"])
    evaluation_dict = manager.time_series_wise_evaluation(
        mock_actual, mock_uncertainty_predictions, "target", True
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
    assert np.allclose(UncertaintyEvaluationMetrics.evaluation_dict_to_dataframe(evaluation_dict), df_evaluation_test, atol=0.000001)


def test_month_wise_evaluation_point(mock_actual, mock_point_predictions):
    manager = EvaluationManager(metrics_list=["RMSLE", "CRPS", "ABCD"])
    evaluation_dict = manager.month_wise_evaluation(
        mock_actual, mock_point_predictions, "target", False
    )

    actuals = [[1, 2], [2, 3, 2, 3], [3, 4, 3, 4], [4, 5]]
    preds = [[1, 3], [5, 7, 2, 4], [9, 7, 6, 8], [10, 8]]
    df_evaluation_test = pd.DataFrame({
            "RMSLE": [
                root_mean_squared_log_error(actual, pred)
                for (actual, pred) in zip(actuals, preds)
            ],
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
    assert np.allclose(PointEvaluationMetrics.evaluation_dict_to_dataframe(evaluation_dict), df_evaluation_test, atol=0.000001)


def test_month_wise_evaluation_uncertainty(mock_actual, mock_uncertainty_predictions):
    manager = EvaluationManager(metrics_list=["RMSLE", "CRPS", "ABCD"])
    evaluation_dict = manager.month_wise_evaluation(
        mock_actual, mock_uncertainty_predictions, "target", True
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
    assert np.allclose(UncertaintyEvaluationMetrics.evaluation_dict_to_dataframe(evaluation_dict), df_evaluation_test, atol=0.000001)







