import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, mock_open
from sklearn.metrics import root_mean_squared_log_error
import properscoring as ps
from views_evaluation.evaluation.evaluation_manager import EvaluationManager


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
            "depvar": [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
            "covariate_1": [3, 2, 4, 5, 2, 6, 8, 5, 3, 2, 9, 4],
        },
        index=index,
    )
    return df


@pytest.fixture
def mock_point_predictions(mock_index):
    df1 = pd.DataFrame({"pred_depvar": [1, 3, 5, 7, 9, 7]}, index=mock_index[0])
    df2 = pd.DataFrame({"pred_depvar": [2, 4, 6, 8, 10, 8]}, index=mock_index[1])
    return [df1, df2]


@pytest.fixture
def mock_uncertainty_predictions(mock_index):
    df1 = pd.DataFrame(
        {
            "pred_depvar": [
                [1, 2, 3],
                [2, 3, 4],
                [3, 4, 5],
                [4, 5, 6],
                [5, 6, 7],
                [6, 7, 8],
            ]
        },
        index=mock_index[0],
    )
    df2 = pd.DataFrame(
        {
            "pred_depvar": [
                [4, 6, 8],
                [5, 7, 9],
                [6, 8, 10],
                [7, 9, 11],
                [8, 10, 12],
                [9, 11, 13],
            ]
        },
        index=mock_index[1],
    )
    return [df1, df2]


def test_check_dataframes_valid_type(mock_point_predictions):
    with pytest.raises(TypeError):
        EvaluationManager._check_dataframes(
            mock_point_predictions[0], "depvar", is_uncertainty=False
        )


def test_check_dataframes_valid_columns(mock_point_predictions):
    with pytest.raises(ValueError):
        EvaluationManager._check_dataframes(
            mock_point_predictions, "y", is_uncertainty=False
        )


def test_check_dataframes_valid_point(mock_uncertainty_predictions):
    with pytest.raises(ValueError):
        EvaluationManager._check_dataframes(
            mock_uncertainty_predictions, "depvar", is_uncertainty=False
        )


def test_check_dataframes_valid_uncertainty(mock_point_predictions):
    with pytest.raises(ValueError):
        EvaluationManager._check_dataframes(
            mock_point_predictions, "devpar", is_uncertainty=True
        )


def test_match_actual_pred_point(
    mock_actual, mock_point_predictions, mock_uncertainty_predictions, mock_index
):
    df_matched = [
        pd.DataFrame({"depvar": [1, 2, 2, 3, 3, 4]}, index=mock_index[0]),
        pd.DataFrame({"depvar": [2, 3, 3, 4, 4, 5]}, index=mock_index[1]),
    ]
    for i in range(len(df_matched)):
        df_matched_actual_point, df_matched_point = (
            EvaluationManager._match_actual_pred(
                mock_actual, mock_point_predictions[i], "depvar"
            )
        )
        df_matched_actual_uncertainty, df_matched_uncertainty = (
            EvaluationManager._match_actual_pred(
                mock_actual, mock_uncertainty_predictions[i], "depvar"
            )
        )
        assert df_matched[i].equals(df_matched_actual_point)
        assert df_matched_point.equals(mock_point_predictions[i])
        assert df_matched[i].equals(df_matched_actual_uncertainty)
        assert df_matched_uncertainty.equals(mock_uncertainty_predictions[i])


def test_split_dfs_by_step(mock_point_predictions, mock_uncertainty_predictions):
    df_splitted_point = [
        pd.DataFrame(
            {"pred_depvar": [1, 3, 2, 4]},
            index=pd.MultiIndex.from_tuples(
                [(100, 1), (100, 2), (101, 1), (101, 2)], names=["month", "country"]
            ),
        ),
        pd.DataFrame(
            {"pred_depvar": [5, 7, 6, 8]},
            index=pd.MultiIndex.from_tuples(
                [(101, 1), (101, 2), (102, 1), (102, 2)], names=["month", "country"]
            ),
        ),
        pd.DataFrame(
            {"pred_depvar": [9, 7, 10, 8]},
            index=pd.MultiIndex.from_tuples(
                [(102, 1), (102, 2), (103, 1), (103, 2)], names=["month", "country"]
            ),
        ),
    ]
    df_splitted_uncertainty = [
        pd.DataFrame(
            {"pred_depvar": [[1, 2, 3], [2, 3, 4], [4, 6, 8], [5, 7, 9]]},
            index=pd.MultiIndex.from_tuples(
                [(100, 1), (100, 2), (101, 1), (101, 2)], names=["month", "country"]
            ),
        ),
        pd.DataFrame(
            {"pred_depvar": [[3, 4, 5], [4, 5, 6], [6, 8, 10], [7, 9, 11]]},
            index=pd.MultiIndex.from_tuples(
                [(101, 1), (101, 2), (102, 1), (102, 2)], names=["month", "country"]
            ),
        ),
        pd.DataFrame(
            {"pred_depvar": [[5, 6, 7], [6, 7, 8], [8, 10, 12], [9, 11, 13]]},
            index=pd.MultiIndex.from_tuples(
                [(102, 1), (102, 2), (103, 1), (103, 2)], names=["month", "country"]
            ),
        ),
    ]
    df_splitted_point_test = EvaluationManager._split_dfs_by_step(
        mock_point_predictions
    )
    df_splitted_uncertainty_test = EvaluationManager._split_dfs_by_step(
        mock_uncertainty_predictions
    )
    for df1, df2 in zip(df_splitted_point, df_splitted_point_test):
        assert df1.equals(df2)
    for df1, df2 in zip(df_splitted_uncertainty, df_splitted_uncertainty_test):
        assert df1.equals(df2)


def test_step_wise_evaluation_point(mock_actual, mock_point_predictions):
    manager = EvaluationManager(metrics_list=["RMSLE", "CRPS", "ABCD"])
    evaluation_dict, df_evaluation = manager._step_wise_evaluation(
        mock_actual, mock_point_predictions, "depvar", [1, 2, 3], False
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
    assert np.allclose(df_evaluation, df_evaluation_test, atol=0.000001)


def test_step_wise_evaluation_uncertainty(mock_actual, mock_uncertainty_predictions):
    manager = EvaluationManager(metrics_list=["RMSLE", "CRPS", "ABCD"])
    evaluation_dict, df_evaluation = manager._step_wise_evaluation(
        mock_actual, mock_uncertainty_predictions, "depvar", [1, 2, 3], True
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
    manager = EvaluationManager(metrics_list=["RMSLE", "CRPS", "ABCD"])
    evaluation_dict, df_evaluation = manager._time_series_wise_evaluation(
        mock_actual, mock_point_predictions, "depvar", False
    )

    actuals = [[1, 2, 2, 3, 3, 4], [2, 3, 3, 4, 4, 5]]
    preds = [1, 3, 5, 7, 9, 7], [2, 4, 6, 8, 10, 8]
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
        index=["ts00", "ts01"],
    )

    assert ["ts00", "ts01"] == list(evaluation_dict.keys())
    assert np.allclose(df_evaluation, df_evaluation_test, atol=0.000001)


def test_time_series_wise_evaluation_uncertainty(mock_actual, mock_uncertainty_predictions):
    manager = EvaluationManager(metrics_list=["RMSLE", "CRPS", "ABCD"])
    evaluation_dict, df_evaluation = manager._time_series_wise_evaluation(
        mock_actual, mock_uncertainty_predictions, "depvar", True
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
    manager = EvaluationManager(metrics_list=["RMSLE", "CRPS", "ABCD"])
    evaluation_dict, df_evaluation = manager._month_wise_evaluation(
        mock_actual, mock_point_predictions, "depvar", False
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
    assert np.allclose(df_evaluation, df_evaluation_test, atol=0.000001)


def test_month_wise_evaluation_uncertainty(mock_actual, mock_uncertainty_predictions):
    manager = EvaluationManager(metrics_list=["RMSLE", "CRPS", "ABCD"])
    evaluation_dict, df_evaluation = manager._month_wise_evaluation(
        mock_actual, mock_uncertainty_predictions, "depvar", True
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
