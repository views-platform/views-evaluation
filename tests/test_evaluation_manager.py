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
            "target": [0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0],
            "covariate_1": [3.0, 2.0, 4.0, 5.0, 2.0, 6.0, 8.0, 5.0, 3.0, 2.0, 9.0, 4.0],
        },
        index=index,
    )
    return EvaluationManager.convert_to_arrays(df)


@pytest.fixture
def mock_point_predictions(mock_index):
    df1 = pd.DataFrame({"pred_target": [1.0, 3.0, 5.0, 7.0, 9.0, 7.0]}, index=mock_index[0])
    df2 = pd.DataFrame({"pred_target": [2.0, 4.0, 6.0, 8.0, 10.0, 8.0]}, index=mock_index[1])
    return [EvaluationManager.convert_to_arrays(df1), EvaluationManager.convert_to_arrays(df2)]


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
    return [EvaluationManager.convert_to_arrays(df1), EvaluationManager.convert_to_arrays(df2)]


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
    # Test case 1: All DataFrames for uncertainty evaluation
    predictions_uncertainty = [
        pd.DataFrame({'pred_target': [[1.0, 2.0], [3.0, 4.0]]}),
        pd.DataFrame({'pred_target': [[5.0, 6.0], [7.0, 8.0]]}),
    ]
    assert EvaluationManager.get_evaluation_type(predictions_uncertainty) == True

    # Test case 2: All DataFrames for point evaluation
    predictions_point = [
        pd.DataFrame({'pred_target': [[1.0], [2.0]]}),
        pd.DataFrame({'pred_target': [[3.0], [4.0]]}),
    ]
    assert EvaluationManager.get_evaluation_type(predictions_point) == False

    # Test case 3: Mixed evaluation types
    predictions_mixed = [
        pd.DataFrame({'pred_target': [[1.0, 2.0], [3.0, 4.0]]}),
        pd.DataFrame({'pred_target': [[5.0], [6.0]]}),
    ]
    with pytest.raises(ValueError):
        EvaluationManager.get_evaluation_type(predictions_mixed)

    # Test case 4: Single element lists
    predictions_single_element = [
        pd.DataFrame({'pred_target': [[1.0], [2.0]]}),
        pd.DataFrame({'pred_target': [[3.0], [4.0]]}),
    ]
    assert EvaluationManager.get_evaluation_type(predictions_single_element) == False


def test_match_actual_pred_point(
    mock_actual, mock_point_predictions, mock_uncertainty_predictions, mock_index
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
        df_matched_actual_uncertainty, df_matched_uncertainty = (
            EvaluationManager._match_actual_pred(
                mock_actual, mock_uncertainty_predictions[i], "target"
            )
        )
        assert df_matched[i].equals(df_matched_actual_point)
        assert df_matched_point.equals(mock_point_predictions[i])
        assert df_matched[i].equals(df_matched_actual_uncertainty)
        assert df_matched_uncertainty.equals(mock_uncertainty_predictions[i])


def test_split_dfs_by_step(mock_point_predictions, mock_uncertainty_predictions):
    df_splitted_point = [
        EvaluationManager.convert_to_arrays(pd.DataFrame(
            {"pred_target": [[1.0], [3.0], [2.0], [4.0]]},
            index=pd.MultiIndex.from_tuples(
                [(100, 1), (100, 2), (101, 1), (101, 2)], names=["month", "country"]
            ),
        )),
        EvaluationManager.convert_to_arrays(pd.DataFrame(
            {"pred_target": [[5.0], [7.0], [6.0], [8.0]]},
            index=pd.MultiIndex.from_tuples(
                [(101, 1), (101, 2), (102, 1), (102, 2)], names=["month", "country"]
            ),
        )),
        EvaluationManager.convert_to_arrays(pd.DataFrame(
            {"pred_target": [[9.0], [7.0], [10.0], [8.0]]},
            index=pd.MultiIndex.from_tuples(
                [(102, 1), (102, 2), (103, 1), (103, 2)], names=["month", "country"]
            ),
        )),
    ]
    df_splitted_uncertainty = [
        EvaluationManager.convert_to_arrays(pd.DataFrame(
            {"pred_target": [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [4.0, 6.0, 8.0], [5.0, 7.0, 9.0]]},
            index=pd.MultiIndex.from_tuples(
                [(100, 1), (100, 2), (101, 1), (101, 2)], names=["month", "country"]
            ),
        )),
        EvaluationManager.convert_to_arrays(pd.DataFrame(
            {"pred_target": [[3.0, 4.0, 5.0], [4.0, 5.0, 6.0], [6.0, 8.0, 10.0], [7.0, 9.0, 11.0]]},
            index=pd.MultiIndex.from_tuples(
                [(101, 1), (101, 2), (102, 1), (102, 2)], names=["month", "country"]
            ),
        )),
        EvaluationManager.convert_to_arrays(pd.DataFrame(
            {"pred_target": [[5.0, 6.0, 7.0], [6.0, 7.0, 8.0], [8.0, 10.0, 12.0], [9.0, 11.0, 13.0]]},
            index=pd.MultiIndex.from_tuples(
                [(102, 1), (102, 2), (103, 1), (103, 2)], names=["month", "country"]
            ),
        )),
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
    evaluation_dict, df_evaluation = manager.step_wise_evaluation(
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
    assert np.allclose(df_evaluation, df_evaluation_test, atol=0.000001)


def test_step_wise_evaluation_uncertainty(mock_actual, mock_uncertainty_predictions):
    manager = EvaluationManager(metrics_list=["RMSLE", "CRPS", "ABCD"])
    evaluation_dict, df_evaluation = manager.step_wise_evaluation(
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
    assert np.allclose(df_evaluation, df_evaluation_test, atol=0.000001)


def test_time_series_wise_evaluation_point(mock_actual, mock_point_predictions):
    manager = EvaluationManager(metrics_list=["RMSLE", "CRPS", "ABCD"])
    evaluation_dict, df_evaluation = manager.time_series_wise_evaluation(
        mock_actual, mock_point_predictions, "target", False
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
    evaluation_dict, df_evaluation = manager.time_series_wise_evaluation(
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
    assert np.allclose(df_evaluation, df_evaluation_test, atol=0.000001)


def test_month_wise_evaluation_point(mock_actual, mock_point_predictions):
    manager = EvaluationManager(metrics_list=["RMSLE", "CRPS", "ABCD"])
    evaluation_dict, df_evaluation = manager.month_wise_evaluation(
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
    assert np.allclose(df_evaluation, df_evaluation_test, atol=0.000001)


def test_month_wise_evaluation_uncertainty(mock_actual, mock_uncertainty_predictions):
    manager = EvaluationManager(metrics_list=["RMSLE", "CRPS", "ABCD"])
    evaluation_dict, df_evaluation = manager.month_wise_evaluation(
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
    assert np.allclose(df_evaluation, df_evaluation_test, atol=0.000001)


def test_calculate_ap_point_predictions():
    actual_data = {'target': [[40], [20], [35], [25]]}
    pred_data = {'pred_target': [[35], [30], [20], [15]]}
    threshold=30
    
    matched_actual = pd.DataFrame(actual_data)
    matched_pred = pd.DataFrame(pred_data)
    
    ap_score = EvaluationManager._calculate_ap(matched_actual, matched_pred, 'target', threshold)
    
    actual_binary = [1, 0, 1, 0]  # 40>30, 20<30, 35>30, 25<30
    pred_binary = [1, 1, 0, 0]    # 35>30, 30=30, 20<30, 15<30
    from sklearn.metrics import average_precision_score
    expected_ap = average_precision_score(actual_binary, pred_binary)
    
    assert abs(ap_score - expected_ap) < 0.01


def test_calculate_ap_uncertainty_predictions():
    actual_data = {'target': [[40], [20], [35], [25]]}
    pred_data = {
        'pred_target': [
            [35, 40, 45],
            [30, 35, 40],
            [20, 25, 30],
            [15, 20, 25]
        ]
    }
    threshold=30
    matched_actual = pd.DataFrame(actual_data)
    matched_pred = pd.DataFrame(pred_data)
    
    ap_score = EvaluationManager._calculate_ap(matched_actual, matched_pred, 'target', threshold)
    
    pred_values = [35, 40, 45, 30, 35, 40, 20, 25, 30, 15, 20, 25]
    actual_values = [40, 40, 40, 20, 20, 20, 35, 35, 35, 25, 25, 25]
    actual_binary = [1 if x > threshold else 0 for x in actual_values]
    pred_binary = [1 if x >= threshold else 0 for x in pred_values]

    from sklearn.metrics import average_precision_score
    expected_ap = average_precision_score(actual_binary, pred_binary)
    
    assert abs(ap_score - expected_ap) < 0.01




