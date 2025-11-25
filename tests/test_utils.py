import pandas as pd
import pytest
from views_evaluation.evaluation.utils import DataUtils

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
    return DataUtils.convert_to_array(df, "target")


@pytest.fixture
def mock_point_predictions(mock_index):
    df1 = pd.DataFrame({"pred_target": [1.0, 3.0, 5.0, 7.0, 9.0, 7.0]}, index=mock_index[0])
    df2 = pd.DataFrame({"pred_target": [2.0, 4.0, 6.0, 8.0, 10.0, 8.0]}, index=mock_index[1])
    return [DataUtils.convert_to_array(df1, "pred_target"), DataUtils.convert_to_array(df2, "pred_target")]


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
    return [DataUtils.convert_to_array(df1, "pred_target"), DataUtils.convert_to_array(df2, "pred_target")]


def test_validate_dataframes_valid_type(mock_point_predictions):
    with pytest.raises(TypeError):
        DataUtils.validate_predictions(
            mock_point_predictions[0], "target"
        )


def test_validate_dataframes_valid_columns(mock_point_predictions):
    with pytest.raises(ValueError):
        DataUtils.validate_predictions(
            mock_point_predictions, "y"
        )

def test_get_evaluation_type():
    # Test case 1: All DataFrames for uncertainty evaluation
    predictions_uncertainty = [
        pd.DataFrame({'pred_target': [[1.0, 2.0], [3.0, 4.0]]}),
        pd.DataFrame({'pred_target': [[5.0, 6.0], [7.0, 8.0]]}),
    ]
    assert DataUtils.get_evaluation_type(predictions_uncertainty, "pred_target") == True

    # Test case 2: All DataFrames for point evaluation
    predictions_point = [
        pd.DataFrame({'pred_target': [[1.0], [2.0]]}),
        pd.DataFrame({'pred_target': [[3.0], [4.0]]}),
    ]
    assert DataUtils.get_evaluation_type(predictions_point, "pred_target") == False

    # Test case 3: Mixed evaluation types
    predictions_mixed = [
        pd.DataFrame({'pred_target': [[1.0, 2.0], [3.0, 4.0]]}),
        pd.DataFrame({'pred_target': [[5.0], [6.0]]}),
    ]
    with pytest.raises(ValueError):
        DataUtils.get_evaluation_type(predictions_mixed, "pred_target")

    # Test case 4: Single element lists
    predictions_single_element = [
        pd.DataFrame({'pred_target': [[1.0], [2.0]]}),
        pd.DataFrame({'pred_target': [[3.0], [4.0]]}),
    ]
    assert DataUtils.get_evaluation_type(predictions_single_element, "pred_target") == False


def test_match_actual_pred_point(
    mock_actual, mock_point_predictions, mock_uncertainty_predictions, mock_index
):
    df_matched = [
        pd.DataFrame({"target": [[1.0], [2.0], [2.0], [3.0], [3.0], [4.0]]}, index=mock_index[0]),
        pd.DataFrame({"target": [[2.0], [3.0], [3.0], [4.0], [4.0], [5.0]]}, index=mock_index[1]),
    ]
    for i in range(len(df_matched)):
        df_matched_actual_point, df_matched_point = (
            DataUtils.match_actual_pred(
                mock_actual, mock_point_predictions[i], "target"
            )
        )
        df_matched_actual_uncertainty, df_matched_uncertainty = (
            DataUtils.match_actual_pred(
                mock_actual, mock_uncertainty_predictions[i], "target"
            )
        )
        assert df_matched[i].equals(df_matched_actual_point)
        assert df_matched_point.equals(mock_point_predictions[i])
        assert df_matched[i].equals(df_matched_actual_uncertainty)
        assert df_matched_uncertainty.equals(mock_uncertainty_predictions[i])


def test_split_dfs_by_step(mock_point_predictions, mock_uncertainty_predictions):
    df_splitted_point = [
        DataUtils.convert_to_array(pd.DataFrame(
            {"pred_target": [[1.0], [3.0], [2.0], [4.0]]},
            index=pd.MultiIndex.from_tuples(
                [(100, 1), (100, 2), (101, 1), (101, 2)], names=["month", "country"]
            ),
        ), "pred_target"),
        DataUtils.convert_to_array(pd.DataFrame(
            {"pred_target": [[5.0], [7.0], [6.0], [8.0]]},
            index=pd.MultiIndex.from_tuples(
                [(101, 1), (101, 2), (102, 1), (102, 2)], names=["month", "country"]
            ),
        ), "pred_target"),
        DataUtils.convert_to_array(pd.DataFrame(
            {"pred_target": [[9.0], [7.0], [10.0], [8.0]]},
            index=pd.MultiIndex.from_tuples(
                [(102, 1), (102, 2), (103, 1), (103, 2)], names=["month", "country"]
            ),
        ), "pred_target"),
    ]
    df_splitted_uncertainty = [
        DataUtils.convert_to_array(pd.DataFrame(
            {"pred_target": [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [4.0, 6.0, 8.0], [5.0, 7.0, 9.0]]},
            index=pd.MultiIndex.from_tuples(
                [(100, 1), (100, 2), (101, 1), (101, 2)], names=["month", "country"]
            ),
        ), "pred_target"),
        DataUtils.convert_to_array(pd.DataFrame(
            {"pred_target": [[3.0, 4.0, 5.0], [4.0, 5.0, 6.0], [6.0, 8.0, 10.0], [7.0, 9.0, 11.0]]},
            index=pd.MultiIndex.from_tuples(
                [(101, 1), (101, 2), (102, 1), (102, 2)], names=["month", "country"]
            ),
        ), "pred_target"),
        DataUtils.convert_to_array(pd.DataFrame(
            {"pred_target": [[5.0, 6.0, 7.0], [6.0, 7.0, 8.0], [8.0, 10.0, 12.0], [9.0, 11.0, 13.0]]},
            index=pd.MultiIndex.from_tuples(
                [(102, 1), (102, 2), (103, 1), (103, 2)], names=["month", "country"]
            ),
        ), "pred_target"),
    ]
    df_splitted_point_test = DataUtils.split_dfs_by_step(
        mock_point_predictions
    )
    df_splitted_uncertainty_test = DataUtils.split_dfs_by_step(
        mock_uncertainty_predictions
    )
    for df1, df2 in zip(df_splitted_point, df_splitted_point_test):
        assert df1.equals(df2)
    for df1, df2 in zip(df_splitted_uncertainty, df_splitted_uncertainty_test):
        assert df1.equals(df2)