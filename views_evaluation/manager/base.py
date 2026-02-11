from typing import List, Tuple
import logging
import pandas as pd
import numpy as np
from views_evaluation.evaluation.metrics import (
    PointEvaluationMetrics,
    UncertaintyEvaluationMetrics,
)
from views_evaluation.evaluation.metric_calculators import (
    POINT_METRIC_FUNCTIONS,
    UNCERTAINTY_METRIC_FUNCTIONS,
)

logger = logging.getLogger(__name__)


class EvaluationManager:
    """
    A class for calculating metrics on time series predictions
    Refer to https://github.com/prio-data/views_pipeline/blob/eval_docs/documentation/evaluation/schema.MD for more details on three evaluation schemas.
    """

    def __init__(self, metrics_list: list):
        """
        Initialize the manager with a list of metric names to calculate.

        Args:
            metrics_list (List[str]): A list of metric names to evaluate.
        """

        self.metrics_list = metrics_list
        self.point_metric_functions = POINT_METRIC_FUNCTIONS
        self.uncertainty_metric_functions = UNCERTAINTY_METRIC_FUNCTIONS

    @staticmethod
    def transform_data(df: pd.DataFrame, target: str | list[str]) -> pd.DataFrame:
        """
        Transform the data.
        """
        if isinstance(target, str):
            target = [target]
        for t in target:
            if t.startswith("ln") or t.startswith("pred_ln"):
                df[[t]] = df[[t]].applymap(
                    lambda x: (
                        np.exp(x) - 1
                        if isinstance(x, (list, np.ndarray))
                        else np.exp(x) - 1
                    )
                )
            elif t.startswith("lx") or t.startswith("pred_lx"):
                df[[t]] = df[[t]].applymap(
                    lambda x: (
                        np.exp(x) - np.exp(100)
                        if isinstance(x, (list, np.ndarray))
                        else np.exp(x) - np.exp(100)
                    )
                )
            elif t.startswith("lr") or t.startswith("pred_lr"):
                df[[t]] = df[[t]].applymap(
                    lambda x: x if isinstance(x, (list, np.ndarray)) else x
                )
            else:
                raise ValueError(f"Target {t} is not a valid target")
        return df

    @staticmethod
    def convert_to_array(df: pd.DataFrame, target: str | list[str]) -> pd.DataFrame:
        """
        Convert columns in a DataFrame to numpy arrays.

        Args:
            df (pd.DataFrame): The input DataFrame with columns that may contain lists.

        Returns:
            pd.DataFrame: A new DataFrame with columns converted to numpy arrays.
        """
        converted = df.copy()
        if isinstance(target, str):
            target = [target]

        for t in target:
            converted[t] = converted[t].apply(
                lambda x: (
                    x
                    if isinstance(x, np.ndarray)
                    else (np.array(x) if isinstance(x, list) else np.array([x]))
                )
            )
        return converted

    @staticmethod
    def convert_to_scalar(df: pd.DataFrame, target: str | list[str]) -> pd.DataFrame:
        """
        Convert columns in a DataFrame to scalar values by taking the mean of the list.
        """
        converted = df.copy()
        if isinstance(target, str):
            target = [target]
        for t in target:
            converted[t] = converted[t].apply(
                lambda x: np.mean(x) if isinstance(x, (list, np.ndarray)) else x
            )
        return converted

    @staticmethod
    def get_evaluation_type(predictions: List[pd.DataFrame], target: str) -> bool:
        """
        Validates the values in each DataFrame in the list.
        The return value indicates whether all DataFrames are for uncertainty evaluation.

        Args:
            predictions (List[pd.DataFrame]): A list of DataFrames to check.

        Returns:
            bool: True if all DataFrames are for uncertainty evaluation,
                  False if all DataFrame are for point evaluation.

        Raises:
            ValueError: If there is a mix of single and multiple values in the lists,
                      or if uncertainty lists have different lengths.
        """
        is_uncertainty = False
        is_point = False
        uncertainty_length = None

        for df in predictions:
            for value in df[target].values.flatten():
                if not (isinstance(value, np.ndarray) or isinstance(value, list)):
                    raise ValueError(
                        "All values must be lists or numpy arrays. Convert the data."
                    )

                if len(value) > 1:
                    is_uncertainty = True
                    # For uncertainty evaluation, check that all lists have the same length
                    if uncertainty_length is None:
                        uncertainty_length = len(value)
                    elif len(value) != uncertainty_length:
                        raise ValueError(
                            f"Inconsistent list lengths in uncertainty evaluation. "
                            f"Found lengths {uncertainty_length} and {len(value)}"
                        )
                elif len(value) == 1:
                    is_point = True
                else:
                    raise ValueError("Empty lists are not allowed")

        if is_uncertainty and is_point:
            raise ValueError(
                "Mix of evaluation types detected: some rows contain single values, others contain multiple values. "
                "Please ensure all rows are consistent in their evaluation type"
            )

        return is_uncertainty

    @staticmethod
    def validate_predictions(predictions: List[pd.DataFrame], target: str):
        """
        Checks if the predictions are valid DataFrames.
        - Each DataFrame must have exactly one column named `pred_column_name`.

        Args:
            predictions (List[pd.DataFrame]): A list of DataFrames containing the predictions.
            target (str): The target column in the actual DataFrame.
        """
        pred_column_name = f"pred_{target}"
        if not isinstance(predictions, list):
            raise TypeError("Predictions must be a list of DataFrames.")

        for i, df in enumerate(predictions):
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"Predictions[{i}] must be a DataFrame.")
            if df.empty:
                raise ValueError(f"Predictions[{i}] must not be empty.")
            if pred_column_name not in df.columns:
                raise ValueError(
                    f"Predictions[{i}] must contain the column named '{pred_column_name}'."
                )

    @staticmethod
    def _match_actual_pred(
        actual: pd.DataFrame, pred: pd.DataFrame, target: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Matches the actual and predicted DataFrames based on the index and target column.

        Parameters:
        - actual: pd.DataFrame with a MultiIndex (e.g., month, level).
        - pred: pd.DataFrame with a MultiIndex that may contain duplicated indices.
        - target: str, the target column in actual.

        Returns:
        - matched_actual: pd.DataFrame aligned with pred.
        - matched_pred: pd.DataFrame aligned with actual.
        """
        actual_target = actual[[target]]
        common_indices = actual_target.index.intersection(pred.index)
        matched_pred = pred[pred.index.isin(common_indices)].copy()
        
        # Create matched_actual by reindexing actual_target to match pred's index structure
        # This will duplicate rows in actual where pred has duplicate indices
        matched_actual = actual_target.reindex(matched_pred.index)
        
        matched_actual = matched_actual.sort_index()
        matched_pred = matched_pred.sort_index()

        return matched_actual, matched_pred

    def _process_data(
        self, actual: pd.DataFrame, predictions: List[pd.DataFrame], target: str
    ):
        """
        Process the data for evaluation.
        """
        actual = EvaluationManager.transform_data(
            EvaluationManager.convert_to_array(actual, target), target
        )
        predictions = [
            EvaluationManager.transform_data(
                EvaluationManager.convert_to_array(pred, f"pred_{target}"),
                f"pred_{target}",
            )
            for pred in predictions
        ]
        return actual, predictions

    def time_series_wise_evaluation(
        self,
        actual: pd.DataFrame,
        predictions: List[pd.DataFrame],
        target: str,
        is_uncertainty: bool,
        **kwargs,
    ):
        """
        Evaluates the predictions time series-wise and calculates the specified metrics.

        Args:
            actual (pd.DataFrame): The actual values.
            predictions (List[pd.DataFrame]): A list of DataFrames containing the predictions.
            target (str): The target column in the actual DataFrame.
            is_uncertainty (bool): Flag to indicate if the evaluation is for uncertainty.

        Returns:
            Tuple: A tuple containing the evaluation dictionary and the evaluation DataFrame.
        """
        if is_uncertainty:
            evaluation_dict = (
                UncertaintyEvaluationMetrics.make_time_series_wise_evaluation_dict(
                    len(predictions)
                )
            )
            metric_functions = self.uncertainty_metric_functions
        else:
            evaluation_dict = (
                PointEvaluationMetrics.make_time_series_wise_evaluation_dict(
                    len(predictions)
                )
            )
            metric_functions = self.point_metric_functions

        ts_matched_data = {}
        for i, pred in enumerate(predictions):
            matched_actual, matched_pred = EvaluationManager._match_actual_pred(
                actual, pred, target
            )
            ts_matched_data[i] = (matched_actual, matched_pred)

        for metric in self.metrics_list:
            if metric in metric_functions:
                for i, (matched_actual, matched_pred) in ts_matched_data.items():
                    evaluation_dict[f"ts{str(i).zfill(2)}"].__setattr__(
                        metric,
                        metric_functions[metric](
                            matched_actual, matched_pred, target, **kwargs
                        ),
                    )
            else:
                logger.warning(f"Metric {metric} is not a default metric, skipping...")

        return (
            evaluation_dict,
            PointEvaluationMetrics.evaluation_dict_to_dataframe(evaluation_dict),
        )

    def evaluate(
        self,
        actual: pd.DataFrame,
        predictions: List[pd.DataFrame],
        target: str,
        **kwargs,
    ):
        """
        Evaluates the predictions and calculates the specified point metrics for time series-wise evaluation.

        Args:
            actual (pd.DataFrame): The actual values.
            predictions (List[pd.DataFrame]): A list of DataFrames containing the predictions.
            target (str): The target column in the actual DataFrame.

        Returns:
            dict: A dictionary containing the evaluation results.
        """
        EvaluationManager.validate_predictions(predictions, target)
        self.actual, self.predictions = self._process_data(actual, predictions, target)
        self.is_uncertainty = EvaluationManager.get_evaluation_type(
            self.predictions, f"pred_{target}"
        )
        evaluation_results = {}

        evaluation_results["time_series"] = self.time_series_wise_evaluation(
            self.actual, self.predictions, target, self.is_uncertainty, **kwargs
        )

        return evaluation_results


