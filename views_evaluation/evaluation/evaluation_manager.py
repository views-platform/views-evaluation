from typing import List, Dict, Tuple, Optional
import logging
import pandas as pd
import numpy as np
from views_evaluation.evaluation.metrics import (
    BaseEvaluationMetrics,
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


    @staticmethod
    def _split_dfs_by_step(dfs: list) -> list:
        """¨
        This function splits a list of DataFrames into a list of DataFrames by step, where the key is the step.
        For example, assume df0 has month_id from 100 to 102, df1 has month_id from 101 to 103, and df2 has month_id from 102 to 104.
        This function returns a list of three dataframes, with the first dataframe having month_id 100 from df0, month_id 101 from df1, and month_id 102 from df;
        the second dataframe having month_id 101 from df0, month_id 102 from df1, and month_id 103 from df2; and the third dataframe having month_id 102 from df1 and month_id 104 from df2.

        Args:
            dfs (list): List of DataFrames with overlapping time ranges.

        Returns:
            dict (list): A list of DataFrames where each contains one unique month_id from each input DataFrame.
        """
        time_id = dfs[0].index.names[0]
        all_month_ids = [df.index.get_level_values(0).unique() for df in dfs]

        grouped_month_ids = list(zip(*all_month_ids))

        result_dfs = []
        for i, group in enumerate(grouped_month_ids):
            step = i + 1
            combined = pd.concat(
                [df.loc[month_id] for df, month_id in zip(dfs, group)],
                keys=group,
                names=[time_id],
            )
            result_dfs.append(combined)

        return result_dfs

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

    def step_wise_evaluation(
        self,
        actual: pd.DataFrame,
        predictions: List[pd.DataFrame],
        target: str,
        steps: List[int],
        is_uncertainty: bool,
        **kwargs,
    ):
        """
        Evaluates the predictions step-wise and calculates the specified metrics.

        Args:
            actual (pd.DataFrame): The actual values.
            predictions (List[pd.DataFrame]): A list of DataFrames containing the predictions.
            target (str): The target column in the actual DataFrame.
            steps (List[int]): The steps to evaluate.
            is_uncertainty (bool): Flag to indicate if the evaluation is for uncertainty.

        Returns:
            Tuple: A tuple containing the evaluation dictionary and the evaluation DataFrame.
        """
        if is_uncertainty:
            evaluation_dict = (
                UncertaintyEvaluationMetrics.make_step_wise_evaluation_dict(
                    steps=max(steps)
                )
            )
            metric_functions = self.uncertainty_metric_functions
        else:
            evaluation_dict = PointEvaluationMetrics.make_step_wise_evaluation_dict(
                steps=max(steps)
            )
            metric_functions = self.point_metric_functions

        result_dfs = EvaluationManager._split_dfs_by_step(predictions)

        step_matched_data = {}
        for i, pred in enumerate(result_dfs):
            step = i + 1
            matched_actual, matched_pred = EvaluationManager._match_actual_pred(
                actual, pred, target
            )
            step_matched_data[step] = (matched_actual, matched_pred)

        for metric in self.metrics_list:
            if metric in metric_functions:
                for step, (matched_actual, matched_pred) in step_matched_data.items():
                    evaluation_dict[f"step{str(step).zfill(2)}"].__setattr__(
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

    def month_wise_evaluation(
        self,
        actual: pd.DataFrame,
        predictions: List[pd.DataFrame],
        target: str,
        is_uncertainty: bool,
        **kwargs,
    ):
        """
        Evaluates the predictions month-wise and calculates the specified metrics.

        Args:
            actual (pd.DataFrame): The actual values.
            predictions (List[pd.DataFrame]): A list of DataFrames containing the predictions.
            target (str): The target column in the actual DataFrame.
            is_uncertainty (bool): Flag to indicate if the evaluation is for uncertainty.

        Returns:
            Tuple: A tuple containing the evaluation dictionary and the evaluation DataFrame.
        """
        pred_concat = pd.concat(predictions)
        month_range = pred_concat.index.get_level_values(0).unique()
        month_start = int(month_range.min())
        month_end = int(month_range.max()) 

        if is_uncertainty:
            evaluation_dict = (
                UncertaintyEvaluationMetrics.make_month_wise_evaluation_dict(
                    month_start, month_end
                )
            )
            metric_functions = self.uncertainty_metric_functions
        else:
            evaluation_dict = PointEvaluationMetrics.make_month_wise_evaluation_dict(
                month_start, month_end
            )
            metric_functions = self.point_metric_functions

        matched_actual, matched_pred = EvaluationManager._match_actual_pred(
            actual, pred_concat, target
        )
        # matched_concat = pd.merge(matched_actual, matched_pred, left_index=True, right_index=True)
        
        g = matched_pred.groupby(level=matched_pred.index.names[0], sort=False, observed=True)
        groups = g.indices  # dict: {month -> np.ndarray of row positions}

        for metric in self.metrics_list:
            if metric in metric_functions:
                for month, pos in groups.items():
                    value = metric_functions[metric](
                        matched_actual.iloc[pos],
                        matched_pred.iloc[pos],
                        target,
                        **kwargs,
                    )
                    evaluation_dict[f"month{str(month)}"].__setattr__(metric, value)
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
        config: dict,
        **kwargs,
    ):
        """
        Evaluates the predictions and calculates the specified point metrics.

        Args:
            actual (pd.DataFrame): The actual values.
            predictions (List[pd.DataFrame]): A list of DataFrames containing the predictions.
            target (str): The target column in the actual DataFrame.
            config (dict): The configuration dictionary.
        """
        EvaluationManager.validate_predictions(predictions, target)
        self.actual, self.predictions = self._process_data(actual, predictions, target)
        self.is_uncertainty = EvaluationManager.get_evaluation_type(
            self.predictions, f"pred_{target}"
        )
        evaluation_results = {}
        evaluation_results["month"] = self.month_wise_evaluation(
            self.actual, self.predictions, target, self.is_uncertainty, **kwargs
        )

        evaluation_results["time_series"] = self.time_series_wise_evaluation(
            self.actual, self.predictions, target, self.is_uncertainty, **kwargs
        )

        evaluation_results["step"] = self.step_wise_evaluation(
            self.actual,
            self.predictions,
            target,
            config["steps"],
            self.is_uncertainty,
            **kwargs,
        )

        return evaluation_results

    @staticmethod
    def filter_step_wise_evaluation(
        step_wise_evaluation_results: dict,
        filter_steps: list[int] = [1, 3, 6, 12, 36],
    ):
        """
        Filter step-wise evaluation results to include only specific steps.

        Args:
            step_wise_evaluation_results (dict): The step-wise evaluation results containing evaluation dict and DataFrame.
            filter_steps (list[int]): List of step numbers to include in the filtered results. Defaults to [1, 3, 6, 12, 36].

        Returns:
            dict: A dictionary containing the filtered evaluation dictionary and DataFrame for the selected steps.
        """
        step_wise_evaluation_dict = step_wise_evaluation_results[0]
        step_wise_evaluation_df = step_wise_evaluation_results[1]

        selected_keys = [f"step{str(step).zfill(2)}" for step in filter_steps]

        filtered_evaluation_dict = {
            key: step_wise_evaluation_dict[key]
            for key in selected_keys
            if key in step_wise_evaluation_dict
        }

        filtered_evaluation_df = step_wise_evaluation_df.loc[
            step_wise_evaluation_df.index.isin(selected_keys)
        ]

        return (filtered_evaluation_dict, filtered_evaluation_df)

    @staticmethod
    def aggregate_month_wise_evaluation(
        month_wise_evaluation_results: dict,
        aggregation_period: int = 6,
        aggregation_type: str = "mean",
    ):
        """
        Aggregate month-wise evaluation results by grouping months into periods and applying aggregation.

        Args:
            month_wise_evaluation_results (dict): The month-wise evaluation results containing evaluation dict and DataFrame.
            aggregation_period (int): Number of months to group together for aggregation.
            aggregation_type (str): Type of aggregation to apply.
        Returns:
            dict: A dictionary containing the aggregated evaluation dictionary and DataFrame.
        """
        month_wise_evaluation_dict = month_wise_evaluation_results[0]
        month_wise_evaluation_df = month_wise_evaluation_results[1]

        available_months = [
            int(month.replace("month", "")) for month in month_wise_evaluation_df.index
        ]
        available_months.sort()

        if len(available_months) < aggregation_period:
            raise ValueError(
                f"Not enough months to aggregate. Available months: {available_months}, aggregation period: {aggregation_period}"
            )

        aggregated_dict = {}
        aggregated_data = []

        for i in range(0, len(available_months), aggregation_period):
            period_months = available_months[i : i + aggregation_period]
            period_start = period_months[0]
            period_end = period_months[-1]
            period_key = f"month_{period_start}_{period_end}"

            period_metrics = []
            for month in period_months:
                month_key = f"month{month}"
                if month_key in month_wise_evaluation_dict:
                    period_metrics.append(month_wise_evaluation_dict[month_key])

            if period_metrics:
                aggregated_metrics = {}
                for metric_name in period_metrics[0].__annotations__.keys():
                    metric_values = [
                        getattr(metric, metric_name)
                        for metric in period_metrics
                        if getattr(metric, metric_name) is not None
                    ]

                    if metric_values:
                        if aggregation_type == "mean":
                            aggregated_value = np.mean(metric_values)
                        elif aggregation_type == "median":
                            aggregated_value = np.median(metric_values)
                        else:
                            raise ValueError(
                                f"Unsupported aggregation type: {aggregation_type}"
                            )

                        aggregated_metrics[metric_name] = aggregated_value
                    else:
                        aggregated_metrics[metric_name] = None

                if hasattr(period_metrics[0], "__class__"):
                    aggregated_eval_metrics = period_metrics[0].__class__(
                        **aggregated_metrics
                    )
                else:
                    aggregated_eval_metrics = aggregated_metrics

                aggregated_dict[period_key] = aggregated_eval_metrics

                aggregated_data.append({"month_id": period_key, **aggregated_metrics})

        if aggregated_data:
            aggregated_df = BaseEvaluationMetrics.evaluation_dict_to_dataframe(
                aggregated_dict
            )

        return (aggregated_dict, aggregated_df)
