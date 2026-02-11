from views_evaluation.manager.base import EvaluationManager
from views_evaluation.core import PointEvaluationMetrics, UncertaintyEvaluationMetrics
import logging
import pandas as pd
from typing import List

logger = logging.getLogger(__name__)


class StepshifterEvaluationManager(EvaluationManager):
    """
    A class for evaluating the predictions of a stepshifter model.
    """
    def __init__(self, metrics_list: list):
        super().__init__(metrics_list)
    
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

