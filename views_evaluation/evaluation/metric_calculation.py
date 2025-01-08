from typing import List, Dict, Tuple
import logging
import pandas as pd
import numpy as np
from metrics import EvaluationMetrics
from sklearn.metrics import root_mean_squared_error, root_mean_squared_log_error, average_precision_score
import properscoring as ps

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    A class for calculating metrics on time series predictions
    Refer to https://github.com/prio-data/views_pipeline/blob/eval_docs/documentation/evaluation/schema.MD for more details on three evaluation schemas.
    """

    def __init__(self, metrics_list):
        """
        Initialize the calculator with a list of metric names to calculate.

        Args:
            metrics_list (List[str]): A list of metric names to evaluate.
        """

        self.metrics_list = metrics_list
        self.metric_functions = {
            "RMSLE": self._calculate_rmsle,
            "CRPS": self._calculate_crps,
            "AP": self._calculate_ap,
            "Brier": self._calculate_brier,
            "Jeffreys": self._calculate_jeffreys,
            "Coverage": self._calculate_coverage,
            "EMD": self._calculate_emd,
            "SD": self._calculate_sd,
            "pEMDiv": self._calculate_pEMDiv,
            "Pearson": self._calculate_pearson,
            "Variogram": self._calculate_variogram,
        }

    def time_series_wise_evaluation(
            self, actual: pd.DataFrame, predictions: List[pd.DataFrame], target: str
            ) -> pd.DataFrame:
        """
        Evaluates the predictions time series-wise and calculates the specified metrics.

        Args:
            actual (pd.DataFrame): The actual values.
            predictions (List[pd.DataFrame]): A list of DataFrames containing the predictions.
            target (str): The target column in the actual DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the evaluation metrics.
        """
        evaluation_dict = EvaluationMetrics.make_time_series_wise_evaluation_dict(len(predictions))
        
        for metric in self.metrics_list:
            if metric in self.metric_functions:
                for i, pred in enumerate(predictions):
                    matched_actual, matched_pred = MetricsCalculator.match_actual_pred(actual, pred, target)
                    evaluation_dict[f"ts{str(i).zfill(2)}"].__setattr__(
                        metric, self.metric_functions[metric](matched_actual, matched_pred, target)
                    )
            else:
                logger.warning(f"Metric {metric} is not a default metric, skipping...")

        return EvaluationMetrics.evaluation_dict_to_dataframe(evaluation_dict)   

    def step_wise_evaluation(
            self, actual: pd.DataFrame, predictions: List[pd.DataFrame], target: str, steps: List[int]
            ) -> pd.DataFrame:
        """
        Evaluates the predictions step-wise and calculates the specified metrics.

        Args:
            actual (pd.DataFrame): The actual values.
            predictions (List[pd.DataFrame]): A list of DataFrames containing the predictions.
            target (str): The target column in the actual DataFrame.
            steps (List[int]): The steps to evaluate.

        Returns:
            pd.DataFrame: A DataFrame containing the evaluation metrics.
        """ 
        evaluation_dict = EvaluationMetrics.make_step_wise_evaluation_dict(steps=max(steps))
        step_metrics = {}

        for pred in predictions:
            for i, month in enumerate(pred.index.levels[0]):
                step = i + 1
                step_pred = pred.loc[pred.index.get_level_values(0) == month]
                matched_actual, matched_pred = MetricsCalculator.match_actual_pred(actual, step_pred, target)
                for metric in self.metrics_list:
                    if metric in self.metric_functions:
                        metric_value = self.metric_functions[metric](matched_actual, matched_pred, target)
                        if step not in step_metrics:
                            step_metrics[step] = {m: [] for m in self.metrics_list}
                        step_metrics[step][metric].append(metric_value)
                    
        
        for metric in self.metrics_list:
            if metric in self.metric_functions:
                for step in steps:
                    evaluation_dict[f"step{str(step).zfill(2)}"].__setattr__(
                        metric, np.mean(step_metrics[step][metric])
                    )
            else:
                logger.warning(f"Metric {metric} is not a default metric, skipping...")
            
        return EvaluationMetrics.evaluation_dict_to_dataframe(evaluation_dict)
    
    def month_wise_evaluation(
            self, actual: pd.DataFrame, predictions: List[pd.DataFrame], target: str
            ) -> pd.DataFrame:
        """
        Evaluates the predictions month-wise and calculates the specified metrics.

        Args:
            actual (pd.DataFrame): The actual values.
            predictions (List[pd.DataFrame]): A list of DataFrames containing the predictions.
            target (str): The target column in the actual DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the evaluation metrics.
        """
        pred_concat = pd.concat(predictions)
        pred_concat_target = pred_concat.columns[0]
        month_range = pred_concat.index.get_level_values(0).unique()
        month_start = month_range.min()
        month_end = month_range.max()
        evaluation_dict = EvaluationMetrics.make_month_wise_evaluation_dict(month_start, month_end)

        matched_actual, matched_pred = MetricsCalculator.match_actual_pred(actual, pred_concat, target)
        matched_concat = pd.merge(matched_actual, matched_pred, left_index=True, right_index=True)

        for metric in self.metrics_list:
            if metric in self.metric_functions:
                metric_by_month = matched_concat.groupby(level=matched_concat.index.names[0]).apply(
                    lambda df: self.metric_functions[metric](
                        df[[target]], df[[pred_concat_target]], target
                    )
                )

                for month in month_range:
                    evaluation_dict[f"month{str(month)}"].__setattr__(metric, metric_by_month.loc[month])
            else:
                logger.warning(f"Metric {metric} is not a default metric, skipping...")

        return EvaluationMetrics.evaluation_dict_to_dataframe(evaluation_dict)

    @staticmethod
    def match_actual_pred(actual: pd.DataFrame, pred: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        if target not in actual.columns:
            raise ValueError(f"Target column '{target}' not found in actual DataFrame.")

        actual_target = actual[[target]]
        aligned_actual, aligned_pred = actual_target.align(pred, join="inner")
        matched_actual = aligned_actual.reindex(index=aligned_pred.index)
        matched_actual[[target]] = actual_target

        return matched_actual.sort_index(), pred.sort_index()
    
    @staticmethod
    def _calculate_rmsle(matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str) -> float:
        return (
            root_mean_squared_error(matched_actual, matched_pred)
            if target.startswith("ln")
            else root_mean_squared_log_error(matched_actual, matched_pred)
            )

    @staticmethod
    def _calculate_crps(matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str) -> float:
        return ps.crps_ensemble(matched_actual, matched_pred).mean()

    @staticmethod
    def _calculate_ap(matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str) -> float:
        pass

    @staticmethod
    def _calculate_brier(matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str) -> float:
        pass

    @staticmethod
    def _calculate_jeffreys(matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str) -> float:
        pass

    @staticmethod
    def _calculate_coverage(matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str) -> float:
        pass

    @staticmethod
    def _calculate_emd(matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str) -> float:
        pass

    @staticmethod
    def _calculate_sd(matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str) -> float:
        pass

    @staticmethod
    def _calculate_pEMDiv(matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str) -> float:
        pass

    @staticmethod
    def _calculate_pearson(matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str) -> float:
        pass

    @staticmethod
    def _calculate_variogram(matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str) -> float:
        pass

