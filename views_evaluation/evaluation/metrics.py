from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
import pandas as pd
import numpy as np
import properscoring as ps
from statistics import mean, stdev, median
from sklearn.metrics import root_mean_squared_error, root_mean_squared_log_error, average_precision_score

logger = logging.getLogger(__name__)


# ============================================================ Metrics Dataclass ============================================================

@dataclass
class EvaluationMetrics:
    """
    A data class for storing and managing evaluation metrics for time series forecasting models.
    
    Attributes:
        RMSLE (Optional[float]): Root Mean Squared Logarithmic Error.
        CRPS (Optional[float]): Continuous Ranked Probability Score.
        AP (Optional[float]): Average Precision.
        Brier (Optional[float]): Brier Score.
        Jeffreys (Optional[float]): Jeffreys Divergence.
        Coverage (Optional[float]): Coverage (Histograms).
        EMD (Optional[float]): Earth Mover Distance.
        SD (Optional[float]): Sinkhorn Distance.
        pEMDiv (Optional[float]): pseudo-Earth Mover Divergence.
        Pearson (Optional[float]): Pearson Correlation.
        Variogram (Optional[float]): Variogram.
    """

    RMSLE: Optional[float] = None
    CRPS: Optional[float] = None
    AP: Optional[float] = None
    Brier: Optional[float] = None
    Jeffreys: Optional[float] = None
    Coverage: Optional[float] = None
    EMD: Optional[float] = None
    SD: Optional[float] = None
    pEMDiv: Optional[float] = None
    Pearson: Optional[float] = None
    Variogram: Optional[float] = None

    @classmethod
    def make_time_series_wise_evaluation_dict(cls, time_series_length: int =12) -> dict:
        """
        Generates a dictionary of EvaluationMetrics instances for a specified number of time series.

        This method facilitates the batch creation of metric containers for multiple time series, initializing them with None.

        Args:
            time_series_length (int): The number of time series for which to generate evaluation metrics. Defaults to 12.

        Returns:
            dict: A dictionary where each key is a step label (e.g., 'ts01', 'ts02', ...) and each value is an instance of EvaluationMetrics.

        Example:
            >>> from utils_evaluation_metrics import EvaluationMetrics
            >>> evaluation_dict = EvaluationMetrics.make_evaluation_dict(time_series_length=12)
            >>> evaluation_dict['ts01'].MSE = sklearn.metrics.mean_squared_error(ts01_y_true, ts01_y_pred)
            >>> evaluation_dict['ts02'].MSE = sklearn.metrics.mean_squared_error(ts02_y_true, ts02_y_pred)
            >>> ...
            
        """
        return {f"ts{str(i).zfill(2)}": cls() for i in range(0, time_series_length)}
    
    @classmethod
    def make_step_wise_evaluation_dict(cls, steps: int =36) -> dict:
        """
        Generates a dictionary of EvaluationMetrics instances for a specified number of steps.

        This method facilitates the batch creation of metric containers for multiple steps, initializing them with None.

        Args:
            steps (int): The number of forecasting steps for which to generate evaluation metrics. Defaults to 36.

        Returns:
            dict: A dictionary where each key is a step label (e.g., 'step01', 'step02', ...) and each value is an instance of EvaluationMetrics.

        Example:
            >>> from utils_evaluation_metrics import EvaluationMetrics
            >>> evaluation_dict = EvaluationMetrics.make_evaluation_dict(steps=36)
            >>> evaluation_dict['step01'].MSE = sklearn.metrics.mean_squared_error(step01_y_true, step01_y_pred)
            >>> evaluation_dict['step02'].MSE = sklearn.metrics.mean_squared_error(step02_y_true, step02_y_pred)
            >>> ...
            
        """
        return {f"step{str(i).zfill(2)}": cls() for i in range(1, steps + 1)}
    
    @classmethod
    def make_month_wise_evaluation_dict(cls, month_start: int, month_end: int) -> dict:
        """
        Generates a dictionary of EvaluationMetrics instances for a specified range of months.

        This method facilitates the batch creation of metric containers for multiple months, initializing them with None.

        Args:
            month_start (int): The first month for which to generate evaluation metrics.
            month_end (int): The last month for which to generate evaluation metrics.

        Returns:
            dict: A dictionary where each key is a step label (e.g., 'month501', 'month502', ...) and each value is an instance of EvaluationMetrics.

        Example:
            >>> from utils_evaluation_metrics import EvaluationMetrics
            >>> evaluation_dict = EvaluationMetrics.make_evaluation_dict(month_start=501, month_end=548)
            >>> evaluation_dict['month501'].MSE = sklearn.metrics.mean_squared_error(month501_y_true, month501_y_pred)
            >>> evaluation_dict['month502'].MSE = sklearn.metrics.mean_squared_error(month502_y_true, month502_y_pred)
            >>> ...
            
        """
        return {f"month{str(i)}": cls() for i in range(month_start, month_end + 1)}

    @staticmethod
    def evaluation_dict_to_dataframe(evaluation_dict: dict) -> pd.DataFrame:
        """
        Converts a dictionary of EvaluationMetrics instances into a pandas DataFrame for easy analysis.

        This static method transforms a structured dictionary of evaluation metrics into a DataFrame, where each row corresponds to a forecasting step and columns represent different metrics.

        Args:
            evaluation_dict (dict): A dictionary of EvaluationMetrics instances, typically generated by the make_evaluation_dict class method.

        Returns:
            pd.DataFrame: A pandas DataFrame where each row indexes a forecasting step and columns correspond to the various metrics stored in EvaluationMetrics.

        Example:
            >>> evaluation_df = EvaluationMetrics.evaluation_dict_to_dataframe(evaluation_dict)

        """
        df = pd.DataFrame.from_dict(evaluation_dict, orient='index')
        return df.loc[:, df.notna().any()]


# ============================================================ Metrics Manager ============================================================

class MetricsManager:
    """
    A class for calculating metrics on time series predictions
    Refer to https://github.com/prio-data/views_pipeline/blob/eval_docs/documentation/evaluation/schema.MD for more details on three evaluation schemas.
    """

    def __init__(self, metrics_list):
        """
        Initialize the manager with a list of metric names to calculate.

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
    
    @staticmethod
    def _calculate_rmsle(matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, depvar: str) -> float:
        return (
            root_mean_squared_error(matched_actual, matched_pred)
            if depvar.startswith("ln")
            else root_mean_squared_log_error(matched_actual, matched_pred)
            )

    @staticmethod
    def _calculate_crps(matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, depvar: str) -> float:
        return ps.crps_ensemble(matched_actual, matched_pred).mean()

    @staticmethod
    def _calculate_ap(matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, depvar: str, threshold=0.01) -> float:
        """
        Calculate Average Precision (AP) for binary predictions with a threshold.
        """
        matched_pred_binary = (matched_pred >= threshold).astype(int)
        matched_actual_binary = (matched_actual > 0).astype(int)
        return average_precision_score(matched_actual_binary, matched_pred_binary)

    @staticmethod
    def _calculate_brier(matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, depvar: str) -> float:
        pass

    @staticmethod
    def _calculate_jeffreys(matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, depvar: str) -> float:
        pass

    @staticmethod
    def _calculate_coverage(matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, depvar: str) -> float:
        pass

    @staticmethod
    def _calculate_emd(matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, depvar: str) -> float:
        pass

    @staticmethod
    def _calculate_sd(matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, depvar: str) -> float:
        pass

    @staticmethod
    def _calculate_pEMDiv(matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, depvar: str) -> float:
        pass

    @staticmethod
    def _calculate_pearson(matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, depvar: str) -> float:
        pass

    @staticmethod
    def _calculate_variogram(matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, depvar: str) -> float:
        pass

    @staticmethod
    def _match_actual_pred(actual: pd.DataFrame, pred: pd.DataFrame, depvar: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Matches the actual and predicted DataFrames based on the index and depvar column.

        Parameters:
        - actual: pd.DataFrame with a MultiIndex (e.g., month, level).
        - pred: pd.DataFrame with a MultiIndex that may contain duplicated indices.
        - depvar: str, the depvar column in actual.

        Returns:
        - matched_actual: pd.DataFrame aligned with pred.
        - matched_pred: pd.DataFrame aligned with actual.
        """
        actual_depvar = actual[[depvar]]
        aligned_actual, aligned_pred = actual_depvar.align(pred, join="inner")
        matched_actual = aligned_actual.reindex(index=aligned_pred.index)
        matched_actual[[depvar]] = actual_depvar

        return matched_actual.sort_index(), pred.sort_index()

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
                names=[time_id]
            )
            result_dfs.append(combined)
        
        return result_dfs

    def step_wise_evaluation(
            self, actual: pd.DataFrame, predictions: List[pd.DataFrame], depvar: str, steps: List[int]
            ) -> pd.DataFrame:
        """
        Evaluates the predictions step-wise and calculates the specified metrics.

        Args:
            actual (pd.DataFrame): The actual values.
            predictions (List[pd.DataFrame]): A list of DataFrames containing the predictions.
            depvar (str): The depvar column in the actual DataFrame.
            steps (List[int]): The steps to evaluate.

        Returns:
            pd.DataFrame: A DataFrame containing the evaluation metrics.
        """ 
        evaluation_dict = EvaluationMetrics.make_step_wise_evaluation_dict(steps=max(steps))
        step_metrics = {}

        result_dfs = MetricsManager._split_dfs_by_step(predictions)
        
        for metric in self.metrics_list:
            if metric in self.metric_functions:
                for i, pred in enumerate(result_dfs):
                    step = i + 1
                    matched_actual, matched_pred = MetricsManager._match_actual_pred(actual, pred, depvar)
                    evaluation_dict[f"step{str(step).zfill(2)}"].__setattr__(
                        metric, self.metric_functions[metric](matched_actual, matched_pred, depvar)
                    )
            else:
                logger.warning(f"Metric {metric} is not a default metric, skipping...")
            
        return evaluation_dict, EvaluationMetrics.evaluation_dict_to_dataframe(evaluation_dict)

    def time_series_wise_evaluation(
            self, actual: pd.DataFrame, predictions: List[pd.DataFrame], depvar: str
            ) -> pd.DataFrame:
        """
        Evaluates the predictions time series-wise and calculates the specified metrics.

        Args:
            actual (pd.DataFrame): The actual values.
            predictions (List[pd.DataFrame]): A list of DataFrames containing the predictions.
            depvar (str): The depvar column in the actual DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the evaluation metrics.
        """
        evaluation_dict = EvaluationMetrics.make_time_series_wise_evaluation_dict(len(predictions))
        
        for metric in self.metrics_list:
            if metric in self.metric_functions:
                for i, pred in enumerate(predictions):
                    matched_actual, matched_pred = MetricsManager._match_actual_pred(actual, pred, depvar)
                    evaluation_dict[f"ts{str(i).zfill(2)}"].__setattr__(
                        metric, self.metric_functions[metric](matched_actual, matched_pred, depvar)
                    )
            else:
                logger.warning(f"Metric {metric} is not a default metric, skipping...")

        return evaluation_dict, EvaluationMetrics.evaluation_dict_to_dataframe(evaluation_dict)   
    
    def month_wise_evaluation(
            self, actual: pd.DataFrame, predictions: List[pd.DataFrame], depvar: str
            ) -> pd.DataFrame:
        """
        Evaluates the predictions month-wise and calculates the specified metrics.

        Args:
            actual (pd.DataFrame): The actual values.
            predictions (List[pd.DataFrame]): A list of DataFrames containing the predictions.
            depvar (str): The depvar column in the actual DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the evaluation metrics.
        """
        pred_concat = pd.concat(predictions)
        month_range = pred_concat.index.get_level_values(0).unique()
        month_start = month_range.min()
        month_end = month_range.max()
        evaluation_dict = EvaluationMetrics.make_month_wise_evaluation_dict(month_start, month_end)

        matched_actual, matched_pred = MetricsManager._match_actual_pred(actual, pred_concat, depvar)
        # matched_concat = pd.merge(matched_actual, matched_pred, left_index=True, right_index=True)

        for metric in self.metrics_list:
            if metric in self.metric_functions:
                metric_by_month = matched_pred.groupby(level=matched_pred.index.names[0]).apply(
                    lambda df: self.metric_functions[metric](
                        matched_actual.loc[df.index, [depvar]], matched_pred.loc[df.index, [f"pred_{depvar}"]], depvar
                    )
                )

                for month in month_range:
                    evaluation_dict[f"month{str(month)}"].__setattr__(metric, metric_by_month.loc[month])
            else:
                logger.warning(f"Metric {metric} is not a default metric, skipping...")

        return evaluation_dict, EvaluationMetrics.evaluation_dict_to_dataframe(evaluation_dict)
    