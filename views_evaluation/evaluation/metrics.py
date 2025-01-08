from dataclasses import dataclass
from typing import Optional
import pandas as pd
from statistics import mean, stdev, median

import properscoring as ps
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, brier_score_loss, average_precision_score, roc_auc_score

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
            >>> evaluation_dict['step01'].MSE = sklearn.metrics.mean_squared_error(ts01_y_true, ts01_y_pred)
            >>> evaluation_dict['step02'].MSE = sklearn.metrics.mean_squared_error(ts02_y_true, ts02_y_pred)
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