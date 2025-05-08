from typing import List, Dict, Tuple, Optional
import logging
import pandas as pd
import numpy as np
import properscoring as ps
from sklearn.metrics import (
    root_mean_squared_log_error,
    average_precision_score,
)
from scipy.stats import wasserstein_distance, pearsonr
from views_evaluation.evaluation.metrics import (
    PointEvaluationMetrics,
    UncertaintyEvaluationMetrics,
)
from views_pipeline_core.data.handlers import _ViewsDataset

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
        self.point_metric_functions = {
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
        self.uncertainty_metric_functions = {
            "CRPS": self._calculate_crps,
        }

    @staticmethod
    def _calculate_rmsle(
        matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str
    ) -> float:
        """
        Calculate Root Mean Squared Logarithmic Error (RMSLE) for each prediction.
        
        Args:
            matched_actual (pd.DataFrame): DataFrame containing actual values
            matched_pred (pd.DataFrame): DataFrame containing predictions
            target (str): The target column name

        Returns:
            float: Average RMSLE score
        """
        actual_values = np.concatenate(matched_actual[target].values)
        pred_values = np.concatenate(matched_pred[f'pred_{target}'].values)

        actual_expanded = np.repeat(actual_values, 
                              [len(x) for x in matched_pred[f'pred_{target}']])

        return root_mean_squared_log_error(actual_expanded, pred_values)
        

    @staticmethod
    def _calculate_crps(
        matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str
    ) -> float:
        """
        Calculate Continuous Ranked Probability Score (CRPS) for each prediction.
        
        Args:
            matched_actual (pd.DataFrame): DataFrame containing actual values
            matched_pred (pd.DataFrame): DataFrame containing predictions
            target (str): The target column name

        Returns:
            float: Average CRPS score
        """
        return np.mean(
            [
                ps.crps_ensemble(actual[0], np.array(pred))
                for actual, pred in zip(
                    matched_actual[target], matched_pred[f"pred_{target}"]
                )
            ]
        )

    @staticmethod
    def _calculate_ap(
        matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str, threshold=30
    ) -> float:
        """
        Calculate Average Precision (AP) for binary predictions with a threshold.
        
        Args:
            matched_actual (pd.DataFrame): DataFrame containing actual values
            matched_pred (pd.DataFrame): DataFrame containing predictions
            target (str): The target column name
            threshold (float): Threshold to convert predictions to binary values
            
        Returns:
            float: Average Precision score
        """
        actual_values = np.concatenate(matched_actual[target].values)
        pred_values = np.concatenate(matched_pred[f'pred_{target}'].values)

        actual_expanded = np.repeat(actual_values, 
                              [len(x) for x in matched_pred[f'pred_{target}']])

        actual_binary = (actual_expanded > threshold).astype(int)
        pred_binary = (pred_values >= threshold).astype(int)
        
        return average_precision_score(actual_binary, pred_binary)

    @staticmethod
    def _calculate_brier(
        matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str
    ) -> float:
        pass

    @staticmethod
    def _calculate_jeffreys(
        matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str
    ) -> float:
        pass

    @staticmethod
    def _calculate_coverage(
        matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str
    ) -> float:
        pass

    @staticmethod
    def _calculate_emd(
        matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str
    ) -> float:
        """
        Calculate Earth Mover's Distance (EMD) between predicted and actual distributions.
        EMD measures the minimum amount of work needed to transform one distribution into another.
        
        Args:
            matched_actual (pd.DataFrame): DataFrame containing actual values
            matched_pred (pd.DataFrame): DataFrame containing predictions
            target (str): The target column name

        Returns:
            float: Average EMD score
        """
        actual_values = np.concatenate(matched_actual[target].values)
        pred_values = np.concatenate(matched_pred[f'pred_{target}'].values)

        actual_expanded = np.repeat(actual_values, 
                              [len(x) for x in matched_pred[f'pred_{target}']])

        # Calculate EMD (1D Wasserstein distance)
        emd = wasserstein_distance(actual_expanded, pred_values)

        return emd

    @staticmethod
    def _calculate_sd(
        matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str
    ) -> float:
        pass

    @staticmethod
    def _calculate_pEMDiv(
        matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str
    ) -> float:
        pass

    @staticmethod
    def _calculate_pearson(
        matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str
    ) -> float:
        """
        Calculate Pearson correlation coefficient between actual and predicted values.
        This measures the linear correlation between predictions and actual values.
        
        Args:
            matched_actual (pd.DataFrame): DataFrame containing actual values
            matched_pred (pd.DataFrame): DataFrame containing predictions
            target (str): The target column name

        Returns:
            float: Pearson correlation coefficient
        """
        actual_values = np.concatenate(matched_actual[target].values)
        pred_values = np.concatenate(matched_pred[f'pred_{target}'].values)

        actual_expanded = np.repeat(actual_values, 
                              [len(x) for x in matched_pred[f'pred_{target}']])

        # Calculate Pearson correlation
        correlation, _ = pearsonr(actual_expanded, pred_values)
        return correlation

    @staticmethod
    def _calculate_variogram(
        matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str
    ) -> float:
        """
        Calculate the variogram score between actual and predicted values.
        This measures the spatial/temporal correlation structure.
        
        Args:
            matched_actual (pd.DataFrame): DataFrame containing actual values
            matched_pred (pd.DataFrame): DataFrame containing predictions
            target (str): The target column name

        Returns:
            float: Variogram score
        """
        actual_values = np.concatenate(matched_actual[target].values)
        pred_values = np.concatenate(matched_pred[f'pred_{target}'].values)

        actual_expanded = np.repeat(actual_values, 
                              [len(x) for x in matched_pred[f'pred_{target}']])

        # Calculate empirical variogram for actual values
        n = len(actual_expanded)
        actual_variogram = np.zeros(n-1)
        for i in range(n-1):
            actual_variogram[i] = np.mean((actual_expanded[i+1:] - actual_expanded[i])**2)

        # Calculate empirical variogram for predicted values
        pred_variogram = np.zeros(n-1)
        for i in range(n-1):
            pred_variogram[i] = np.mean((pred_values[i+1:] - pred_values[i])**2)

        # Calculate mean squared difference between variograms
        variogram_score = np.mean((actual_variogram - pred_variogram)**2)

        return variogram_score

    @staticmethod
    def transform_data(df: pd.DataFrame, target: str) -> pd.DataFrame:
        """
        Transform the data to normal distribution.
        """
        if target.startswith("ln"):
            df[target] = df[target].applymap(lambda x: np.exp(x) - 1 if isinstance(x, (list, np.ndarray)) else np.exp(x) - 1)
        elif target.startswith("lx"):
            df[target] = df[target].applymap(lambda x: np.exp(x) - np.exp(100) if isinstance(x, (list, np.ndarray)) else np.exp(x) - np.exp(100))
        elif target.startswith("lr"):
            df[target] = df[target].applymap(lambda x: x if isinstance(x, (list, np.ndarray)) else x)
        else:
            raise ValueError(f"Target {target} is not a valid target")
        return df
    
    @staticmethod
    def get_evaluation_type(predictions: List[pd.DataFrame]) -> bool:
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
            for value in df.values.flatten():
                if not isinstance(value, list):
                    raise ValueError("All values must be lists. Use _ViewsDataset to convert the data.")
                
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
    def validate_predictions(
        predictions: List[pd.DataFrame], target: str, is_uncertainty: bool
    ):
        """
        Checks if the predictions are valid DataFrames.
        - Each DataFrame must have exactly one column named `pred_column_name`.
        - If is_uncertainty is True, all elements in the column must be lists.
        - If is_uncertainty is False, all elements in the column must be floats.

        Args:
            predictions (List[pd.DataFrame]): A list of DataFrames containing the predictions.
            target (str): The target column in the actual DataFrame.
            is_uncertainty (bool): Flag to indicate if the evaluation is for uncertainty.
        """
        pred_column_name = f"pred_{target}"
        if not isinstance(predictions, list):
            raise TypeError("Predictions must be a list of DataFrames.")

        for i, df in enumerate(predictions):
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"Predictions[{i}] must be a DataFrame.")
            if df.empty:
                raise ValueError(f"Predictions[{i}] must not be empty.")
            if df.columns.tolist() != [pred_column_name]:
                raise ValueError(
                    f"Predictions[{i}] must contain only one column named '{pred_column_name}'."
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
        aligned_actual, aligned_pred = actual_target.align(pred, join="inner")
        matched_actual = aligned_actual.reindex(index=aligned_pred.index)
        matched_actual[[target]] = actual_target

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

        step_metrics = {}
        result_dfs = EvaluationManager._split_dfs_by_step(predictions)

        for metric in self.metrics_list:
            if metric in metric_functions:
                for i, pred in enumerate(result_dfs):
                    step = i + 1
                    matched_actual, matched_pred = EvaluationManager._match_actual_pred(
                        actual, pred, target
                    )
                    evaluation_dict[f"step{str(step).zfill(2)}"].__setattr__(
                        metric,
                        metric_functions[metric](matched_actual, matched_pred, target),
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

        for metric in self.metrics_list:
            if metric in metric_functions:
                for i, pred in enumerate(predictions):
                    matched_actual, matched_pred = EvaluationManager._match_actual_pred(
                        actual, pred, target
                    )
                    evaluation_dict[f"ts{str(i).zfill(2)}"].__setattr__(
                        metric,
                        metric_functions[metric](matched_actual, matched_pred, target),
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
        month_start = month_range.min()
        month_end = month_range.max()
        
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

        for metric in self.metrics_list:
            if metric in metric_functions:
                metric_by_month = matched_pred.groupby(
                    level=matched_pred.index.names[0]
                ).apply(
                    lambda df: metric_functions[metric](
                        matched_actual.loc[df.index, [target]],
                        matched_pred.loc[df.index, [f"pred_{target}"]],
                        target,
                    )
                )

                for month in month_range:
                    evaluation_dict[f"month{str(month)}"].__setattr__(
                        metric, metric_by_month.loc[month]
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
        steps: List[int],
    ):
        """
        Evaluates the predictions and calculates the specified point metrics.

        Args:
            actual (pd.DataFrame): The actual values.
            predictions (List[pd.DataFrame]): A list of DataFrames containing the predictions.
            target (str): The target column in the actual DataFrame.
            steps (List[int]): The steps to evaluate.

        """
        is_uncertainty = EvaluationManager.get_evaluation_type(predictions)
        EvaluationManager.validate_predictions(predictions, target, is_uncertainty)
        actual = EvaluationManager.transform_data(_ViewsDataset(actual).dataframe, target)
        predictions = [EvaluationManager.transform_data(_ViewsDataset(pred).dataframe, target) for pred in predictions]

        evaluation_results = {}
        evaluation_results["month"] = self.month_wise_evaluation(
            actual, predictions, target, is_uncertainty
        )
        evaluation_results["time_series"] = self.time_series_wise_evaluation(
            actual, predictions, target, is_uncertainty
        )
        evaluation_results["step"] = self.step_wise_evaluation(
            actual, predictions, target, steps, is_uncertainty,
        )

        return evaluation_results
