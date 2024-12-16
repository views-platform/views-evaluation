from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from metrics import EvaluationMetrics
from sklearn.metrics import root_mean_squared_error, root_mean_squared_log_error, average_precision_score
import properscoring as ps


def calculate_rmsle(matched_actual: pd.DataFrame, matched_pred: pd.DataFrame, target: str) -> float:
    return (
        root_mean_squared_error(matched_actual, matched_pred)
        if target.startswith("ln")
        else root_mean_squared_log_error(matched_actual, matched_pred)
        )


def calculate_crps(matched_actual: pd.DataFrame, matched_pred: pd.DataFrame) -> float:
    return ps.crps_ensemble(matched_actual, matched_pred).mean()



def match_actual_pred(
        actual: pd.DataFrame, 
        pred: pd.DataFrame, 
        target: str
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
    if target not in actual.columns:
        raise ValueError(f"Target column '{target}' not found in actual DataFrame.")
    
    actual_target = actual[[target]]
    
    aligned_actual, aligned_pred = actual_target.align(pred, join="inner")

    matched_actual = aligned_actual.reindex(index=aligned_pred.index)

    matched_actual[[target]] = actual_target

    return matched_actual.sort_index(), pred.sort_index()


def time_series_wise_evaluation(
        actual: pd.DataFrame, 
        predictions: List[pd.DataFrame], 
        target: str
        ) -> Dict[str, EvaluationMetrics]:

    evaluation_dict = EvaluationMetrics.make_time_series_wise_evaluation_dict(len(predictions))

    for i, pred in enumerate(predictions):
        matched_actual, matched_pred = match_actual_pred(actual, pred, target)
        evaluation_dict[f"ts{str(i).zfill(2)}"].RMSLE = calculate_rmsle(matched_actual, matched_pred, target)
        evaluation_dict[f"ts{str(i).zfill(2)}"].CRPS = calculate_crps(matched_actual, matched_pred)
        
    return EvaluationMetrics.evaluation_dict_to_dataframe(evaluation_dict)


def step_wise_evaluation(
        actual: pd.DataFrame, 
        predictions: List[pd.DataFrame], 
        target: str, 
        steps: list[int]
        ) -> Dict[str, EvaluationMetrics]:

    evaluation_dict = EvaluationMetrics.make_step_wise_evaluation_dict(steps=max(steps))
    step_metrics = {}

    for pred in predictions:
        for i, month in enumerate(pred.index.levels[0]):
            step = i + 1
            step_pred = pred.loc[pred.index.get_level_values(0) == month]
            matched_actual, matched_pred = match_actual_pred(actual, step_pred, target)
            rmsle = calculate_rmsle(matched_actual, matched_pred, target)
            crps = calculate_crps(matched_actual, matched_pred)

            if step not in step_metrics:
                step_metrics[step] = {"RMSLE": [], "CRPS": []}
            step_metrics[step]["RMSLE"].append(rmsle)
            step_metrics[step]["CRPS"].append(crps)

    for step in steps:
        evaluation_dict[f"step{str(step).zfill(2)}"].RMSLE = np.mean(step_metrics[step]["RMSLE"])
        # evaluation_dict[f"step{str(step).zfill(2)}"].CRPS = np.mean(step_metrics[step]["CRPS"])

    return EvaluationMetrics.evaluation_dict_to_dataframe(evaluation_dict)


def month_wise_evaluation(
        actual: pd.DataFrame,
        predictions: List[pd.DataFrame],
        target: str,
        ) -> Dict[str, EvaluationMetrics]:
    
    pred_concat = pd.concat(predictions)
    pred_concat_target = pred_concat.columns[0]
    month_range = pred_concat.index.get_level_values(0).unique()
    month_start = month_range.min()
    month_end = month_range.max()
    evaluation_dict = EvaluationMetrics.make_month_wise_evaluation_dict(month_start, month_end)

    matched_actual, matched_pred = match_actual_pred(actual, pred_concat, target)
    matched_concat = pd.merge(matched_actual, matched_pred, left_index=True, right_index=True)

    rmsle_by_month = matched_concat.groupby(level=matched_concat.index.names[0]).apply(
        lambda df: calculate_rmsle(
            df[[target]], df[[pred_concat_target]], target
        )
    )

    crps_by_month = matched_concat.groupby(level=matched_concat.index.names[0]).apply(
        lambda df: calculate_crps(
            df[[target]], df[[pred_concat_target]]
        )
    )

    for month in month_range:
        evaluation_dict[f"month{str(month)}"].RMSLE = rmsle_by_month.loc[month]
        evaluation_dict[f"month{str(month)}"].CRPS = crps_by_month.loc[month]

    return EvaluationMetrics.evaluation_dict_to_dataframe(evaluation_dict)

