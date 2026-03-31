"""
PHASE-3-DELETE
This module is TEMPORARY and will be deleted in Phase 3 of the orchestrator migration.
See: reports/2026-02-25_evaluation_frame_refactor/10_orchestrator_migration_plan.md

After Phase 3:
  - Adapters live in views-pipeline-core (or in the calling repository)
  - EvaluationManager is fully replaced by NativeEvaluator in pipeline-core
  - This file will not exist in this repository

Do not add new functionality to this file.
"""
from typing import List, Tuple
import logging
import warnings
import pandas as pd
import numpy as np
from views_evaluation.adapters.pandas import PandasAdapter
from views_evaluation.evaluation.native_evaluator import NativeEvaluator
from views_evaluation.evaluation.evaluation_frame import EvaluationFrame
from views_evaluation.evaluation.native_metric_calculators import (
    REGRESSION_POINT_NATIVE,
    REGRESSION_SAMPLE_NATIVE,
    CLASSIFICATION_POINT_NATIVE,
    CLASSIFICATION_SAMPLE_NATIVE,
)

logger = logging.getLogger(__name__)


class EvaluationManager:
    """
    A class for calculating metrics on time series predictions
    Refer to https://github.com/prio-data/views_pipeline/blob/eval_docs/documentation/evaluation/schema.MD for more details on three evaluation schemas.
    """

    def __init__(self):
        """
        Initialize the EvaluationManager.

        Metrics to compute and targets to evaluate are declared in the config
        passed to evaluate(). No metric list is accepted here.
        """

        warnings.warn(
            "EvaluationManager is deprecated and will be removed in Phase 3 of the "
            "orchestrator migration. Use NativeEvaluator directly with an adapter. "
            "See documentation/integration_guide.md.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.regression_point_functions           = REGRESSION_POINT_NATIVE
        self.regression_sample_functions          = REGRESSION_SAMPLE_NATIVE
        self.classification_point_functions       = CLASSIFICATION_POINT_NATIVE
        self.classification_sample_functions      = CLASSIFICATION_SAMPLE_NATIVE


    @staticmethod
    def transform_data(df: pd.DataFrame, target: str | list[str]) -> pd.DataFrame:
        """
        DEPRECATED. Apply legacy inverse transformations based on target name prefix.

        This method will be removed once all model repos have migrated to returning
        predictions on the original scale. Do not add new logic here.
        """
        if isinstance(target, str):
            target = [target]
        for t in target:
            if t.startswith("ln") or t.startswith("pred_ln"):
                df[[t]] = df[[t]].applymap(lambda x: np.exp(x) - 1)
            elif t.startswith("lx") or t.startswith("pred_lx"):
                df[[t]] = df[[t]].applymap(lambda x: np.exp(x) - np.exp(100))
            elif t.startswith("lr") or t.startswith("pred_lr"):
                pass  # identity — lr_ targets are already on the original scale
            else:
                logger.warning(
                    f"transform_data: unrecognised prefix for target '{t}'. "
                    "Applying identity (no transformation). "
                    "If this target requires inverse transformation it must be applied "
                    "by the model manager before calling evaluate(). "
                    "This fallback will be removed when transform_data is deprecated."
                )
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
        The return value indicates whether all DataFrames are for sample evaluation.

        Args:
            predictions (List[pd.DataFrame]): A list of DataFrames to check.

        Returns:
            bool: True if all DataFrames are for sample evaluation,
                  False if all DataFrame are for point evaluation.

        Raises:
            ValueError: If there is a mix of single and multiple values in the lists,
                      or if uncertainty lists have different lengths.
        """
        is_sample = False
        is_point = False
        sample_length = None

        for df in predictions:
            for value in df[target].values.flatten():
                if not (isinstance(value, np.ndarray) or isinstance(value, list)):
                    raise ValueError(
                        "All values must be lists or numpy arrays. Convert the data."
                    )

                if len(value) > 1:
                    is_sample = True
                    # For sample evaluation, check that all lists have the same length
                    if sample_length is None:
                        sample_length = len(value)
                    elif len(value) != sample_length:
                        raise ValueError(
                            f"Inconsistent list lengths in sample evaluation. "
                            f"Found lengths {sample_length} and {len(value)}"
                        )
                elif len(value) == 1:
                    is_point = True
                else:
                    raise ValueError("Empty lists are not allowed")

        if is_sample and is_point:
            raise ValueError(
                "Mix of evaluation types detected: some rows contain single values, others contain multiple values. "
                "Please ensure all rows are consistent in their evaluation type"
            )

        return is_sample

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
            
            if len(df.columns) != 1:
                raise ValueError(
                    f"Predictions[{i}] must contain exactly one column, but found {len(df.columns)}: {list(df.columns)}" # <--------
                )

            if pred_column_name not in df.columns:
                raise ValueError(
                    f"Predictions[{i}] must contain the column named '{pred_column_name}'. Columns found: {list(df.columns)}"
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
        for group in grouped_month_ids:
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

    @staticmethod
    def _normalise_config(config: dict) -> dict:
        """
        Translate legacy config keys to canonical keys, warning loudly.

        Legacy key 'targets' → 'regression_targets'
        Legacy key 'metrics' → 'regression_point_metrics'
        Legacy key 'regression_uncertainty_metrics' → 'regression_sample_metrics'
        Legacy key 'classification_uncertainty_metrics' → 'classification_sample_metrics'
        """
        canonical = config.copy()
        if "targets" in config and "regression_targets" not in config:
            logger.warning(
                "Config key 'targets' is DEPRECATED and will be rejected in a future "
                "version. It has been treated as 'regression_targets'. "
                "Update your config."
            )
            canonical["regression_targets"] = canonical.pop("targets")
        if "metrics" in config and "regression_point_metrics" not in config:
            logger.warning(
                "Config key 'metrics' is DEPRECATED and will be rejected in a future "
                "version. It has been treated as 'regression_point_metrics'. "
                "Update your config."
            )
            canonical["regression_point_metrics"] = canonical.pop("metrics")

        if "regression_uncertainty_metrics" in config and "regression_sample_metrics" not in config:
            logger.warning(
                "Config key 'regression_uncertainty_metrics' is DEPRECATED and will be rejected in a future "
                "version. It has been treated as 'regression_sample_metrics'. "
                "Update your config."
            )
            canonical["regression_sample_metrics"] = canonical.pop("regression_uncertainty_metrics")

        if "classification_uncertainty_metrics" in config and "classification_sample_metrics" not in config:
            logger.warning(
                "Config key 'classification_uncertainty_metrics' is DEPRECATED and will be rejected in a future "
                "version. It has been treated as 'classification_sample_metrics'. "
                "Update your config."
            )
            canonical["classification_sample_metrics"] = canonical.pop("classification_uncertainty_metrics")

        return canonical

    @staticmethod
    def _validate_config(config: dict) -> None:
        """
        Fail loud and fast on an invalid or incomplete config.

        Raises KeyError if required keys are absent.
        """
        if "steps" not in config:
            raise KeyError("Config must contain 'steps'.")
        has_regression     = bool(config.get("regression_targets"))
        has_classification = bool(config.get("classification_targets"))
        if not has_regression and not has_classification:
            raise KeyError(
                "Config must declare at least one of 'regression_targets' or "
                "'classification_targets'."
            )
        if has_regression and not (
            config.get("regression_point_metrics") or config.get("regression_sample_metrics")
        ):
            raise KeyError(
                "Config declares 'regression_targets' but has neither "
                "'regression_point_metrics' nor 'regression_sample_metrics'."
            )
        if has_classification and not (
            config.get("classification_point_metrics") or config.get("classification_sample_metrics")
        ):
            raise KeyError(
                "Config declares 'classification_targets' but has neither "
                "'classification_point_metrics' nor 'classification_sample_metrics'."
            )

        # Validate that metrics are valid for the task type (ADR-014)
        from views_evaluation.evaluation.native_metric_calculators import (
            REGRESSION_POINT_NATIVE, REGRESSION_SAMPLE_NATIVE,
            CLASSIFICATION_POINT_NATIVE, CLASSIFICATION_SAMPLE_NATIVE
        )
        
        for metric in config.get("regression_point_metrics", []):
            if metric not in REGRESSION_POINT_NATIVE or metric == "AP":
                raise ValueError(f"Metric '{metric}' is not valid for regression point tasks.")

        for metric in config.get("regression_sample_metrics", []):
            if metric not in REGRESSION_SAMPLE_NATIVE:
                raise ValueError(f"Metric '{metric}' is not valid for regression sample tasks.")
        for metric in config.get("classification_point_metrics", []):
            if metric not in CLASSIFICATION_POINT_NATIVE:
                raise ValueError(f"Metric '{metric}' is not valid for classification point tasks.")
        for metric in config.get("classification_sample_metrics", []):
            if metric not in CLASSIFICATION_SAMPLE_NATIVE:
                raise ValueError(f"Metric '{metric}' is not valid for classification sample tasks.")


    def step_wise_evaluation(
        self,
        actual: pd.DataFrame,
        predictions: List[pd.DataFrame],
        target: str,
        steps: List[int],
        metrics_list: List[str],
        metric_functions: dict,
        metrics_cls: type,
        **kwargs,
    ):
        """
        Evaluates the predictions step-wise and calculates the specified metrics.

        Args:
            actual (pd.DataFrame): The actual values.
            predictions (List[pd.DataFrame]): A list of DataFrames containing the predictions.
            target (str): The target column in the actual DataFrame.
            steps (List[int]): The steps to evaluate.
            metrics_list (List[str]): Metrics to compute, declared in config.
            metric_functions (dict): Dispatch dict for the resolved task/pred type.
            metrics_cls (type): Dataclass to use for result storage.

        Returns:
            Tuple: A tuple containing the evaluation dictionary and the evaluation DataFrame.
        """
        evaluation_dict = metrics_cls.make_step_wise_evaluation_dict(steps=max(steps))
        result_dfs = EvaluationManager._split_dfs_by_step(predictions)

        step_matched_data = {}
        for i, pred in enumerate(result_dfs):
            step = i + 1
            matched_actual, matched_pred = EvaluationManager._match_actual_pred(
                actual, pred, target
            )
            step_matched_data[step] = (matched_actual, matched_pred)

        for metric in metrics_list:
            for step, (matched_actual, matched_pred) in step_matched_data.items():
                evaluation_dict[f"step{str(step).zfill(2)}"].__setattr__(
                    metric,
                    metric_functions[metric](
                        matched_actual, matched_pred, target, **kwargs
                    ),
                )

        return (
            evaluation_dict,
            metrics_cls.evaluation_dict_to_dataframe(evaluation_dict),
        )

    def time_series_wise_evaluation(
        self,
        actual: pd.DataFrame,
        predictions: List[pd.DataFrame],
        target: str,
        metrics_list: List[str],
        metric_functions: dict,
        metrics_cls: type,
        **kwargs,
    ):
        """
        Evaluates the predictions time series-wise and calculates the specified metrics.

        Args:
            actual (pd.DataFrame): The actual values.
            predictions (List[pd.DataFrame]): A list of DataFrames containing the predictions.
            target (str): The target column in the actual DataFrame.
            metrics_list (List[str]): Metrics to compute, declared in config.
            metric_functions (dict): Dispatch dict for the resolved task/pred type.
            metrics_cls (type): Dataclass to use for result storage.

        Returns:
            Tuple: A tuple containing the evaluation dictionary and the evaluation DataFrame.
        """
        evaluation_dict = metrics_cls.make_time_series_wise_evaluation_dict(
            len(predictions)
        )

        ts_matched_data = {}
        for i, pred in enumerate(predictions):
            matched_actual, matched_pred = EvaluationManager._match_actual_pred(
                actual, pred, target
            )
            ts_matched_data[i] = (matched_actual, matched_pred)

        for metric in metrics_list:
            for i, (matched_actual, matched_pred) in ts_matched_data.items():
                evaluation_dict[f"ts{str(i).zfill(2)}"].__setattr__(
                    metric,
                    metric_functions[metric](
                        matched_actual, matched_pred, target, **kwargs
                    ),
                )

        return (
            evaluation_dict,
            metrics_cls.evaluation_dict_to_dataframe(evaluation_dict),
        )

    def month_wise_evaluation(
        self,
        actual: pd.DataFrame,
        predictions: List[pd.DataFrame],
        target: str,
        metrics_list: List[str],
        metric_functions: dict,
        metrics_cls: type,
        **kwargs,
    ):
        """
        Evaluates the predictions month-wise and calculates the specified metrics.

        Args:
            actual (pd.DataFrame): The actual values.
            predictions (List[pd.DataFrame]): A list of DataFrames containing the predictions.
            target (str): The target column in the actual DataFrame.
            metrics_list (List[str]): Metrics to compute, declared in config.
            metric_functions (dict): Dispatch dict for the resolved task/pred type.
            metrics_cls (type): Dataclass to use for result storage.

        Returns:
            Tuple: A tuple containing the evaluation dictionary and the evaluation DataFrame.
        """
        pred_concat = pd.concat(predictions)
        month_range = pred_concat.index.get_level_values(0).unique()
        month_start = int(month_range.min())
        month_end   = int(month_range.max())

        evaluation_dict = metrics_cls.make_month_wise_evaluation_dict(
            month_start, month_end
        )

        matched_actual, matched_pred = EvaluationManager._match_actual_pred(
            actual, pred_concat, target
        )

        g = matched_pred.groupby(level=matched_pred.index.names[0], sort=False, observed=True)
        groups = g.indices  # dict: {month -> np.ndarray of row positions}

        for metric in metrics_list:
            for month, pos in groups.items():
                value = metric_functions[metric](
                    matched_actual.iloc[pos],
                    matched_pred.iloc[pos],
                    target,
                    **kwargs,
                )
                evaluation_dict[f"month{str(month)}"].__setattr__(metric, value)

        return (
            evaluation_dict,
            metrics_cls.evaluation_dict_to_dataframe(evaluation_dict),
        )

    def evaluate(
        self,
        actual: pd.DataFrame = None,
        predictions: List[pd.DataFrame] = None,
        target: str = None,
        config: dict = None,
        ef: EvaluationFrame = None,
        verify_parity: bool = False,
        **kwargs,
    ):
        """
        Evaluate predictions. Supports legacy DataFrame inputs OR Native EvaluationFrame.

        Args:
            actual (pd.DataFrame): Optional. Legacy actuals.
            predictions (List[pd.DataFrame]): Optional. Legacy predictions.
            target (str): Target column name.
            config (dict): Evaluation configuration.
            ef (EvaluationFrame): Optional. Pre-adapted native frame.
            verify_parity (bool): If True and both ef and legacy inputs are provided,
                verifies bit-wise parity between them.
        """
        config = EvaluationManager._normalise_config(config)
        EvaluationManager._validate_config(config)

        if ef is not None:
            # PATH B: Direct Native Evaluation
            if not isinstance(ef, EvaluationFrame):
                raise TypeError("Provided 'ef' must be an EvaluationFrame instance.")
            target = ef.metadata.get('target', target)
            
            if verify_parity and actual is not None and predictions is not None:
                # ADR-024 Shadow Run: Verify external adaptation matches internal
                ef_internal = PandasAdapter.from_dataframes(actual, predictions, target)
                # Check data parity
                if not np.array_equal(ef.y_true, ef_internal.y_true):
                    raise ValueError("Parity Failure: y_true mismatch between external and internal adaptation.")
                if not np.array_equal(ef.y_pred, ef_internal.y_pred):
                    raise ValueError("Parity Failure: y_pred mismatch between external and internal adaptation.")
                for key in ef.identifiers:
                    if not np.array_equal(ef.identifiers[key], ef_internal.identifiers[key]):
                        raise ValueError(f"Parity Failure: identifier '{key}' mismatch.")
        else:
            # PATH A: Legacy Adaptation
            if actual is None or predictions is None or target is None:
                raise ValueError("If 'ef' is not provided, 'actual', 'predictions', and 'target' are required.")
            
            EvaluationManager.validate_predictions(predictions, target)
            # ADR-010: Adapt legacy DataFrames to canonical EvaluationFrame
            ef = PandasAdapter.from_dataframes(actual, predictions, target)

            # PHASE-3-DELETE: internal state for backward-compat reflective tests
            self.actual, self.predictions = self._process_data(actual, predictions, target)

        # ADR-010: Delegate to the NativeEvaluator (Pure Math Engine)
        evaluator = NativeEvaluator(config)
        
        # Phase 1: Enable legacy compatibility to maintain bit-wise parity
        # (Reproduces truncation bugs and positional step assumptions)
        try:
            report = evaluator.evaluate(ef, legacy_compatibility=True)
            # Map report back to legacy dictionary structure for backward compatibility
            return {
                schema: (report.get_schema_results(schema), report.to_dataframe(schema))
                for schema in ["month", "time_series", "step"]
            }
        except ValueError as e:
            # Re-wrap error message to match legacy test expectations if needed
            if "Target" in str(e) and "not found in config" in str(e):
                raise ValueError(f"Target '{target}' is not declared in config")
            raise e
