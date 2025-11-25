import pandas as pd
import numpy as np
from typing import List, Tuple

class DataUtils:
    """
    A collection of static methods for data preparation, transformation,
    and alignment necessary for the evaluation pipeline.
    """
    @staticmethod
    def transform_data(df: pd.DataFrame, target: str | list[str]) -> pd.DataFrame:
        """
        Applies the inverse transformation (e.g., np.exp) to restore the data to its original scale based on column prefixes (ln, lx, lr).

        Args:
            df (pd.DataFrame): The input DataFrame with columns that may contain lists.
            target (str | list[str]): The target column in the actual DataFrame.

        Returns:
            pd.DataFrame: A new DataFrame with columns transformed to their original scale.

        Raises:
            ValueError: If the target column is not a valid target.
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
        Ensures target columns are stored as numpy arrays for consistent
        metric calculation, especially for uncertainty metrics.

        Args:
            df (pd.DataFrame): The input DataFrame with columns that may contain lists.
            target (str | list[str]): The target column in the actual DataFrame.

        Returns:
            pd.DataFrame: A new DataFrame with columns converted to numpy arrays.

        Raises:
            ValueError: If the target column is not a valid target.
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
        Converts array-like columns to single scalar values (mean) for
        pure point metric evaluation.

        Args:
            df (pd.DataFrame): The input DataFrame with columns that may contain lists.
            target (str | list[str]): The target column in the actual DataFrame.

        Returns:
            pd.DataFrame: A new DataFrame with columns converted to scalar values.

        Raises:
            ValueError: If the target column is not a valid target.
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
    def match_actual_pred(
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
    def split_dfs_by_step(dfs: list) -> list:
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