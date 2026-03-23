from views_evaluation.manager.base import EvaluationManager
from views_evaluation.core import PointEvaluationMetrics, UncertaintyEvaluationMetrics
from typing import List
import pandas as pd


class ImpactEvaluationManager(EvaluationManager):
    """
    A class for evaluating the predictions of an impact model.
    """
    def __init__(self, metrics_list: list):
        super().__init__(metrics_list)
    
    @staticmethod
    def _verify_month_id_same_year(start_month: int, end_month: int) -> bool:
        """
        Verify if the start and end month are January and December in the same year.
        1 = Jan 1980
        """

        # Check if start is Jan
        if (start_month - 1) % 12 != 0:
            return False

        # Check if end is Dec of the same year
        if end_month != start_month + 11:
            return False

        return True
    
    @staticmethod
    def _month_id_to_year(month_id: int) -> int:
        """
        Converts month_id (1 = Jan 1980) to calendar year.
        """
        return 1980 + (month_id - 1) // 12
        


    def evaluate(
        self,
        actual: pd.DataFrame,
        predictions: List[pd.DataFrame],
        target: str,
        config: dict,
        **kwargs,
    ):
        """
        Evaluates impact model strictly on yearly (calendar year) level.
        """

        EvaluationManager.validate_predictions(predictions, target)
        actual, predictions = self._process_data(actual, predictions, target)

        is_uncertainty = EvaluationManager.get_evaluation_type(
            predictions, f"pred_{target}"
        )

        pred_concat = pd.concat(predictions)

        matched_actual, matched_pred = EvaluationManager._match_actual_pred(
            actual, pred_concat, target
        )


        month_index = matched_pred.index.get_level_values(0)
        unique_months = sorted(month_index.unique())

        yearly_groups = []
        i = 0

        while i < len(unique_months):
            start_month = unique_months[i]

            if i + 11 >= len(unique_months):
                break  # not enough months left

            end_month = unique_months[i + 11]

            if self._verify_month_id_same_year(start_month, end_month):
                yearly_groups.append(unique_months[i:i+12])
                i += 12
            else:
                i += 1  # move forward until January found


        yearly_actual_list = []
        yearly_pred_list = []

        for months in yearly_groups:

            mask = month_index.isin(months)

            actual_slice = matched_actual.loc[mask]
            pred_slice = matched_pred.loc[mask]

            # IMPORTANT:
            # Since yearly target was uniformly disaggregated, yearly value should equal the SUM of 12 months
            yearly_actual = actual_slice.groupby(
                actual_slice.index.get_level_values(1)
            ).mean()

            yearly_pred = pred_slice.groupby(
                pred_slice.index.get_level_values(1)
            ).mean()
            

            yearly_actual_list.append(yearly_actual)
            yearly_pred_list.append(yearly_pred)

        if len(yearly_actual_list) == 0:
            raise ValueError("No complete calendar years available for evaluation.")

        yearly_actual_df = pd.concat(yearly_actual_list)
        yearly_pred_df = pd.concat(yearly_pred_list)
        
       
        if is_uncertainty:
            evaluation_obj = UncertaintyEvaluationMetrics()
            metric_functions = self.uncertainty_metric_functions
        else:
            evaluation_obj = PointEvaluationMetrics()
            metric_functions = self.point_metric_functions

        for metric in self.metrics_list:
            if metric in metric_functions:
                value = metric_functions[metric](
                    yearly_actual_df,
                    yearly_pred_df,
                    target,
                    **kwargs,
                )
                setattr(evaluation_obj, metric, value)

       
        evaluation_dict = {"year": evaluation_obj}

        evaluation_df = evaluation_obj.__class__.evaluation_dict_to_dataframe(
            evaluation_dict
        )

        return {
            "year": (evaluation_dict, evaluation_df)
        }