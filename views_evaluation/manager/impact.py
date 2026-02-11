from views_evaluation.manager.base import EvaluationManager
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
        

    def evaluate(self, actual: pd.DataFrame, predictions: List[pd.DataFrame], target: str, config: dict, **kwargs):
        """
        Evaluates the predictions of an impact model.
        """
        pass