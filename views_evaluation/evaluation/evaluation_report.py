from typing import Dict, Any
from views_evaluation.evaluation.metrics import (
    RegressionPointEvaluationMetrics,
    RegressionSampleEvaluationMetrics,
    ClassificationPointEvaluationMetrics,
    ClassificationSampleEvaluationMetrics,
)

class EvaluationReport:
    """
    A structured, framework-agnostic container for evaluation results.
    
    This class decouples the raw result data from its final presentation
    format, allowing for flexible export to JSON, Dictionaries, or Pandas.
    """
    def __init__(self, target: str, task: str, pred_type: str, results: Dict[str, Dict[str, Any]]):
        self.target = target
        self.task = task
        self.pred_type = pred_type
        # Internal structure: {schema_name: {group_id: {metric_name: value}}}
        self._results = results
        
        # Map task/type to legacy dataclasses for formatting
        self._metrics_map = {
            ("regression", "point"): RegressionPointEvaluationMetrics,
            ("regression", "sample"): RegressionSampleEvaluationMetrics,
            ("classification", "point"): ClassificationPointEvaluationMetrics,
            ("classification", "sample"): ClassificationSampleEvaluationMetrics,
        }

    def _get_metrics_cls(self):
        return self._metrics_map[(self.task, self.pred_type)]

    def get_schema_results(self, schema: str) -> Dict[str, Any]:
        """
        Returns the result dictionary for a specific schema, 
        mapped to legacy dataclass instances for backward compatibility.
        """
        if schema not in self._results:
            raise KeyError(f"Schema '{schema}' not found in report.")
        
        raw_results = self._results[schema]
        metrics_cls = self._get_metrics_cls()
        
        mapped_results = {}
        for group_id, metrics in raw_results.items():
            container = metrics_cls()
            for k, v in metrics.items():
                setattr(container, k, v)
            mapped_results[group_id] = container
            
        return mapped_results

    def to_dataframe(self, schema: str):
        """
        Converts a specific schema's results into a Pandas DataFrame.
        """
        import pandas as pd
        mapped_results = self.get_schema_results(schema)
        if not mapped_results:
            return pd.DataFrame()
        
        metrics_cls = self._get_metrics_cls()
        return metrics_cls.evaluation_dict_to_dataframe(mapped_results)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the entire report into a nested dictionary."""
        return {
            "target": self.target,
            "task": self.task,
            "pred_type": self.pred_type,
            "schemas": self._results
        }

    def __repr__(self):
        schemas = list(self._results.keys())
        return f"EvaluationReport(target='{self.target}', task='{self.task}', schemas={schemas})"
