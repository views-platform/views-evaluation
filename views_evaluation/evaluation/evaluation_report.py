from typing import Dict, Any, Tuple, Optional

class EvaluationReport:
    """
    A structured, framework-agnostic container for evaluation results.
    
    This class decouples the raw result data from its final presentation
    format, allowing for flexible export to JSON, Dictionaries, or Pandas.
    """
    def __init__(self, target: str, results: Dict[str, Dict[str, Any]]):
        self.target = target
        # Internal structure: {schema_name: {group_id: metrics_dataclass_instance}}
        self._results = results

    def get_schema_results(self, schema: str) -> Dict[str, Any]:
        """Returns the result dictionary for a specific schema."""
        if schema not in self._results:
            raise KeyError(f"Schema '{schema}' not found in report.")
        return self._results[schema]

    def to_dataframe(self, schema: str):
        """
        Converts a specific schema's results into a Pandas DataFrame.
        Uses the legacy dataclass's static method for formatting.
        """
        import pandas as pd
        results_dict = self.get_schema_results(schema)
        if not results_dict:
            return pd.DataFrame()
        
        # Determine the dataclass from any instance in the dict
        # (Assuming all instances in a schema are the same type)
        first_val = next(iter(results_dict.values()))
        return first_val.__class__.evaluation_dict_to_dataframe(results_dict)



    def to_dict(self) -> Dict[str, Any]:
        """Converts the entire report into a nested dictionary."""
        # Note: We only export the metric values, not the DataFrames
        return {
            "target": self.target,
            "schemas": {
                name: {
                    group_id: {
                        metric: val 
                        for metric, val in vars(container).items() 
                        if val is not None
                    }
                    for group_id, container in schema_results.items()
                }
                for name, schema_results in self._results.items()
            }
        }

    def __repr__(self):
        schemas = list(self._results.keys())
        return f"EvaluationReport(target='{self.target}', schemas={schemas})"
