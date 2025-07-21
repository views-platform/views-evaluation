import numpy as np
import pandas as pd


class EvalReportGenerator:
    """Generate evaluation reports for ensemble or single model forecasts."""

    def __init__(self, config: dict, target: str, conflict_type: str):
        self.config = config
        self.target = target
        self.conflict_type = conflict_type
        self.level = config.get("level")
        self.run_type = config.get("run_type")
        self.eval_type = config.get("eval_type")
        self.is_ensemble = True if "models" in config else False

    def generate_eval_report_dict(self, df_preds: list[pd.DataFrame], df_eval_ts: pd.DataFrame):
        """Return a dictionary with evaluation report data."""
        eval_report = {
            "Target": self.target,
            "Forecast Type": self._forecast_type(df_preds),
            "Level of Analysis": self.level,
            "Data Partition": self.run_type,
            "Training Period": self._partition("train"),
            "Testing Period": self._partition("test"),
            "Forecast Horizon": len(self.config.get("steps", [])),
            "Number of Rolling Origins": len(df_preds), 
            "Evaluation Results": []
        }

        eval_report["Evaluation Results"].append(
            self._single_result(
                "Ensemble" if self.is_ensemble else "Model",
                self.config["name"],
                df_eval_ts,
                df_preds
            )
        )

        if self.is_ensemble:
            from views_pipeline_core.managers.model import ModelPathManager
            for model_name in self.config["models"]:
                pm = ModelPathManager(model_name)
                eval_report["Evaluation Results"].append(
                    self._single_result(
                        "Constituent",
                        model_name,
                        self._eval_ts(pm),
                        self._preds(pm, rolling_origin_number=len(df_preds))
                    )
                )
        return eval_report

    def _forecast_type(self, df_preds: list[pd.DataFrame]):
        from views_evaluation.evaluation.evaluation_manager import EvaluationManager
        arr = [EvaluationManager.convert_to_array(df_pred, f"pred_{self.target}") for df_pred in df_preds]
        return "point" if not EvaluationManager.get_evaluation_type(arr, f"pred_{self.target}") else "uncertainty"

    def _partition(self, key: str):
        return self.config[self.run_type][key]

    def _eval_ts(self, pm):
        from views_pipeline_core.files.utils import read_dataframe
        path = pm._get_eval_file_paths(self.run_type, self.conflict_type)[0]
        return read_dataframe(path)

    def _preds(self, pm, rolling_origin_number: int):
        from views_pipeline_core.files.utils import read_dataframe
        paths = pm._get_generated_predictions_data_file_paths(self.run_type)[:rolling_origin_number]
        return [read_dataframe(path) for path in paths]

    def _single_result(self, model_type: str, model_name: str, df_eval_ts: pd.DataFrame, df_preds: list[pd.DataFrame]):
        # mse = df_eval_ts["MSE"].mean() # Add back after publishing latest version of views-evaluation
        msle = np.sqrt(df_eval_ts["RMSLE"]).mean()
        mean_pred = np.mean([df_pred[f"pred_{self.target}"].mean() for df_pred in df_preds])
        
        return {
            "Type": model_type,
            "Model Name": model_name,
            # "MSE": mse,
            "MSLE": msle,
            "mean prediction": mean_pred
        }


