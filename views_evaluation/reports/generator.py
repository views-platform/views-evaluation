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
        self.eval_report = {}

    def generate_eval_report_dict(self, df_preds: list[pd.DataFrame], df_eval_ts: pd.DataFrame, mean_prediction: float=None):
        """Return a dictionary with evaluation report data."""
        self.eval_report = {
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

        self.eval_report["Evaluation Results"].append(
            self._single_result(
                "Ensemble" if self.is_ensemble else "Model",
                self.config["name"],
                df_eval_ts,
                df_preds,
                mean_prediction
            )
        )
        return self.eval_report

    def update_ensemble_eval_report(self, model_name, df_preds: list[pd.DataFrame], df_eval_ts: pd.DataFrame, mean_prediction: float=None):
        self.eval_report["Evaluation Results"].append(
            self._single_result(
                "Constituent",
                model_name,
                df_eval_ts,
                df_preds,
                mean_prediction
            )
        )
        return self.eval_report

    def _forecast_type(self, df_preds: list[pd.DataFrame]):
        from views_evaluation.evaluation.evaluation_manager import EvaluationManager
        arr = [EvaluationManager.convert_to_array(df_pred, f"pred_{self.target}") for df_pred in df_preds]
        return "point" if not EvaluationManager.get_evaluation_type(arr, f"pred_{self.target}") else "uncertainty"

    def _partition(self, key: str):
        return self.config[self.run_type][key]

    def _single_result(self, model_type: str, model_name: str, df_eval_ts: pd.DataFrame, df_preds: list[pd.DataFrame], mean_prediction: float=None):
        from views_evaluation.evaluation.evaluation_manager import EvaluationManager
        df_preds = [
            EvaluationManager.transform_data(
                EvaluationManager.convert_to_array(df_pred, f"pred_{self.target}"), f"pred_{self.target}"
            )
            for df_pred in df_preds
        ]
        mse = df_eval_ts["MSE"].mean() 
        msle = df_eval_ts["MSLE"].mean()
        if mean_prediction is None:
            all_preds = np.concatenate([np.asarray(v).flatten() for df_pred in df_preds for v in df_pred[f"pred_{self.target}"]])
            mean_pred = np.mean(all_preds)
        else:
            mean_pred = mean_prediction
        
        return {
            "Type": model_type,
            "Model Name": model_name,
            "MSE": mse,
            "MSLE": msle,
            r"$\bar{\hat{y}}$": mean_pred
        }


