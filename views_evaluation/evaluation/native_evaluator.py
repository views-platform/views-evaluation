import numpy as np
from typing import Dict, List
from views_evaluation.evaluation.evaluation_frame import EvaluationFrame
from views_evaluation.evaluation.evaluation_report import EvaluationReport
from views_evaluation.evaluation.native_metric_calculators import (
    REGRESSION_POINT_NATIVE,
    REGRESSION_SAMPLE_NATIVE,
    CLASSIFICATION_POINT_NATIVE,
    CLASSIFICATION_SAMPLE_NATIVE,
)

class NativeEvaluator:
    """
    The 'Pure Math Engine' that operates on EvaluationFrame.
    Reproduces the three schemas using native grouping.
    """
    def __init__(self, config: dict):
        self.config = config
        # Mapping task/type to metric dispatch dicts (No dataclasses here)
        self.metrics_map = {
            ("regression", "point"): REGRESSION_POINT_NATIVE,
            ("regression", "sample"): REGRESSION_SAMPLE_NATIVE,
            ("classification", "point"): CLASSIFICATION_POINT_NATIVE,
            ("classification", "sample"): CLASSIFICATION_SAMPLE_NATIVE,
        }

    def _resolve_task_and_metrics(self, ef: EvaluationFrame):
        target = ef.metadata.get('target')
        # Determine task from config
        if target in self.config.get("regression_targets", []):
            task = "regression"
        elif target in self.config.get("classification_targets", []):
            task = "classification"
        else:
            raise ValueError(f"Target {target} not found in config")
        
        pred_type = "sample" if ef.is_sample else "point"
        metrics_list = self.config.get(f"{task}_{pred_type}_metrics", [])
        
        funcs = self.metrics_map[(task, pred_type)]
        return metrics_list, funcs, task, pred_type

    def _calculate_metrics(self, ef: EvaluationFrame, metrics_list: List[str], funcs: Dict) -> Dict[str, float]:
        """
        Calculates metrics for a single EvaluationFrame view using native NumPy logic.
        """
        results = {}
        for m in metrics_list:
            if m not in funcs:
                # ADR-013: Fail loud on missing implementations
                # Re-wrap as ValueError to match legacy test expectations
                raise ValueError(f"Metric '{m}' is not valid for this task.")
            results[m] = funcs[m](ef.y_true, ef.y_pred)
        return results

    def evaluate(self, ef: EvaluationFrame, legacy_compatibility: bool = True) -> EvaluationReport:
        metrics_list, funcs, task, pred_type = self._resolve_task_and_metrics(ef)
        
        results = {}
        
        # 1. Month-wise
        month_results = {}
        month_indices = ef.get_group_indices('time')
        for month, idx in month_indices.items():
            sub_ef = ef.select_indices(idx)
            month_results[f"month{month}"] = self._calculate_metrics(sub_ef, metrics_list, funcs)
        results["month"] = month_results
        
        # 2. Sequence-wise (Time-Series)
        ts_results = {}
        origin_indices = ef.get_group_indices('origin')
        for origin, idx in origin_indices.items():
            sub_ef = ef.select_indices(idx)
            ts_results[f"ts{str(origin).zfill(2)}"] = self._calculate_metrics(sub_ef, metrics_list, funcs)
        results["time_series"] = ts_results
        
        # 3. Step-wise
        step_results = {}
        config_steps = self.config.get("steps", [])
        if config_steps:
            max_step = max(config_steps)
            # Pre-initialize with empty results for all steps up to max
            step_results = {f"step{str(i).zfill(2)}": {} for i in range(1, max_step + 1)}
        
        # LEGACY PARITY: Truncate steps to the shortest sequence length if in compat mode
        max_allowed_step = 999
        if legacy_compatibility:
            origin_indices = ef.get_group_indices('origin')
            seq_lengths = []
            for origin, idx in origin_indices.items():
                # Count unique steps per origin
                seq_lengths.append(len(np.unique(ef.identifiers['step'][idx])))
            max_allowed_step = min(seq_lengths) if seq_lengths else 0

        step_indices = ef.get_group_indices('step')
        for step, idx in step_indices.items():
            if step > max_allowed_step:
                continue
                
            key = f"step{str(step).zfill(2)}"
            if key in step_results:
                sub_ef = ef.select_indices(idx)
                step_results[key] = self._calculate_metrics(sub_ef, metrics_list, funcs)
        results["step"] = step_results
        
        return EvaluationReport(
            target=ef.metadata.get('target', 'unknown'), 
            task=task,
            pred_type=pred_type,
            results=results
        )

