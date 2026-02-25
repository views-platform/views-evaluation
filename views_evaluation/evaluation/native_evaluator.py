import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from views_evaluation.evaluation.evaluation_frame import EvaluationFrame
from views_evaluation.evaluation.metrics import (
    RegressionPointEvaluationMetrics,
    RegressionSampleEvaluationMetrics,
    ClassificationPointEvaluationMetrics,
    ClassificationSampleEvaluationMetrics,
)
from views_evaluation.evaluation.metric_calculators import (
    REGRESSION_POINT_METRIC_FUNCTIONS,
    REGRESSION_SAMPLE_METRIC_FUNCTIONS,
    CLASSIFICATION_POINT_METRIC_FUNCTIONS,
    CLASSIFICATION_SAMPLE_METRIC_FUNCTIONS,
)

class NativeEvaluator:
    """
    The 'Pure Math Engine' that operates on EvaluationFrame.
    Reproduces the three schemas using native grouping.
    """
    def __init__(self, config: dict):
        self.config = config
        # Mapping task/type to metric classes (reusing existing ones for parity)
        self.metrics_map = {
            ("regression", "point"): (REGRESSION_POINT_METRIC_FUNCTIONS, RegressionPointEvaluationMetrics),
            ("regression", "sample"): (REGRESSION_SAMPLE_METRIC_FUNCTIONS, RegressionSampleEvaluationMetrics),
            ("classification", "point"): (CLASSIFICATION_POINT_METRIC_FUNCTIONS, ClassificationPointEvaluationMetrics),
            ("classification", "sample"): (CLASSIFICATION_SAMPLE_METRIC_FUNCTIONS, ClassificationSampleEvaluationMetrics),
        }

    def _resolve_task_and_metrics(self, ef: EvaluationFrame):
        target = ef.metadata.get('target')
        if target in self.config.get("regression_targets", []):
            task = "regression"
        elif target in self.config.get("classification_targets", []):
            task = "classification"
        else:
            raise ValueError(f"Target {target} not found in config")
        
        pred_type = "sample" if ef.is_sample else "point"
        metrics_list = self.config.get(f"{task}_{pred_type}_metrics", [])
        
        funcs, cls = self.metrics_map[(task, pred_type)]
        return metrics_list, funcs, cls

    def _calculate_metrics(self, ef: EvaluationFrame, metrics_list: List[str], funcs: Dict) -> Dict[str, float]:
        """
        Calculates metrics for a single EvaluationFrame view.
        NOTE: For parity, we currently wrap the ef back into a DataFrame 
        to use existing calculators, but we will eventually vectorise.
        """
        # TEMP: Re-wrap to use existing calculators for bit-wise parity investigation
        # These calculators expect DataFrames with specific columns
        target = ef.metadata['target']
        matched_actual = pd.DataFrame({target: [[v] for v in ef.y_true]})
        # Handle samples: legacy expects Series[List]
        if ef.is_sample:
            matched_pred = pd.DataFrame({f"pred_{target}": [list(row) for row in ef.y_pred]})
        else:
            matched_pred = pd.DataFrame({f"pred_{target}": [[v] for v in ef.y_pred.flatten()]})
        
        results = {}
        for m in metrics_list:
            results[m] = funcs[m](matched_actual, matched_pred, target)
        return results

    def evaluate(self, ef: EvaluationFrame, legacy_compatibility: bool = True) -> Dict[str, Tuple[Dict, pd.DataFrame]]:
        metrics_list, funcs, metrics_cls = self._resolve_task_and_metrics(ef)
        
        results = {}
        
        # 1. Month-wise
        month_dict = {}
        month_indices = ef.get_group_indices('time')
        for month, idx in month_indices.items():
            sub_ef = ef.select_indices(idx)
            m_results = self._calculate_metrics(sub_ef, metrics_list, funcs)
            container = metrics_cls()
            for k, v in m_results.items():
                setattr(container, k, v)
            month_dict[f"month{month}"] = container
        results["month"] = (month_dict, metrics_cls.evaluation_dict_to_dataframe(month_dict))
        
        # 2. Sequence-wise (Time-Series)
        ts_dict = {}
        origin_indices = ef.get_group_indices('origin')
        for origin, idx in origin_indices.items():
            sub_ef = ef.select_indices(idx)
            ts_results = self._calculate_metrics(sub_ef, metrics_list, funcs)
            container = metrics_cls()
            for k, v in ts_results.items():
                setattr(container, k, v)
            ts_dict[f"ts{str(origin).zfill(2)}"] = container
        results["time_series"] = (ts_dict, metrics_cls.evaluation_dict_to_dataframe(ts_dict))
        
        # 3. Step-wise
        step_dict = {}
        config_steps = self.config.get("steps", [])
        if config_steps:
            max_step = max(config_steps)
            step_dict = {f"step{str(i).zfill(2)}": metrics_cls() for i in range(1, max_step + 1)}
        
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
            if key in step_dict:
                sub_ef = ef.select_indices(idx)
                s_results = self._calculate_metrics(sub_ef, metrics_list, funcs)
                container = step_dict[key]
                for k, v in s_results.items():
                    setattr(container, k, v)
        results["step"] = (step_dict, metrics_cls.evaluation_dict_to_dataframe(step_dict))
        
        return results
