import numpy as np
from typing import Dict, List
from views_evaluation.evaluation.evaluation_frame import EvaluationFrame
from views_evaluation.evaluation.evaluation_report import EvaluationReport
from views_evaluation.evaluation.metric_catalog import (
    METRIC_CATALOG,
    METRIC_MEMBERSHIP,
    resolve_metric_params,
)
from views_evaluation.profiles import PROFILES

class NativeEvaluator:
    """
    The 'Pure Math Engine' that operates on EvaluationFrame.
    Reproduces the three schemas using native grouping.

    Uses the MetricCatalog for dispatch and the Chain of Responsibility
    pattern for hyperparameter resolution:
        model overrides → evaluation profile → fail loud

    Config keys:
        evaluation_profile (str): Name of the evaluation profile to use.
            Must be a key in PROFILES. Defaults to "base" during transition.
        metric_hyperparameters (dict): Optional per-metric overrides.
            E.g. {"twCRPS": {"threshold": 2.0}, "Coverage": {"alpha": 0.05}}
    """
    def __init__(self, config: dict):
        self.config = config

        # Resolve evaluation profile
        profile_name = config.get("evaluation_profile", "base")
        if profile_name not in PROFILES:
            raise ValueError(
                f"Unknown evaluation profile '{profile_name}'. "
                f"Available: {sorted(PROFILES.keys())}"
            )
        self.profile = PROFILES[profile_name]
        self.metric_overrides = config.get("metric_hyperparameters", {})

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

        return metrics_list, task, pred_type

    def _calculate_metrics(self, ef: EvaluationFrame, metrics_list: List[str],
                           task: str, pred_type: str) -> Dict[str, float]:
        """
        Calculates metrics for a single EvaluationFrame view using native NumPy logic.

        Resolves hyperparameters via Chain of Responsibility:
            model overrides → evaluation profile → fail loud
        """
        valid_metrics = METRIC_MEMBERSHIP[(task, pred_type)]
        results = {}
        for m in metrics_list:
            if m not in valid_metrics:
                # ADR-013: Fail loud on missing implementations
                raise ValueError(f"Metric '{m}' is not valid for ({task}, {pred_type}).")
            spec = METRIC_CATALOG[m]
            overrides = self.metric_overrides.get(m, {})
            resolved = resolve_metric_params(m, overrides, self.profile)
            results[m] = spec.function(ef.y_true, ef.y_pred, **resolved)
        return results

    def evaluate(self, ef: EvaluationFrame, legacy_compatibility: bool = True) -> EvaluationReport:
        metrics_list, task, pred_type = self._resolve_task_and_metrics(ef)

        results = {}

        # 1. Month-wise
        month_results = {}
        month_indices = ef.get_group_indices('time')
        for month, idx in month_indices.items():
            sub_ef = ef.select_indices(idx)
            month_results[f"month{month}"] = self._calculate_metrics(sub_ef, metrics_list, task, pred_type)
        results["month"] = month_results

        # 2. Sequence-wise (Time-Series)
        ts_results = {}
        origin_indices = ef.get_group_indices('origin')
        for origin, idx in origin_indices.items():
            sub_ef = ef.select_indices(idx)
            ts_results[f"ts{str(origin).zfill(2)}"] = self._calculate_metrics(sub_ef, metrics_list, task, pred_type)
        results["time_series"] = ts_results

        # 3. Step-wise
        step_results = {}
        config_steps = self.config.get("steps", [])
        if config_steps:
            # Pre-initialize only the explicitly declared steps (not all steps up to max)
            step_results = {f"step{str(s).zfill(2)}": {} for s in config_steps}

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
                step_results[key] = self._calculate_metrics(sub_ef, metrics_list, task, pred_type)
        results["step"] = step_results

        return EvaluationReport(
            target=ef.metadata.get('target', 'unknown'),
            task=task,
            pred_type=pred_type,
            results=results
        )
