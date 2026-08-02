import numpy as np
from typing import Dict, List
from views_evaluation.evaluation.config_schema import EvaluationConfig
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
    # Config keys this evaluator understands, derived from the EvaluationConfig
    # TypedDict so the schema has exactly one authority (ADR-012). Adding a key
    # there is all that is required to support it here.
    _VALID_CONFIG_KEYS = frozenset(EvaluationConfig.__annotations__)

    # Each metric-list key declares the (task, pred_type) cell it supplies metrics for.
    _METRIC_LIST_KEYS = {
        "regression_point_metrics":      ("regression", "point"),
        "regression_sample_metrics":     ("regression", "sample"),
        "classification_point_metrics":  ("classification", "point"),
        "classification_sample_metrics": ("classification", "sample"),
    }

    @classmethod
    def _validate_config(cls, config: EvaluationConfig) -> None:
        """
        Fail loud on a structurally invalid config, at construction (ADR-015 rulings 4/5).

        Before this existed, a misspelled metric-list key or an absent ``steps`` key
        produced an empty-but-successful-looking report instead of an error
        (risk register C-02, Tier 1). Nothing is defaulted or repaired here — an
        invalid config is the caller's to fix (ADR-015).
        """
        if not isinstance(config, dict):
            raise ValueError(
                f"Evaluation config must be a dict, got {type(config).__name__}."
            )

        # 0. Profile name. Checked first so the pre-existing "Unknown evaluation
        #    profile" contract is preserved exactly for configs that hit both this
        #    and a structural problem below.
        profile_name = config.get("evaluation_profile", "base")
        if profile_name not in PROFILES:
            raise ValueError(
                f"Unknown evaluation profile '{profile_name}'. "
                f"Available: {sorted(PROFILES.keys())}"
            )

        # 1. Unknown / misspelled keys. Also the loud failure for legacy keys
        #    ('targets', 'metrics', '*_uncertainty_metrics') removed in 0.4.0.
        unknown = sorted(set(config) - cls._VALID_CONFIG_KEYS)
        if unknown:
            raise ValueError(
                f"Unknown evaluation config key(s): {unknown}. "
                f"Valid keys: {sorted(cls._VALID_CONFIG_KEYS)}. "
                f"Note: the legacy keys 'targets', 'metrics', "
                f"'regression_uncertainty_metrics' and 'classification_uncertainty_metrics' "
                f"were removed in 0.4.0 — see the README migration table."
            )

        # 2. 'steps' is required (CIC NativeEvaluator §4) and drives the step-wise schema.
        #    Absent, it silently produced no step-wise results at all.
        if not config.get("steps"):
            raise ValueError(
                "Evaluation config requires a non-empty 'steps' list (1-indexed step "
                "positions to evaluate, e.g. [1, 2, 3]). Without it the step-wise "
                "schema cannot be produced."
            )
        bad_steps = [s for s in config["steps"]
                     if not isinstance(s, int) or isinstance(s, bool) or s < 1]
        if bad_steps:
            raise ValueError(
                f"Evaluation config 'steps' must contain 1-indexed positive integers; "
                f"got invalid entries: {bad_steps}."
            )

        # 3. At least one target list must be declared, or nothing can be evaluated.
        declared_tasks = [
            task for task in ("regression", "classification")
            if config.get(f"{task}_targets")
        ]
        if not declared_tasks:
            raise ValueError(
                "Evaluation config declares no targets. Provide a non-empty "
                "'regression_targets' and/or 'classification_targets'."
            )

        # 4. Each declared task needs at least one metric list, else every group
        #    for that task evaluates to an empty dict.
        for task in declared_tasks:
            if not any(config.get(f"{task}_{pred_type}_metrics")
                       for pred_type in ("point", "sample")):
                raise ValueError(
                    f"Evaluation config declares '{task}_targets' but provides no "
                    f"metrics for it. Add a non-empty '{task}_point_metrics' and/or "
                    f"'{task}_sample_metrics'."
                )

        # 5. Every named metric must exist and be valid for the cell its key declares.
        for key, cell in cls._METRIC_LIST_KEYS.items():
            for metric in config.get(key, []):
                if metric not in METRIC_CATALOG:
                    raise ValueError(
                        f"Unknown metric '{metric}' in '{key}'. "
                        f"Available: {sorted(METRIC_CATALOG)}."
                    )
                if metric not in METRIC_MEMBERSHIP[cell]:
                    raise ValueError(
                        f"Metric '{metric}' in '{key}' is not valid for {cell}. "
                        f"Valid for {cell}: {sorted(METRIC_MEMBERSHIP[cell])}."
                    )

    def __init__(self, config: EvaluationConfig):
        self._validate_config(config)
        self.config = config

        # Profile name was validated in _validate_config (check 0).
        self.profile = PROFILES[config.get("evaluation_profile", "base")]
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

        # ADR-015 ruling 4: an empty metric list yields an empty-but-successful-looking
        # report. Construction-time validation cannot catch this case, because pred_type
        # is a property of the frame (n_samples > 1), not of the config.
        if not metrics_list:
            raise ValueError(
                f"No metrics configured for ({task}, {pred_type}). The frame for target "
                f"'{target}' has {ef.n_samples} sample(s) per row, so it is a "
                f"'{pred_type}' evaluation and requires a non-empty "
                f"'{task}_{pred_type}_metrics' list in the config."
            )

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
            try:
                results[m] = spec.function(ef.y_true, ef.y_pred, **resolved)
            except Exception as e:
                raise ValueError(
                    f"Metric '{m}' failed for ({task}, {pred_type}): {e}"
                ) from e
        return results

    def evaluate(self, ef: EvaluationFrame, legacy_compatibility: bool = False) -> EvaluationReport:
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
        max_allowed_step = float('inf')
        if legacy_compatibility:
            origin_indices = ef.get_group_indices('origin')
            seq_lengths = []
            for origin, idx in origin_indices.items():
                # Count unique steps per origin
                seq_lengths.append(len(np.unique(ef.identifiers['step'][idx])))
            max_allowed_step = min(seq_lengths) if seq_lengths else 0

            # ADR-015 ruling 7: config['steps'] is a request list, not a hint. Truncated
            # steps were left as pre-initialised empty dicts, so a requested step came
            # back looking evaluated while scoring nothing and emitting no MetricFrame
            # rows — the same silent-empty pattern as C-02. Silently not fulfilling an
            # explicit request is a contract violation regardless of which flag was set.
            dropped = sorted(s for s in config_steps if s > max_allowed_step)
            if dropped:
                raise ValueError(
                    f"legacy_compatibility=True truncates step-wise evaluation at step "
                    f"{max_allowed_step} (the shortest origin sequence has "
                    f"{max_allowed_step} step(s)), but config['steps'] explicitly "
                    f"requested {dropped}, which would be silently omitted. Either "
                    f"remove {dropped} from 'steps', or set legacy_compatibility=False "
                    f"to evaluate every step that has data."
                )

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
