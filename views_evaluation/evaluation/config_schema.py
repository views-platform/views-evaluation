"""
Typed schema for evaluation config dicts.

Type-checking only — no runtime enforcement. Regular dicts continue to work
at runtime. This TypedDict documents the expected config structure and
enables IDE autocompletion and static analysis (mypy, pyright).
"""

from typing import Any, Dict, List, TypedDict


class EvaluationConfig(TypedDict, total=False):
    """
    Config dict for NativeEvaluator.

    All keys are optional (total=False) to match the existing .get() patterns.
    Downstream validators (EvaluationManager._validate_config) enforce
    required-key semantics at runtime.

    Keys:
        steps: List of 1-indexed step positions to evaluate.
        regression_targets: Target names for regression tasks.
        classification_targets: Target names for classification tasks.
        regression_point_metrics: Metric names for regression point predictions.
        regression_sample_metrics: Metric names for regression sample predictions.
        classification_point_metrics: Metric names for classification point predictions.
        classification_sample_metrics: Metric names for classification sample predictions.
        evaluation_profile: Name of a named profile (default: "base").
        metric_hyperparameters: Per-metric override parameters.
            E.g. {"twCRPS": {"threshold": 2.0}, "Coverage": {"alpha": 0.05}}
    """

    steps: List[int]
    regression_targets: List[str]
    classification_targets: List[str]
    regression_point_metrics: List[str]
    regression_sample_metrics: List[str]
    classification_point_metrics: List[str]
    classification_sample_metrics: List[str]
    evaluation_profile: str
    metric_hyperparameters: Dict[str, Dict[str, Any]]
