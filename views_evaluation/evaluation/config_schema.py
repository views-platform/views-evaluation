"""
Typed schema for evaluation config dicts.

This TypedDict is the **single authority** for which config keys the evaluator
understands. It serves two purposes:

  1. Static analysis — documents the expected structure and enables IDE
     autocompletion and type checking (mypy, pyright).
  2. **Runtime validation** — ``NativeEvaluator._validate_config`` derives its
     valid key set directly from ``EvaluationConfig.__annotations__``, so adding
     a key here is all that is required to have the evaluator accept it. There is
     no second, hand-maintained list to keep in step.

Regular dicts continue to work at runtime; the TypedDict is never instantiated.
"""

from typing import Any, Dict, List, TypedDict


class EvaluationConfig(TypedDict, total=False):
    """
    Config dict for NativeEvaluator.

    All keys are ``total=False`` for typing purposes only — that does NOT mean they
    are optional at runtime. ``NativeEvaluator._validate_config`` enforces which keys
    are actually required (``steps``, at least one target list, and at least one
    metric list per declared task), raising at construction if they are absent.

    (Until 0.4.0 this docstring named ``EvaluationManager._validate_config`` as the
    enforcer. That class was removed in Phase 3 and validation was unowned until
    0.5.0 — risk register C-02.)

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
