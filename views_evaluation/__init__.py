# ── Permanent public API ─────────────────────────────────────────────────────
# These classes are the stable, long-term interface of this library.
# They will remain after Phase 3 of the orchestrator migration.
from views_evaluation.evaluation.evaluation_frame import EvaluationFrame
from views_evaluation.evaluation.native_evaluator import NativeEvaluator
from views_evaluation.evaluation.evaluation_report import EvaluationReport
from views_evaluation.evaluation.metric_catalog import (
    MetricSpec,
    METRIC_CATALOG,
    METRIC_MEMBERSHIP,
    resolve_metric_params,
)
from views_evaluation.evaluation.config_schema import EvaluationConfig
from views_evaluation.profiles import PROFILES

# ── Temporary (PHASE-3-DELETE) ────────────────────────────────────────────────
# These classes exist for backward compatibility and parity testing while the
# orchestrator migration (ADR-011, report 10) completes in views-pipeline-core.
# They will be removed once upstream parity is confirmed. Do not build new
# integrations on them.
from views_evaluation.evaluation.evaluation_manager import EvaluationManager
from views_evaluation.adapters.pandas import PandasAdapter

__all__ = [
    # Permanent
    "EvaluationFrame",
    "NativeEvaluator",
    "EvaluationReport",
    "MetricSpec",
    "METRIC_CATALOG",
    "METRIC_MEMBERSHIP",
    "resolve_metric_params",
    "EvaluationConfig",
    "PROFILES",
    # Temporary — PHASE-3-DELETE
    "EvaluationManager",
    "PandasAdapter",
]
