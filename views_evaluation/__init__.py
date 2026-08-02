# ── Public API ────────────────────────────────────────────────────────────────
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

__all__ = [
    "EvaluationFrame",
    "NativeEvaluator",
    "EvaluationReport",
    "MetricSpec",
    "METRIC_CATALOG",
    "METRIC_MEMBERSHIP",
    "resolve_metric_params",
    "EvaluationConfig",
    "PROFILES",
]

# MetricFrame requires the optional 'views-frames' dependency. Expose it only when the
# extra is installed, so the core API stays importable without it (ADR-011 minimal core).
# Gate on find_spec (not a bare ImportError catch) so a genuine broken import inside
# metric_frame.py still surfaces loudly when views-frames IS installed.
import importlib.util as _importlib_util

if _importlib_util.find_spec("views_frames") is not None:
    from views_evaluation.evaluation.metric_frame import MetricFrame, MetricFrameMetadata
    __all__ += ["MetricFrame", "MetricFrameMetadata"]
