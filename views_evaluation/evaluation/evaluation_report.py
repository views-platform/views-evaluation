import dataclasses
import warnings
from typing import Dict, Any, Optional
from views_evaluation.evaluation.metrics import (
    RegressionPointEvaluationMetrics,
    RegressionSampleEvaluationMetrics,
    ClassificationPointEvaluationMetrics,
    ClassificationSampleEvaluationMetrics,
)

class EvaluationReport:
    """
    A structured, framework-agnostic container for evaluation results.
    
    This class decouples the raw result data from its final presentation
    format, allowing for flexible export to JSON, Dictionaries, or Pandas.
    """
    def __init__(self, target: str, task: str, pred_type: str, results: Dict[str, Dict[str, Any]]):
        self.target = target
        self.task = task
        self.pred_type = pred_type
        # Internal structure: {schema_name: {group_id: {metric_name: value}}}
        self._results = results
        
        # Map task/type to legacy dataclasses for formatting
        self._metrics_map = {
            ("regression", "point"): RegressionPointEvaluationMetrics,
            ("regression", "sample"): RegressionSampleEvaluationMetrics,
            ("classification", "point"): ClassificationPointEvaluationMetrics,
            ("classification", "sample"): ClassificationSampleEvaluationMetrics,
        }

    def _get_metrics_cls(self):
        return self._metrics_map[(self.task, self.pred_type)]

    def get_schema_results(self, schema: str) -> Dict[str, Any]:
        """
        Returns the result dictionary for a specific schema, 
        mapped to legacy dataclass instances for backward compatibility.
        """
        if schema not in self._results:
            raise KeyError(f"Schema '{schema}' not found in report.")
        
        raw_results = self._results[schema]
        metrics_cls = self._get_metrics_cls()
        
        valid_fields = {f.name for f in dataclasses.fields(metrics_cls)}
        mapped_results = {}
        for group_id, metrics in raw_results.items():
            container = metrics_cls()
            for k, v in metrics.items():
                if k not in valid_fields:
                    raise ValueError(
                        f"Metric '{k}' computed for ({self.task}, {self.pred_type}) "
                        f"but no field exists in {metrics_cls.__name__}. "
                        f"Add '{k}: Optional[float] = None' to the dataclass."
                    )
                setattr(container, k, v)
            mapped_results[group_id] = container
            
        return mapped_results

    def to_dataframe(self, schema: str):
        """
        Converts a specific schema's results into a Pandas DataFrame.
        If schema='raw', returns the internal results dict — the same object as
        ``to_dict()['schemas']``; do not mutate it.

        **Deprecated; removed in 2.0.0** (ADR-022 §2; register C-40, C-44). Build a
        DataFrame in the caller from ``to_dict()['schemas'][schema]`` instead:
        ``pd.DataFrame.from_dict(report.to_dict()['schemas'][schema], orient='index')``
        carries the same values. It is not byte-identical: this method orders columns by
        dataclass field, drops any column that is NaN in every group (C-40), and keeps a
        group with no metrics as a NaN row; the recipe orders columns as the metrics were
        configured, keeps every column, and omits an empty group.
        """
        # One warning on every call, whatever the schema: the `raw` passthrough used to
        # carry its own; it is folded in here so a caller sees exactly one.
        replacement = "to_dict()['schemas']" if schema == "raw" else "to_dict()['schemas'][schema]"
        warnings.warn(
            f"EvaluationReport.to_dataframe() is deprecated and will be removed in 2.0.0. "
            f"Build a DataFrame from {replacement} in the caller.",
            DeprecationWarning,
            stacklevel=2,
        )
        if schema == "raw":
            return self._results

        # Gate on find_spec so the helpful error fires only when the extra is truly
        # absent; a genuine import error inside pandas then propagates loudly. Until
        # 2026-09-17 this was a bare `import pandas` that surfaced as
        # `ModuleNotFoundError: No module named 'pandas'` with no mention of the extra
        # (register C-44). Raised as ModuleNotFoundError — the type the bare import
        # raised — so no `except ModuleNotFoundError` caller changes behaviour (ADR-022
        # §1 counts raised types as public surface). No log: this path computes a value
        # in memory and Level 0 does not log (logging standard §5.1); only the
        # `to_metric_frame()` emit path below does.
        import importlib.util
        if importlib.util.find_spec("pandas") is None:
            raise ModuleNotFoundError(
                "EvaluationReport.to_dataframe() requires the optional 'pandas' "
                "dependency. Install it with: pip install views-evaluation[dataframe]",
                name="pandas",
            )
        import pandas as pd
        mapped_results = self.get_schema_results(schema)
        if not mapped_results:
            return pd.DataFrame()
        
        metrics_cls = self._get_metrics_cls()
        # The helper warns on its own for direct callers; this call already has.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return metrics_cls.evaluation_dict_to_dataframe(mapped_results)


    def to_metric_frame(
        self,
        *,
        model_id: Optional[str] = None,
        run_id: Optional[str] = None,
        data_version: Optional[str] = None,
        run_type: Optional[str] = None,
        timestamp: Optional[int] = None,
        seed: Optional[int] = None,
        partition: Optional[str] = None,
        level: Optional[str] = None,
        scoring_code_version: Optional[str] = None,
        evaluation_timestamp: Optional[str] = None,
    ):
        """
        Emit this report as a typed, provenance-stamped ``MetricFrame`` (views-frames ADR-020).

        A Level-1 bridge: it flattens the nested per-group results into rows keyed by
        ``(eval_type, target, metric, group_id, partition, level)`` and attaches provenance.
        ``to_dict()`` is unaffected — this is purely additive. (``to_dataframe()`` is
        deprecated and goes in 2.0.0.)

        For each schema (month/time_series/step) present, one row is emitted per
        (group_id, metric), PLUS a cross-group aggregate row with ``group_id="mean"`` carrying
        the ``nanmean`` over the groups that reported that metric (the value views-reporting
        matches on). The denominator is every group that reported a number: a group
        carrying `nan` (MCR, Pearson or AP on degenerate input, ADR-015 R1/R2/R9) is
        excluded, and a metric present in only some groups is averaged over just those.
        `inf` is not `nan`: MCR's `inf` (predicted conflict where none occurred, R1) is a
        calibration statement, `nanmean` keeps it, and the mean row reads `inf`. Schema
        names are mapped to the consumer-facing ``eval_type`` spelling via
        ``SCHEMA_TO_EVAL_TYPE``.

        Provenance is split per ADR-020 (register C-47): generic identity goes in the reused
        ``views_frames.FrameMetadata``; ``scoring_code_version`` and ``evaluation_timestamp``
        stay in the MetricFrame's own metadata. ``scoring_code_version`` defaults to the
        installed package version (NOT a git SHA — unavailable in an installed wheel).
        ``evaluation_timestamp`` is caller-injected (not auto-stamped) to keep output deterministic.

        All identity is injected, not inferred — pipeline-core supplies model_id/run_id/
        data_version/partition/level at the evaluation call site (``run_id`` may be None at
        emit time, when the WandB run does not yet exist).

        Requires the optional ``views-frames`` dependency
        (``pip install views-evaluation[frames]``).
        """
        # Gate on find_spec so the helpful error fires only when the extra is truly absent;
        # genuine import errors inside numpy/metric_frame then propagate loudly (not masked).
        import importlib.util
        if importlib.util.find_spec("views_frames") is None:
            # Level-1 emit path: logs before raising even though this guard physically
            # resides in a Level-0 file. Logging follows the emit path, not the file
            # (logging standard §5.1). No other raise in this module logs.
            import logging
            err_msg = (
                "EvaluationReport.to_metric_frame() requires the optional 'views-frames' "
                "dependency. Install it with: pip install views-evaluation[frames]"
            )
            logging.getLogger("views_evaluation.evaluation.metric_frame").error(err_msg)
            raise ImportError(err_msg)
        import numpy as np
        from views_frames import FrameMetadata
        from views_evaluation.evaluation.metric_frame import (
            MetricFrame,
            MetricFrameMetadata,
            SCHEMA_TO_EVAL_TYPE,
            MEAN_GROUP_ID,
            AXES,
            default_scoring_code_version,
        )

        if scoring_code_version is None:
            scoring_code_version = default_scoring_code_version()

        # Missing partition/level become "" — reporting does not key on these axes, but a
        # present (constant) column keeps the frame's key space complete (ADR-020).
        partition_str = "" if partition is None else str(partition)
        level_str = "" if level is None else str(level)

        columns: Dict[str, list] = {axis: [] for axis in AXES}
        values: list = []

        def _emit(eval_type: str, metric: str, group_id: str, value: float) -> None:
            columns["eval_type"].append(eval_type)
            columns["target"].append(str(self.target))
            columns["metric"].append(str(metric))
            columns["group_id"].append(str(group_id))
            columns["partition"].append(partition_str)
            columns["level"].append(level_str)
            values.append(value)

        for schema, eval_type in SCHEMA_TO_EVAL_TYPE.items():
            group_results = self._results.get(schema, {})
            if not group_results:
                continue

            # Per-group rows + accumulate per-metric values for the aggregate row.
            metric_order: list = []
            metric_values: Dict[str, list] = {}
            for group_id, metrics in group_results.items():
                for metric, value in metrics.items():
                    _emit(eval_type, metric, group_id, value)
                    if metric not in metric_values:
                        metric_values[metric] = []
                        metric_order.append(metric)
                    metric_values[metric].append(value)

            # Cross-group aggregate row (group_id="mean") — what views-reporting reads.
            for metric in metric_order:
                arr = np.asarray(metric_values[metric], dtype=np.float64)
                mean = float("nan") if np.all(np.isnan(arr)) else float(np.nanmean(arr))
                _emit(eval_type, metric, MEAN_GROUP_ID, mean)

        # ADR-015 ruling 6: an empty emit produces a structurally valid zero-row
        # MetricFrame that passes every envelope check and persists to disk as a
        # legitimate audit artifact — indistinguishable from a real one, and rendering
        # downstream as "not calculated" exactly like a genuine metric failure.
        # The evaluation-of-record must never record nothing while looking complete.
        if not values:
            present = sorted(k for k, v in self._results.items() if v)
            empty = sorted(k for k, v in self._results.items() if not v)
            err_msg = (
                f"to_metric_frame() produced no rows — the report contains no metric "
                f"values for any schema, so the emitted evaluation-of-record would be "
                f"empty. Target='{self.target}', task='{self.task}', "
                f"pred_type='{self.pred_type}'. Schemas with groups but no metric "
                f"values: {present or 'none'}; schemas with no groups: {empty or 'none'}. "
                f"This usually means the evaluator was misconfigured."
            )
            # Level-1 emit path: log before raising (logging standard §5.1).
            import logging
            logging.getLogger("views_evaluation.evaluation.metric_frame").error(err_msg)
            raise ValueError(err_msg)

        # ADR-015 R6, amended 2026-09-18: a frame in which EVERY value is a sentinel —
        # every group of every metric degenerate — is structurally valid and persists,
        # but it records nothing while looking complete. Raising here would abort a
        # legitimate workflow (a constant baseline scored only on Pearson), which is
        # the R2 reversal; so it logs at WARNING on the emit path and emits.
        # Tested on the coerced array, not the raw list: a `None` value coerces to
        # nan under float32 exactly as it did in 1.0.0, and `np.isnan(None)` would
        # have raised TypeError before the coercion could happen.
        values_arr = np.asarray(values, dtype=np.float32).reshape(-1, 1)
        if np.isnan(values_arr).all():
            import logging
            logging.getLogger("views_evaluation.evaluation.metric_frame").warning(
                "to_metric_frame(): every value is a sentinel (nan) — no metric produced "
                "a number for any group. Target='%s', task='%s', pred_type='%s'. The frame "
                "is emitted, but an evaluation that scores nothing usually means the truth "
                "column is degenerate (all zeros, or constant) or the wrong column was passed.",
                self.target, self.task, self.pred_type,
            )
        identifiers = {axis: np.asarray(columns[axis], dtype=str) for axis in AXES}

        metadata = MetricFrameMetadata(
            provenance=FrameMetadata(
                model=model_id,
                run_type=run_type,
                timestamp=timestamp,
                seed=seed,
                run_id=run_id,
                data_version=data_version,
            ),
            scoring_code_version=scoring_code_version,
            evaluation_timestamp=evaluation_timestamp,
        )
        return MetricFrame(values=values_arr, identifiers=identifiers, metadata=metadata)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the entire report into a nested dictionary."""
        return {
            "target": self.target,
            "task": self.task,
            "pred_type": self.pred_type,
            "schemas": self._results
        }

    def __repr__(self):
        schemas = list(self._results.keys())
        return f"EvaluationReport(target='{self.target}', task='{self.task}', schemas={schemas})"
