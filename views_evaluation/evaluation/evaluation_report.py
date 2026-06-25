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
        If schema='raw', returns the dictionary of mapped metrics dataclasses.
        """
        if schema == "raw":
            warnings.warn(
                "to_dataframe(schema='raw') is deprecated. Use to_dict()['schemas'] instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return self._results
            
        import pandas as pd
        mapped_results = self.get_schema_results(schema)
        if not mapped_results:
            return pd.DataFrame()
        
        metrics_cls = self._get_metrics_cls()
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
        ``to_dict()``/``to_dataframe()`` are unaffected — this is purely additive.

        For each schema (month/time_series/step) present, one row is emitted per
        (group_id, metric), PLUS a cross-group aggregate row with ``group_id="mean"`` carrying
        the mean over groups (the value views-reporting matches on). Schema names are mapped to
        the consumer-facing ``eval_type`` spelling via ``SCHEMA_TO_EVAL_TYPE``.

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
            raise ImportError(
                "EvaluationReport.to_metric_frame() requires the optional 'views-frames' "
                "dependency. Install it with: pip install views-evaluation[frames]"
            )
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

        values_arr = np.asarray(values, dtype=np.float32).reshape(-1, 1)
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
