# Class Intent Contract: EvaluationReport

**Status:** Active  
**Owner:** Evaluation Core  
**Last reviewed:** 2026-08-02  
**Related ADRs:** ADR-010 (Ontology), ADR-041 (Output Schema), views-frames ADR-020 (MetricFrame contract home)

---

## 1. Purpose

A structured, framework-agnostic container for evaluation results. It decouples the mathematical outcomes from their final presentation format (DataFrames, JSON, etc.).

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** calculate metrics.
- This class does **not** handle data alignment.
- This class does **not** depend on Pandas internally (though it may provide conversion methods).

---

## 3. Responsibilities and Guarantees

- **Multi-Schema Storage**: Guarantees storage of results across Month, Sequence, and Step schemas.
- **Traceability**: Maintains metadata about the target, task type, and prediction type.
- **Representation Agnosticism**: Provides a standard internal representation that can be converted to external formats (Dictionary or Pandas).
- **Field Validation**: Guarantees that computed metric names match dataclass fields. Raises `ValueError` with an actionable message if a metric is computed but has no corresponding field in the typed container (FM1 guard).

---

## 4. Inputs and Assumptions

- **Constructor**: `EvaluationReport(target: str, task: str, pred_type: str, results: Dict[str, Dict[str, Any]])`.
  - `target`: target variable name (e.g. `"ged_sb_best"`)
  - `task`: `"regression"` or `"classification"`
  - `pred_type`: `"point"` or `"sample"`
  - `results`: nested dict `{schema_name: {group_id: {metric_name: value}}}`
- **Pre-computed Results**: Assumes results have already been computed by `NativeEvaluator`.
- **Schema Conformity**: Expects data organized by the standard Views schemas.

---

## 5. Outputs and Side Effects

- **`to_dict()`**: Returns a nested dictionary `{target, task, pred_type, schemas: {...}}` for persistence (ADR-041).
- **`get_schema_results(schema)`**: Returns a dict mapping group keys to typed metrics dataclass instances.
- **`to_dataframe(schema)`**: Returns a Pandas DataFrame for backward compatibility. `schema='raw'` is deprecated — use `to_dict()['schemas']` instead.
- **`to_metric_frame(...)`**: Returns a `MetricFrame` (the typed, transportable, provenance-stamped evaluation-of-record per views-frames ADR-020). Flattens the nested results to rows keyed by `(eval_type, target, metric, group_id, partition, level)`, emitting per-group rows plus a `group_id="mean"` cross-group aggregate. All identity (`model_id`/`run_id`/`data_version`/`partition`/`level`) is **injected** by the caller; `scoring_code_version` (defaults to the installed package version) and `evaluation_timestamp` are stamped into the MetricFrame's own metadata (the generic header stays in the reused `views_frames.FrameMetadata`). Requires the optional `views-frames` dependency. Purely additive — `to_dict()`/`to_dataframe()` are unaffected.

---

## 6. Failure Modes and Loudness

- Raises `KeyError` if a requested schema is not found in the report.
- Raises `ValueError` if a computed metric name has no corresponding field in the typed metrics dataclass (FM1 guard against silent metric loss).
- Fails loud if the input result structure is inconsistent.
- `to_metric_frame()` raises `ImportError` (loud, actionable) if the optional `views-frames` dependency is not installed.
- `to_metric_frame()` raises `ValueError` if the report contains **no metric values for any schema** (ADR-015 ruling 6, risk register C-30). Such an emit previously produced a structurally valid **zero-row** MetricFrame that satisfied `assert_frame_envelope` and persisted to disk as a legitimate-looking audit artifact recording nothing. The message names the target and which schemas were present-but-empty. A **partial** report — at least one metric value in any schema — still emits normally.

**Loudness note:** this class is Level 0 and maintains no logger, with **one deliberate exception**: the `to_metric_frame()` emit path logs at `ERROR` before raising, because it belongs to Level 1 (it produces the persisted, cross-repo evaluation-of-record). Logging follows the emit path, not the file (logging standard §5.1). No other raise in this class logs.

---

## 7. Boundaries and Interactions

- **Upstream**: Produced by **NativeEvaluator**.
- **Downstream**: Consumed by Pipeline Core or reporting tools.

---

## 8. Examples of Correct Usage

```python
report = EvaluationReport(target="ged_sb_best", task="regression", pred_type="point", results=native_results)
df = report.to_dataframe(schema="month")       # pd.DataFrame
data = report.to_dict()                         # nested dict
schema = report.get_schema_results("month")     # dict → typed metrics dataclass
```

---

## 9. Examples of Incorrect Usage

- Calling `to_dataframe(schema='raw')` — this is deprecated and returns the internal dict, not a DataFrame. Use `to_dict()['schemas']` instead.
- Adding a new metric to `METRIC_CATALOG` without adding a corresponding field to the typed metrics dataclass — the FM1 guard will raise `ValueError`.
- Treating the report as mutable and modifying `_results` after construction.

---

## 10. Test Alignment

- **Green:** `tests/test_evaluation_report.py` — construction, schema access, to_dict, to_dataframe. `tests/test_metric_frame.py` — `to_metric_frame()` for all 4 cells, envelope conformance, save/load round-trip, provenance split.
- **Beige:** `tests/test_evaluation_report.py` — empty schemas, single-entry schemas. `tests/test_metric_frame.py` — empty/fully-empty schema, NaN values, run_id None at emit.
- **Red:** `tests/test_evaluation_report.py` — missing schema keys, field mismatch (FM1 guard). `tests/test_metric_frame.py` — fail-loud MetricFrame construction.

---

## 11. Evolution Notes

- The `to_dataframe()` method imports Pandas lazily. After Phase 3, this method may be removed or moved to an adapter.
- The `_metrics_map` mapping 4 (task, pred_type) combinations to dataclass types is stable but must be extended if new task types are added.

---

## 12. Known Deviations

- **Lazy Pandas import:** `to_dataframe()` imports `pandas` at call time, which means the Level 1 bridge concern leaks into what is otherwise a Level 0 component. This is a pragmatic compromise for backward compatibility.
- **Lazy views-frames import:** `to_metric_frame()` imports `views_frames` at call time (gated on `importlib.util.find_spec`), the same Level-1 bridge compromise as the Pandas case. The core stays installable without the `frames` extra (ADR-011 minimal core); the method fails loud if the extra is absent.
- **Legacy dataclass coupling:** `get_schema_results()` wraps results in legacy dataclass instances (`RegressionPointEvaluationMetrics`, etc.) from `metrics.py`. If a metric is computed but has no field in the dataclass, the FM1 guard raises. This means new metrics require coordinated updates to both `metric_catalog.py` and `metrics.py`.

---

## End of Contract

This document defines the **intended meaning** of `EvaluationReport`.

Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
