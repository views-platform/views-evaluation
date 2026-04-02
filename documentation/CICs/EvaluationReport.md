# Class Intent Contract: EvaluationReport

**Status:** Active  
**Owner:** Evaluation Core  
**Last reviewed:** 2026-04-02  
**Related ADRs:** ADR-010 (Ontology), ADR-041 (Output Schema)

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

---

## 6. Failure Modes and Loudness

- Raises `KeyError` if a requested schema is not found in the report.
- Raises `ValueError` if a computed metric name has no corresponding field in the typed metrics dataclass (FM1 guard against silent metric loss).
- Fails loud if the input result structure is inconsistent.

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

- **Green:** `tests/test_evaluation_report.py` — construction, schema access, to_dict, to_dataframe.
- **Beige:** `tests/test_evaluation_report.py` — empty schemas, single-entry schemas.
- **Red:** `tests/test_evaluation_report.py` — missing schema keys, field mismatch (FM1 guard).

---

## 11. Evolution Notes

- The `to_dataframe()` method imports Pandas lazily. After Phase 3, this method may be removed or moved to an adapter.
- The `_metrics_map` mapping 4 (task, pred_type) combinations to dataclass types is stable but must be extended if new task types are added.

---

## 12. Known Deviations

- **Lazy Pandas import:** `to_dataframe()` imports `pandas` at call time, which means the Level 1 bridge concern leaks into what is otherwise a Level 0 component. This is a pragmatic compromise for backward compatibility.
- **Legacy dataclass coupling:** `get_schema_results()` wraps results in legacy dataclass instances (`RegressionPointEvaluationMetrics`, etc.) from `metrics.py`. If a metric is computed but has no field in the dataclass, the FM1 guard raises. This means new metrics require coordinated updates to both `metric_catalog.py` and `metrics.py`.

---

## End of Contract

This document defines the **intended meaning** of `EvaluationReport`.

Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
