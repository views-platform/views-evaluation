# Class Intent Contract: EvaluationReport

**Status:** Active  
**Owner:** Evaluation Core  
**Last reviewed:** 2026-03-13  
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
- **Downstream**: Consumed by **EvaluationManager**, Pipeline Core, or reporting tools.

---

## 8. Examples of Correct Usage

```python
report = EvaluationReport(target="ged_sb_best", task="regression", pred_type="point", results=native_results)
df = report.to_dataframe(schema="month")       # pd.DataFrame
data = report.to_dict()                         # nested dict
schema = report.get_schema_results("month")     # dict → typed metrics dataclass
```
