# Class Intent Contract: EvaluationReport

**Status:** Active  
**Owner:** Evaluation Core  
**Last reviewed:** 2026-02-25  
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
- **Traceability**: Maintains metadata about the target and model run.
- **Representation Agnosticism**: Provides a standard internal representation that can be converted to external formats (JSON, Dictionary, or Pandas).

---

## 4. Inputs and Assumptions

- **Pre-computed Results**: Assumes results have already been computed by an evaluator.
- **Schema Conformity**: Expects data organized by the standard Views schemas.

---

## 5. Outputs and Side Effects

- **DataFrames**: Can generate Pandas DataFrames for backward compatibility.
- **JSON/Dict**: Can generate machine-readable structures for persistence (ADR-041).

---

## 6. Failure Modes and Loudness

- Raises `KeyError` if a requested schema or metric is missing.
- Fails loud if the input result structure is inconsistent.

---

## 7. Boundaries and Interactions

- **Upstream**: Produced by **NativeEvaluator**.
- **Downstream**: Consumed by **EvaluationManager**, Pipeline Core, or reporting tools.

---

## 8. Examples of Correct Usage

```python
report = EvaluationReport(target="target_name", results=native_results)
df = report.to_dataframe(schema="month")
json_data = report.to_json()
```
