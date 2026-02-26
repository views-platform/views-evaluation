# Class Intent Contract: NativeEvaluator

**Status:** Active  
**Owner:** Evaluation Core  
**Last reviewed:** 2026-02-25  
**Related ADRs:** ADR-010 (Ontology), ADR-011 (Topology), ADR-032 (Schemas)

---

## 1. Purpose

A stateless "Pure Math" engine that executes the three standard Views evaluation schemas (Month, Sequence, Step) by operating exclusively on `EvaluationFrame` instances.

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** know about DataFrames or CSV files.
- This class does **not** perform inverse transformations or scaling.
- This class does **not** define new metrics (it dispatches to MetricCalculators).
- This class does **not** handle report generation or persistence.

---

## 3. Responsibilities and Guarantees

- **Schema Preservation**: Guarantees that month-wise, sequence-wise, and step-wise regrouping logic is consistent with established Views standards (ADR-032).
- **Stateless Execution**: Operates as a pure function of (Configuration + EvaluationFrame).
- **Legacy Compatibility**: Provides an explicit flag to reproduce legacy truncation bugs when needed for parity.
- **Fail-Loud Dispatch**: Guarantees that it fails immediately if a requested metric or configuration is invalid for the provided data.

---

## 4. Inputs and Assumptions

- **EvaluationFrame**: Assumes the frame is valid and internally consistent.
- **Configuration**: Requires a valid config dictionary declaring task types and target names.
- **Identifier Presence**: Assumes `time`, `origin`, and `step` identifiers exist in the frame.

---

## 5. Outputs and Side Effects

- **Evaluation Results**: Produces a nested dictionary of results, mapped to the legacy dataclass structures.
- **Traceability**: Ensures every result can be traced back to its underlying data slice.

---

## 6. Failure Modes and Loudness

- Raises `ValueError` if the target name in metadata is not declared in the config.
- Raises `KeyError` if a requested metric is not implement for the task type.
- Fails loud if the `EvaluationFrame` lacks the required identifiers for a schema.

---

## 7. Boundaries and Interactions

- **Upstream**: Orchestrated by `EvaluationManager`.
- **Internal**: Depends on `EvaluationFrame` and `MetricCalculators`.
- **Isolation**: Must not depend on any IO or dataframe frameworks.

---

## 8. Examples of Correct Usage

```python
evaluator = NativeEvaluator(config)
results = evaluator.evaluate(ef, legacy_compatibility=True)
month_df = results['month'][1]
```
