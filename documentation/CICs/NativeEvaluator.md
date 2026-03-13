# Class Intent Contract: NativeEvaluator

**Status:** Active  
**Owner:** Evaluation Core  
**Last reviewed:** 2026-03-13
**Related ADRs:** ADR-010 (Ontology), ADR-011 (Topology), ADR-032 (Schemas), ADR-042 (Metric Catalog)

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
- **Legacy Compatibility**: Provides an explicit `legacy_compatibility` flag (default `True`) that caps step-wise evaluation to the shortest sequence in the frame, reproducing the historic zip-truncation behaviour required for parity with the legacy system. Set `False` to evaluate all steps with available data.
- **Exact Step Filtering**: Evaluates only the step positions explicitly declared in `config['steps']`. Sparse configs (e.g. `[1, 3, 6, 12]`) produce exactly four step keys, not one key per step up to the maximum.
- **Fail-Loud Dispatch**: Guarantees that it fails immediately if a requested metric or configuration is invalid for the provided data.

---

## 4. Inputs and Assumptions

- **EvaluationFrame**: Assumes the frame is valid and internally consistent.
- **Configuration** (`EvaluationConfig` TypedDict): Requires a dict with:
  - `steps` (List[int]): Exact step positions to evaluate (1-indexed)
  - `regression_targets` / `classification_targets` (List[str]): Target name assignment
  - `regression_point_metrics`, `regression_sample_metrics`, etc. (List[str]): Metrics to compute
  - `evaluation_profile` (str, default `"base"`): Named profile for metric hyperparameters (ADR-042)
  - `metric_hyperparameters` (Dict[str, Dict[str, Any]]): Per-metric overrides, takes precedence over profile
- **Identifier Presence**: Assumes `time`, `origin`, and `step` identifiers exist in the frame.

---

## 5. Outputs and Side Effects

- **Evaluation Results**: Produces a nested dictionary of results, mapped to the legacy dataclass structures.
- **Traceability**: Ensures every result can be traced back to its underlying data slice.

---

## 6. Failure Modes and Loudness

- Raises `ValueError` if the target name in metadata is not declared in the config.
- Raises `ValueError` if a requested metric name is not valid for the task type, or is defined but not yet implemented.
- Fails loud if the `EvaluationFrame` lacks the required identifiers for a schema.

---

## 7. Boundaries and Interactions

- **Upstream**: Called directly or via legacy `EvaluationManager` (PHASE-3-DELETE).
- **Internal**: Depends on `EvaluationFrame` and `MetricCalculators`.
- **Isolation**: Must not depend on any IO or dataframe frameworks.

---

## 8. Examples of Correct Usage

```python
evaluator = NativeEvaluator(config)
report = evaluator.evaluate(ef, legacy_compatibility=True)  # returns EvaluationReport

# Access results
month_df = report.to_dataframe('month')        # pd.DataFrame indexed by group keys
step_dict = report.to_dict()['schemas']['step']  # raw nested dict
schema = report.get_schema_results('time_series')  # dict → typed metrics dataclass
```
