# Class Intent Contract: NativeEvaluator

**Status:** Active  
**Owner:** Evaluation Core  
**Last reviewed:** 2026-08-02  
**Related ADRs:** ADR-010 (Ontology), ADR-011 (Topology), ADR-015 (Degenerate/Empty Results), ADR-032 (Schemas), ADR-042 (Metric Catalog)

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
- **Config Validation at Construction**: Guarantees that a structurally invalid config fails at `__init__`, not at `evaluate()`. Rejects unknown/misspelled keys (validated against `EvaluationConfig`), a missing or empty `steps` list, non-positive step positions, absence of any target list, a declared task with no metrics, and metric names that are unknown or invalid for the cell their key declares. Nothing is defaulted or repaired (ADR-015).
- **Legacy Compatibility**: Provides an explicit `legacy_compatibility` flag (**default `False`**) that caps step-wise evaluation to the shortest sequence in the frame, reproducing the historic zip-truncation behaviour required for parity with the legacy system. If truncation would drop a step that `config['steps']` explicitly requested, it **raises** rather than returning an empty placeholder for it (ADR-015 ruling 7).
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
- **Identifier Presence**: Assumes `time`, `unit`, `origin`, and `step` identifiers exist in the frame — all four are enforced by `EvaluationFrame._validate`. (`unit` is required for construction but is not itself a grouping axis.)

---

## 5. Outputs and Side Effects

- **Evaluation Results**: Produces a nested dictionary of results, mapped to the legacy dataclass structures.
- **Traceability**: Ensures every result can be traced back to its underlying data slice.

---

## 6. Failure Modes and Loudness

At construction (`__init__`) — all `ValueError` (ADR-015 rulings 4/5, risk register C-02):

- Config is not a dict.
- Unknown evaluation profile name.
- Unknown or misspelled config key, including any legacy key removed in 0.4.0.
- `steps` missing or empty; or containing a non-integer or non-positive entry.
- No target list declared.
- A declared task with no metrics for it.
- A metric name absent from `METRIC_CATALOG`, or not valid for the `(task, pred_type)` cell its key declares.

At evaluation (`evaluate`) — all `ValueError`:

- The target in frame metadata is not declared in the config.
- No metrics configured for the resolved `(task, pred_type)`. This can only be detected here, because `pred_type` is a property of the frame (`n_samples > 1`), not of the config.
- A requested metric is defined but not yet implemented.
- `legacy_compatibility=True` would truncate away a step that `config['steps']` explicitly requested.
- A metric function raises — wrapped and re-raised naming the metric, task and pred_type (C-16).
- The `EvaluationFrame` lacks the required identifiers for a schema.

**Loudness note:** this class is Level 0 and maintains **no logger** — exceptions propagate to the orchestrator (logging standard §5.1). That is deliberate; adding a logger here is a violation.

---

## 7. Boundaries and Interactions

- **Upstream**: Called directly by evaluation orchestrators (e.g. views-pipeline-core).
- **Internal**: Depends on `EvaluationFrame` and `MetricCalculators`.
- **Isolation**: Must not depend on any IO or dataframe frameworks.

---

## 8. Examples of Correct Usage

```python
evaluator = NativeEvaluator(config)          # raises now if config is structurally invalid
report = evaluator.evaluate(ef)              # legacy_compatibility defaults to False

# Access results
month_df = report.to_dataframe('month')        # pd.DataFrame indexed by group keys
step_dict = report.to_dict()['schemas']['step']  # raw nested dict
schema = report.get_schema_results('time_series')  # dict → typed metrics dataclass
```

---

## 9. Examples of Incorrect Usage

- Passing a raw dict instead of an `EvaluationFrame` — the evaluator expects validated frames, not ad-hoc data.
- Requesting metrics that are not valid for the (task, pred_type) combination — e.g. asking for `CRPS` on a point prediction. This will fail loud.
- Omitting `evaluation_profile` from config and expecting hardcoded defaults — the resolver requires explicit profile selection.
- Using `legacy_compatibility=False` without understanding that step-wise results will include steps not present in all origins.
- Using `legacy_compatibility=True` while requesting more steps in `config['steps']` than the shortest origin sequence supplies — this raises. Request only the steps that should be scored.
- Relying on a misspelled or legacy config key being ignored — every key is now validated at construction.

---

## 10. Test Alignment

- **Green:** `tests/test_native_evaluator.py` — three-schema evaluation, legacy compat, metric dispatch.
- **Beige:** `tests/test_native_evaluator.py` — sparse step configs, single-origin frames.
- **Red:** `tests/test_native_evaluator.py`, `tests/test_adversarial_inputs.py` — undeclared targets, unimplemented metrics.
- **Integration:** `tests/test_adversarial_inputs.py` — undeclared targets, unimplemented metrics, NaN/Inf defense-in-depth.

---

## 11. Evolution Notes

- `legacy_compatibility` default was flipped to `False` in Phase 3. The flag is retained for callers that need truncation behavior. (§3 of this contract incorrectly stated the default was `True` until 2026-08-02.)
- Config validation added to `__init__` (2026-08-02, C-02): structural config errors now surface at construction rather than at evaluation. The valid key set is derived from `EvaluationConfig.__annotations__`, giving that previously type-only module a runtime authority. Tests: `TestNativeEvaluatorRed` (9 cases).
- Empty-metric-list guard added to `_resolve_task_and_metrics` (2026-08-02, C-02): a resolved `(task, pred_type)` with no configured metrics now raises instead of returning empty per-group dicts.
- `legacy_compatibility` truncation made loud (2026-08-02, C-20): requesting a step that truncation would drop now raises. Previously such steps were returned as empty placeholder dicts.
- The `EvaluationReport` return type is stable; the internal `_calculate_metrics` dispatch may evolve as the `MetricCatalog` grows.
- Exception wrapping added to `_calculate_metrics()` (2026-04-04, C-16): metric function exceptions are now caught and re-raised as `ValueError` naming the metric, task, and pred_type. Test: `test_metric_function_error_includes_metric_name`.
- Step sentinel changed from hardcoded `999` to `float('inf')` (2026-04-04, C-17): steps >= 1000 are no longer silently dropped. Test: `test_step_values_above_999_not_silently_dropped`.

---

## 12. Known Deviations

- **sklearn/scipy in "pure core":** The `NativeEvaluator` dispatches to metric functions that import `sklearn` and `scipy` at module level. This contradicts the stated goal of a zero-external-dep Level 0 core (ADR-011). (Risk register C-05)

---

## End of Contract

This document defines the **intended meaning** of `NativeEvaluator`.

Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
