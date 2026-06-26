# Phase 3 Test Coverage Audit

**Date:** 2026-03-13
**Purpose:** Map PHASE-3-DELETE tests to native equivalents before orchestrator migration.

---

## Summary

| Category | Count |
|----------|-------|
| PHASE-3-DELETE tests total | 77 |
| Direct native equivalents exist | 28 |
| Partial/implicit coverage | 15 |
| Adapter-level (not needed after removal) | 24 |
| Legacy config compat (not needed) | 10 |

**Conclusion:** No critical behavior gaps. The 28 directly-equivalent tests prove that native coverage already exists. The 24 adapter-level tests validate PandasAdapter behavior that moves to views-pipeline-core with the adapter.

---

## File-by-File Mapping

### test_evaluation_manager.py (34 tests) -- 100% PHASE-3-DELETE

| Legacy Test | Behavior | Native Equivalent | Gap? |
|---|---|---|---|
| test_validate_dataframes_valid_type | Rejects non-DataFrame input | PandasAdapter concern | No |
| test_validate_dataframes_valid_columns | Rejects missing columns | PandasAdapter concern | No |
| test_get_evaluation_type | Detects point vs sample | EvaluationFrame.is_sample | No |
| test_match_actual_pred_point | Index alignment | PandasAdapter concern | No |
| test_split_dfs_by_step | Step grouping | test_native_evaluator::test_step_filtering_* | No |
| test_step_wise_evaluation_point | Step schema + RMSLE | test_native_evaluator::test_step_wise_* | No |
| test_step_wise_evaluation_sample | Step schema + CRPS | test_native_evaluator::test_report_has_three_schemas | No |
| test_time_series_wise_evaluation_point | TS schema + RMSLE | test_native_evaluator::test_time_series_wise_* | No |
| test_time_series_wise_evaluation_sample | TS schema + CRPS | Same | No |
| test_month_wise_evaluation_point | Month schema + RMSLE | test_native_evaluator::test_month_wise_* | No |
| test_month_wise_evaluation_sample | Month schema + CRPS | Same | No |
| test_calculate_ap_* | AP metric calculation | test_metric_correctness::test_ap_* | No |
| test_normalise_config_* | Legacy key translation | Not needed (NativeEvaluator uses modern keys) | No |
| test_validate_config_* | Config validation | test_native_evaluator::TestNativeEvaluatorRed | No |
| test_evaluate_target_not_in_config | Unknown target error | test_native_evaluator::test_target_not_in_config_raises | No |
| test_evaluate_invalid_metric | Wrong metric for task | test_native_evaluator::test_classification_metric_on_regression | No |

### test_evaluation_schemas.py (3 tests) -- 100% PHASE-3-DELETE

| Legacy Test | Native Equivalent |
|---|---|
| test_step_wise_schema_grouping | test_native_evaluator::test_step_wise_configured_steps_are_populated |
| test_time_series_wise_schema_grouping | test_native_evaluator::test_time_series_wise_group_count |
| test_month_wise_schema_grouping | test_native_evaluator::test_month_wise_group_count |

All three have direct native equivalents.

### test_adversarial_inputs.py (15 tests) -- PARTIAL

**TestAdversarialInputs (7 tests) -- PHASE-3-DELETE:**

| Legacy Test | Native Equivalent |
|---|---|
| test_corrupted_numerical_data_nan_in_actuals | TestAdversarialNativeInputs::test_nan_in_y_true_* |
| test_corrupted_numerical_data_nan_in_predictions | TestAdversarialNativeInputs::test_nan_in_y_pred_* |
| test_corrupted_numerical_data_inf_in_actuals | TestAdversarialNativeInputs::test_inf_in_y_true_* |
| test_corrupted_numerical_data_inf_in_predictions | TestAdversarialNativeInputs::test_inf_in_y_pred_* |
| test_malformed_structural_data_empty_predictions | PandasAdapter concern |
| test_malformed_structural_data_empty_actuals | PandasAdapter concern |
| test_malformed_structural_data_non_overlapping | PandasAdapter concern |

**TestAdversarialNativeInputs (8 tests) -- RETAINED.** Already uses EvaluationFrame + NativeEvaluator.

### test_parity_adapter_transfer.py (2 tests) -- 100% PHASE-3-DELETE

Both test parity between internal/external adaptation and shadow verification mode. Not needed once EvaluationManager is removed.

### test_parity_green.py (2 tests) -- PHASE-3-DELETE

Parity tests between legacy and native paths. Not needed once legacy is removed. Correctness covered by test_native_evaluator happy-path tests.

### test_parity_red.py (4 tests) -- PHASE-3-DELETE

| Legacy Test | Native Equivalent |
|---|---|
| test_parity_red_unordered | PandasAdapter concern (reordering) |
| test_parity_red_coordinates | PandasAdapter concern (filtering) |
| test_fail_loud_inconsistent_samples | test_evaluation_frame::test_y_pred_row_mismatch_raises |
| test_fail_loud_nan_index | test_evaluation_frame::test_nan_in_float_identifier_raises |

### test_parity_beige.py (1 test) -- PHASE-3-DELETE

`test_parity_beige_ragged` — ragged sequences. Partially covered by test_native_evaluator::test_legacy_compatibility_* tests. **Minor gap:** no native beige test for incomplete parallelograms.

### test_documentation_contracts.py (11 tests) -- PHASE-3-DELETE

All test EvaluationManager API contracts (prefix validation, implicit conversion, config keys). Not needed after EvaluationManager removal.

### test_data_contract.py (6 tests) -- PHASE-3-DELETE

All test PandasAdapter input validation (missing columns, duplicates, type mixing). Should travel with PandasAdapter to views-pipeline-core.

### conftest.py -- PHASE-3-DELETE

`mock_data_factory` fixture used only by legacy tests. Not needed by native tests.

---

## Phase 3 Decision Points

### 1. legacy_compatibility flag (native_evaluator.py:75)

Current: `evaluate(ef, legacy_compatibility=True)` — truncates step results to shortest sequence.

**Recommendation:** Keep the flag. Flip default to `False` in Phase 3. Emit `DeprecationWarning` for one release cycle when `legacy_compatibility=True` is explicitly passed.

**Authoritative tests:**
- `test_native_evaluator::test_legacy_compatibility_true_truncates_to_shortest_sequence`
- `test_native_evaluator::test_legacy_compatibility_false_includes_all_steps_with_data`

### 2. PandasAdapter destination

Move `adapters/pandas.py` + `test_data_contract.py` to views-pipeline-core. The adapter is the caller's responsibility per ADR-011.

### 3. EvaluationManager deletion checklist

1. Port any remaining gap tests (see below)
2. Delete: evaluation_manager.py, adapters/pandas.py, deprecation_msgs.py
3. Delete: conftest.py, test_evaluation_manager.py, test_evaluation_schemas.py, test_parity_*.py, test_documentation_contracts.py, test_data_contract.py
4. Remove PHASE-3-DELETE exports from `__init__.py`
5. Remove pandas from pyproject.toml dependencies

### 4. Remaining test gaps to port before deletion

- **Ragged parallelogram beige test:** Add native beige test with sequences of different lengths and verify step-wise results handle missing diagonals correctly.
- **Metric correctness expansion:** Current golden-value tests (test_metric_correctness.py) only cover 5 metrics. Expand to cover MCR, QIS, MIS, twCRPS with hand-computed reference values.
