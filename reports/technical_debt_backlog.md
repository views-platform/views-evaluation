# Technical Debt / Refactoring Backlog for views-evaluation

This document summarizes identified fragile, un-standard, or non-best-practice elements within the `views-evaluation` library and its documentation, based on the comprehensive test suite conducted through Phases 1, 2, and 3 of the verification plan. These items represent areas for potential future improvement to enhance robustness, clarity, and adherence to best practices, especially for down-stream critical infrastructure use.

---

## 1. Documentation Inaccuracies & Ambiguities

### 1.1. Inaccurate Description of Point Prediction Handling

*   **Source:** `reports/eval_lib_imp.md` (Section 3.2, "Prediction Value Specification")
*   **Description:** The documentation inaccurately states that the `EvaluationManager` (EM) will fail if point predictions are provided as raw `float` or `int` values (non-canonical format).
*   **Actual Behavior:** The EM implicitly converts raw `float`/`int` predictions into a single-element `numpy.ndarray` (`[value]`) without raising an error.
*   **Impact:** Misleading documentation; developers might implement unnecessary reconciliation or incorrectly assume a stricter input contract.
*   **Recommendation:** Update `eval_lib_imp.md` (already done) to clearly state the implicit conversion. Consider if the EM *should* be stricter (e.g., raise a warning) or if this lenient behavior is acceptable.

### 1.2. Overstated "Mandatory" Reconciliation Step

*   **Source:** `reports/eval_lib_imp.md` (Section 3.2.1, "Recommended Reconciliation Step for Point Predictions")
*   **Description:** Originally described as "Mandatory", the reconciliation step for converting raw `float` point predictions to list format is not strictly necessary for the EM to run due to implicit conversion.
*   **Impact:** While the documentation has been updated to "Recommended", the initial emphasis on its "mandatory" nature highlights a past discrepancy between intended design and implementation behavior.
*   **Recommendation:** Reinforce the "Recommended" aspect (for consistency and alignment with uncertainty predictions) without implying a hard runtime requirement for the EM itself.

---

## 2. Lack of Robust Input Validation & Graceful Error Handling (Critical)

A major finding from Phase 2 (Adversarial Testing) is the EM's fragility when encountering corrupted or malformed input data. Instead of graceful failure (e.g., returning `NaN` metrics) or specific, caught exceptions, the library often crashes with unhandled exceptions originating from underlying numerical libraries (`numpy`, `sklearn`, `pandas`).

### 2.1. Unhandled Non-Finite Numerical Data

*   **Description:** The EM crashes with a `ValueError` (from `sklearn.utils.validation._assert_all_finite`) if `np.nan` or `np.inf` values are present in `actuals` or `predictions`.
*   **Impact:** A single non-finite value in production data can halt an entire evaluation pipeline. This is a severe fragility for critical infrastructure.
*   **Recommendation:** Implement explicit checks for non-finite values within the `EvaluationManager` or its metric calculators. Decision points:
    *   **Option A:** Raise a custom, informative `ValueError` before calling `sklearn` metrics.
    *   **Option B:** Filter out (or impute) non-finite values and calculate metrics on the remaining valid data, returning `NaN` for affected points/metrics, with appropriate warnings.

### 2.2. Unhandled Empty `predictions` List

*   **Status:** Resolved.
*   **Description:** Providing an empty list for `predictions` caused a `ValueError: No objects to concatenate` from `pandas.concat`.
*   **Fix:** Added explicit validation in `validate_predictions` to ensure the list is not empty and each DataFrame is valid.

### 2.2.1 Unhandled Multiple/Duplicate Prediction Columns

*   **Status:** Resolved.
*   **Description:** The library previously tolerated extra columns but crashed on duplicate `pred_{target}` names.
*   **Fix:** Hardened `validate_predictions` to strictly enforce the "Exactly One Column" contract, raising a clear `ValueError` if multiple columns are found.

### 2.3. Unhandled Empty `actuals` DataFrame

*   **Description:** Providing an empty `pandas.DataFrame` for `actuals` causes a `KeyError` when the manager tries to access the `target` column.
*   **Impact:** Unexpected input can crash the system.
*   **Recommendation:** Add explicit validation within `EvaluationManager` to check if the `actuals` DataFrame is empty before attempting to access columns.

### 2.4. Unhandled Non-Overlapping Indices

*   **Description:** If `actuals` and `predictions` have no common indices, the data matching process correctly produces empty internal DataFrames. However, these empty DataFrames are then passed to `np.concatenate` within metric calculators, resulting in a `ValueError: need at least one array to concatenate`.
*   **Impact:** This scenario, common in rolling evaluations if data gaps occur, leads to a hard crash rather than a graceful `NaN` metric.
*   **Recommendation:** Implement checks in metric calculators (or the `_match_actual_pred` function) to handle cases where `matched_actual` or `matched_pred` are empty after index matching, returning `np.nan` for affected metrics.

---

## 3. General Best Practice Adherence

### 3.1. Extensive Reliance on External Libraries for Core Metric Calculations

*   **Source:** `views_evaluation/evaluation/metric_calculators.py`
*   **Description:** Many core metrics leverage `sklearn` functions. While efficient, this implicitly inherits their input validation behaviors and error messages.
*   **Impact:** As seen in Phase 2, `sklearn`'s `ValueError` messages can be generic and not specific to the `views-evaluation` context, making debugging harder for users.
*   **Recommendation:** Consider wrapping external metric calls with custom error handling to provide more user-friendly and context-specific error messages, or pre-validate inputs to `sklearn` functions to prevent their general `ValueError`s.

### 3.2. Implicit Data Transformations in `convert_to_array`

*   **Source:** `views_evaluation/evaluation/evaluation_manager.py` (`convert_to_array` method)
*   **Description:** The `convert_to_array` method implicitly converts raw `float`/`int` values to `np.array([value])`.
*   **Impact:** This is the underlying mechanism that makes the EM more lenient than its documentation initially claimed. While it makes the library robust to `stepshifter`-like inputs, it performs a transformation that might be unexpected if not clearly documented, potentially hiding non-canonical data.
*   **Recommendation:** This behavior is now documented. However, a decision should be made if this implicit conversion should be accompanied by a `logging.warning` to alert users when non-canonical data is being transformed.

---

## 4. Unimplemented Metrics (Future Work)

*   **Source:** `reports/eval_lib_imp.md` (Section 4.1), `views_evaluation/evaluation/metric_catalog.py`
*   **Description:** Four metrics are registered in `METRIC_CATALOG` with `implemented=False` and raise `ValueError` — not `NotImplementedError` — when requested: `SD`, `pEMDiv`, `Variogram`, `Jeffreys`. The `ValueError` choice is deliberate; see the comment at `native_metric_calculators.py`'s unimplemented-metric block.
*   **Corrected 2026-08-02:** this entry previously listed `Brier` as unimplemented and named `NotImplementedError` as the exception. Both were wrong. Three Brier variants (`Brier_cls_point`, `Brier_cls_sample`, `Brier_rgs_sample`) shipped 2026-04-09 and carry no `implemented=False` flag. The stale source path `metric_calculators.py` names a module deleted on 2026-02-26 (commit `a256e72`) — five weeks before the Phase 3 purge (`fe6dd4c`, 2026-04-01) that the rest of this repo means by "Phase 3". Same drift as ADR-031 carried until this release — see register C-34.
*   **Impact:** Limits the comprehensiveness of the evaluation framework.
*   **Recommendation:** Prioritize the implementation of these metrics based on user needs, ensuring each implementation is accompanied by rigorous "Golden Dataset" tests (Phase 3).

---

## 5. Demoted from the Risk Register (2026-06-26 strategic review)

Tier-4 items with no correctness/reliability dimension — moved here from `reports/technical_risk_register.md` so the register carries signal, not cleanup tickets.

### 5.1. Stale docstring references deleted `EvaluationManager` validator (was C-21)

*   **Source:** `views_evaluation/evaluation/config_schema.py:17`
*   **Description:** The docstring states "Downstream validators (`EvaluationManager._validate_config`) enforce required-key semantics at runtime." `EvaluationManager` was removed in Phase 3, so no such validator exists; config validation is genuinely unowned (the functional gap is tracked as risk **C-02**).
*   **Impact:** A reader following the docstring to find runtime config validation hits a dead reference.
*   **Recommendation:** Update the docstring to state that no runtime config validator currently exists. (The real risk — unowned validation — stays in the register as C-02.)

### 5.2. Vestigial empty `adapters` package (was C-23)

*   **Source:** `views_evaluation/adapters/__init__.py` (empty, 0 lines)
*   **Description:** Leftover of the removed Pandas-adapter design (`PandasAdapter`/`adapters.pandas`). Nothing imports it; dead structure that implies an adapter layer that no longer exists.
*   **Impact:** Misleads readers auditing the Phase-3/4 migration for completeness.
*   **Recommendation:** Remove the empty package, or add a module docstring stating the adapter layer was intentionally removed.
