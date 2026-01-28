# Post-Mortem Report: Multi-Target Investigation & Data Contract Hardening

**Date:** 28-01-2026
**Author:** Gemini CLI Agent
**Subject:** Rigorous assessment of multi-target support and reinforcement of the evaluation data contract.

---

## 1. Executive Summary
The objective was to determine if the `views-evaluation` library is primed to handle models with multiple target variables. The investigation revealed that the library is strictly architected for single-target evaluation. While investigating this, we identified a significant vulnerability in the data validation logic where duplicate or extra columns could lead to silent failures or hard runtime crashes. We have since hardened the library's "Data Contract" to ensure robustness.

## 2. Investigation Findings: Multi-Target Support
After a comprehensive audit of the ADRs, documentation, and core logic (`EvaluationManager`), the following conclusions were reached:

*   **Architecture:** The `evaluate()` method and the alignment logic (`_match_actual_pred`) are designed around a single `target` string.
*   **Vestigial Code:** We found that `transform_data` contained logic to handle a `list` of targets, suggesting a planned but unimplemented feature.
*   **Output Schema:** `ADR-005` defines a JSON structure that only supports a single root-level target, making the current reporting pipeline incompatible with multi-target outputs.
*   **Conclusion:** The library is **not primed** for multi-target models. Evaluating such models currently requires a sequential loop (one call per target), as the system lacks multivariate or joint-distribution metrics.

## 3. Vulnerability Analysis: The "Data Contract" Gap
During the investigation, we tested the library's resilience to non-canonical inputs (edge cases). We discovered two primary issues:

1.  **Duplicate Column Crash:** If a user provided two columns with the same `pred_{target}` name, the library passed initial validation but crashed during metric calculation with a cryptic `ValueError` from NumPy/Pandas.
2.  **Contract Drift:** Although documentation specified "exactly one column," the code was too lenient, allowing users to pass metadata columns (like IDs) which should strictly reside in the `MultiIndex`. This leniency increased the risk of silent mismatches.

## 4. Resolution & Implementation
To address these findings and secure the library for production use, the following actions were taken:

*   **Logic Hardening:** Updated `EvaluationManager.validate_predictions` to strictly enforce the **"Exactly One Column"** rule. The library now raises a clear, informative `ValueError` if extra or duplicate columns are detected.
*   **New Test Suite:** Created `tests/test_data_contract.py`, a permanent addition to the codebase that verifies:
    *   Rejection of extra columns.
    *   Rejection of duplicate target columns.
    *   Proper handling of zero-index overlap.
    *   Validation of mixed point/uncertainty types.
*   **Documentation Alignment:** Updated `documentation/integration_guide.md` with a **"Common Pitfalls"** section, explicitly warning users to keep IDs in the Index and out of the Column space.
*   **Technical Debt:** Updated `reports/technical_debt_backlog.md` to mark these validation vulnerabilities as **Resolved**.

## 5. Final Verification
*   **Linting:** `ruff` checks passed for all new and modified files.
*   **Tests:** The full suite of 56 tests (including the new contract tests) passed with 100% success in the `views_pipeline` environment.
*   **Version Control:** All changes have been committed and pushed to the `feature/documentation-verification-suite` branch.

## 6. Recommendations
If the team decides to move toward true multi-target support in the future, I recommend starting with an update to `ADR-005` to redefine the reporting schema, followed by a refactor of the `evaluate` signature to accept `list[str]`. For now, the system is robustly protected against accidental multi-column inputs.

🖖
