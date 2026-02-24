# Documentation Discrepancy Report

**Date:** 2026-01-27

## 1. Executive Summary

This report details the findings of a programmatic analysis comparing the `views-evaluation` codebase against its documentation, primarily the Architectural Decision Records (ADRs) and the `offline_chap_draft.md` report.

The analysis concludes that while the core evaluation logic is implemented **correctly** according to its documentation, there is a **significant discrepancy** in the implementation status of documented evaluation metrics. Several metrics defined in `ADR-001` are not implemented in the codebase. Additionally, the `offline_chap_draft.md` report contains internal inconsistencies regarding the standard forecast horizon.

## 2. Verification Method

A dedicated test suite (`tests/test_documentation_adherence.py`) was created to programmatically verify two key areas:
1.  **Metric Implementation Status:** The test checks every metric listed in `ADR-001` and verifies if it is implemented in `views_evaluation/evaluation/metric_calculators.py`.
2.  **Evaluation Schema Logic:** The test confirms that the `EvaluationManager` groups data for its `step-wise`, `time-series-wise`, and `month-wise` schemas exactly as depicted in the diagrams in `offline_chap_draft.md`.

## 3. Findings

### 3.1. Finding: Core Logic is Consistent with Documentation (✅)

The programmatic tests **passed**, confirming that the implementation of the three evaluation schemas in `EvaluationManager` is **correct** and consistent with the architectural diagrams and descriptions.

-   **`step-wise` evaluation:** Correctly groups data by forecast step (diagonals).
-   **`time-series-wise` evaluation:** Correctly groups data by forecast sequence (columns).
-   **`month-wise` evaluation:** Correctly groups data by calendar month (rows).

**Conclusion:** The fundamental evaluation logic is sound and well-documented.

---

### 3.2. Discrepancy: Metric Implementation Gap (❌)

The programmatic tests **failed** when checking for full metric implementation, revealing a gap between `ADR-001` and the codebase.

The following metrics are documented in `ADR-001` but are **not implemented** (i.e., they raise `NotImplementedError`):

| Metric Type | Metric Name | Status                      |
|-------------|-------------|-----------------------------|
| Point       | `SD`        | Defined but Not Implemented |
| Point       | `Variogram` | Defined but Not Implemented |
| Point       | `pEMDiv`    | Defined but Not Implemented |
| Uncertainty | `Brier`     | Defined but Not Implemented |
| Uncertainty | `Jeffreys`  | Defined but Not Implemented |
| Uncertainty | `pEMDiv`    | Defined but Not Implemented |

**Conclusion:** `ADR-001` is outdated and does not reflect the current implementation state. This is also noted in the "Next Steps" section of `offline_chap_draft.md`.

---

### 3.3. Discrepancy: Inconsistent Forecast Horizon in Documentation (❌)

A manual review of `offline_chap_draft.md` reveals conflicting information regarding the forecast horizon.

-   The text mentions a **`48-month`** forward prediction window in one section.
-   In several other sections, and consistent with the ADRs, it refers to a **`36-month`** forecast sequence.

**Conclusion:** The `offline_chap_draft.md` report is inconsistent and needs to be clarified. The ADRs and current implementation practices point towards **36 months** as the standard.

## 4. Recommendations

1.  **Create a Ticket to Update Documentation:**
    -   **Task:** Update `ADR-001` to clearly mark the metrics that are not yet implemented. Use a status like "Proposed" or "Not Implemented" in the metric table.
    -   **Task:** Review and correct the `offline_chap_draft.md` to consistently state the forecast horizon (likely 36 months).
    -   **Justification:** Ensures documentation accurately reflects the state of the code, preventing confusion for current and future developers.

2.  **Create a Ticket for Metric Implementation:**
    -   **Task:** Create a feature/technical debt ticket to implement the missing metrics (`SD`, `Variogram`, `pEMDiv`, `Brier`, `Jeffreys`).
    -   **Justification:** Fulfills the original architectural vision outlined in `ADR-001`. This task is already noted in the "Next Steps" of the draft report, but a formal ticket will make it trackable.

No bugs were found in the core evaluation logic of the code itself. The discovered issues are confined to documentation and incomplete features.
