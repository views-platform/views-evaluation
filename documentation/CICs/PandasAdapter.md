# Class Intent Contract: PandasAdapter

**Status:** Deprecated (PHASE-3-DELETE)
**Owner:** Adapters Layer
**Last reviewed:** 2026-03-13  
**Related ADRs:** ADR-010 (Ontology), ADR-011 (Topology), ADR-012 (Authority), ADR-040 (Input Schema)

---

## 1. Purpose

A framework-specific bridge that transforms Pandas DataFrames into the canonical `EvaluationFrame`. It encapsulates all the "dirty" logic of alignment, reindexing, and list-extraction.

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** calculate metrics.
- This class does **not** persist data.
- This class does **not** handle other frameworks (like Polars).
- This class does **not** own the authoritative math core.

---

## 3. Responsibilities and Guarantees

- **MultiIndex Translation**: Guarantees that Pandas index levels (time, unit) are correctly mapped to `EvaluationFrame` identifiers.
- **Alignment (Truth Duplication)**: Responsible for performing the intersection of indices and duplicating `actuals` to match the sequence-based structure of `predictions`.
- **Sample Extraction**: Guarantees that "lists-in-cells" are correctly exploded into dense 2D NumPy arrays.
- **Metadata Declaration**: Responsible for explicitly declaring task and prediction types (as per ADR-012).

---

## 4. Inputs and Assumptions

- **Pandas Objects**: Expects `pd.DataFrame` and `List[pd.DataFrame]`.
- **Naming Conventions**: Assumes `month_id` and `entity_id` structure in MultiIndex.
- **Rectangular Samples**: Assumes that all prediction cells in a given task contain the same number of samples (or scalars).

---

## 5. Outputs and Side Effects

- **EvaluationFrame**: Produces a single, pre-aligned, flattened `EvaluationFrame`.

---

## 6. Failure Modes and Loudness

- Silently skips prediction DataFrames whose index has no overlap with actuals (continues to the next sequence).
- Raises `ValueError` if sample lengths are inconsistent across cells.
- Fails loud if the input is not a DataFrame.

---

## 7. Boundaries and Interactions

- **Upstream**: Called by users or legacy `EvaluationManager` (PHASE-3-DELETE).
- **Downstream**: Produces input for `EvaluationFrame`.
- **Isolation**: This is one of the few places where a `pandas` import is allowed.
- **Deprecation**: Emits `DeprecationWarning` on use. Will be removed from this repo in Phase 3; adapters belong in the calling repository (e.g. `views-pipeline-core`).

---

## 8. Examples of Correct Usage

```python
ef = PandasAdapter.from_dataframes(actual_df, [pred_df1, pred_df2], "target_name")
```

---

## 9. Examples of Incorrect Usage

- Passing a single DataFrame instead of a list — the adapter expects `List[pd.DataFrame]`.
- Passing predictions with columns not named `pred_{target}` — will raise `KeyError`.
- Assuming `step` corresponds to calendar months — step is positional lead-time, not absolute time.

---

## 10. Test Alignment

- **Parity Green:** `tests/test_parity_green.py` — happy-path round-trip through adapter + evaluator.
- **Parity Beige:** `tests/test_parity_beige.py` — ragged sequences.
- **Parity Red:** `tests/test_parity_red.py` — coordinate mismatches, NaN indices, inconsistent samples.
- **Adapter Transfer:** `tests/test_parity_adapter_transfer.py` — shadow verification mode, corruption detection.

---

## 11. Evolution Notes

- This class is marked PHASE-3-DELETE. It will be removed from this repository when the adapter responsibility moves to `views-pipeline-core`.
- All tests tagged PHASE-3-DELETE will be removed simultaneously.

---

## 12. Known Deviations

- **Silently skips zero-overlap sequences:** When a prediction DataFrame has no index overlap with actuals, the adapter silently skips it (continues to next). This contradicts the fail-loud principle (ADR-013) but matches legacy behavior.
- **Step assignment is positional:** Step is assigned as a 1-indexed ordinal based on the order of unique time values within each origin. This is a semantic risk for irregular sequences where positional step may not equal calendar lead time.
- **Emits DeprecationWarning:** Every call emits a `DeprecationWarning`. This is intentional but may be noisy in test output.

---

## End of Contract

This document defines the **intended meaning** of `PandasAdapter`.

Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
