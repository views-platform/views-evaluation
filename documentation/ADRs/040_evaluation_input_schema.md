# ADR-040: Evaluation Input Schema

**Status:** Accepted  
**Date:** 2025-06-16  
**Deciders:** Xiaolong  
**Consulted:** —  
**Informed:** All contributors  

## Context

A consistent input format is required to compare model performance across the VIEWS pipeline.
The native path via `EvaluationFrame` is the sole integration path. The legacy
`EvaluationManager` path was removed in Phase 3.

## Decision

### Caller-side Pandas convention (NOT this library's input format)

> **⚠ Scope (clarified 2026-08-02).** `views-evaluation` does **not** accept DataFrames.
> Its sole input is an `EvaluationFrame` of NumPy arrays. The Pandas layout below is the
> **upstream convention observed by callers** (views-pipeline-core and model repos) that
> the identifier synthesis in the next section is derived from — it documents where
> `time`/`unit`/`origin`/`step` come from, not an interface this library exposes.
>
> The `PandasAdapter` that once consumed this format was deleted in Phase 3 along with
> `EvaluationManager`; conversion is now entirely the caller's responsibility. This
> section previously read "Pandas Input Format (both paths)", which implied a
> DataFrame-accepting path that has not existed since 2026-04-01 (register **C-34**).

Callers producing an `EvaluationFrame` typically start from:

1. **Actuals**: A single `DataFrame` of observed values.
   - Index: `MultiIndex` of `(month_id, entity_id)`
   - Columns: one column named with the exact target variable name (e.g. `ged_sb_best`)

2. **Predictions**: A `list` of `DataFrames`, one per forecast sequence.
   - Index: same `MultiIndex` format as actuals
   - Columns: exactly one column named `f"pred_{target}"`
   - List order is semantically meaningful: position 0 = origin 0, position 1 = origin 1, etc.

3. **Target**: The string name of the target variable.

4. **Config**: The evaluation configuration dictionary (see integration guide for schema).

### Prediction Type Determination

Prediction type (point vs. sample) is determined structurally from the number of values per cell:
- S = 1 → point evaluation
- S > 1 → sample evaluation

No name-based inference occurs (ADR-012). Callers must ensure all cells in a prediction column
have the same number of values.

### Native Path Invariants

When constructing an `EvaluationFrame`, the following identifiers must be provided:

| Identifier | Source                                           |
|------------|--------------------------------------------------|
| `time`     | `month_id` level of the DataFrame MultiIndex     |
| `unit`     | `entity_id` level of the DataFrame MultiIndex    |
| `origin`   | 0-indexed position of the DataFrame in the list  |
| `step`     | 1-indexed positional ordinal within the sequence |

`step` is equivalent to lead time for regular (contiguous, same-length) sequences. For irregular
sequences, `step` is positional and may diverge from true calendar lead time.

## Consequences

- **Standardisation**: Uniform evaluation across all model repositories.
- **Strictness**: Requires adherence to naming conventions and index structures.
- **Explicit Identifiers**: All grouping logic is based on synthesised integer identifiers, not
  inferred from column names or DataFrame structure.

## Rationale

Enforcing a consistent schema ensures reproducibility. The support for multiple prediction sequences
aligns with the rolling-origin forecasting workflow documented in ADR-030. Explicit identifier
synthesis (rather than sniffing) complies with ADR-012 and makes the grouping logic auditable.
