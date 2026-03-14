# ADR-040: Evaluation Input Schema

| ADR Info            | Details                 |
|---------------------|-------------------------|
| Subject             | Evaluation Input Schema |
| ADR Number          | 040                     |
| Status              | Accepted                |
| Author              | Xiaolong                |
| Date                | 16.06.2025              |

## Context

A consistent input format is required to compare model performance across the VIEWS pipeline.
Two integration paths exist: the native path (primary) and the legacy path (`EvaluationManager`,
deprecated per ADR-011).

## Decision

### Pandas Input Format (both paths)

Both integration paths accept:

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

### Native Path Invariants (PandasAdapter)

When using `PandasAdapter`, the following identifiers are synthesised automatically:

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
