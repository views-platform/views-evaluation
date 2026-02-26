# ADR-040: Evaluation Input Schema

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Evaluation Input Schema  |
| ADR Number          | 040   |
| Status              | Proposed   |
| Author              | Xiaolong   |
| Date                | 16.06.2025     |

## Context
A consistent input format is required to compare model performance across the VIEWS pipeline.

## Decision
The `views-evaluation` package standardizes the input for model evaluation via the `EvaluationManager`.

The system accepts:
1. **Actuals**: A DataFrame of observed values.
2. **Predictions**: A list of DataFrames (sequences).
3. **Target**: The name of the target variable.
4. **Config**: The model configuration dictionary.

### Invariants
- Both Actuals and Predictions must use a MultiIndex of `(month_id, entity_id)`.
- Prediction columns must be named `f'pred_{target}'`.
- The system automatically determines the evaluation type (point or sample) via "sniffing" (Deprecated, see ADR-012).

## Consequences
- **Standardization**: Uniform evaluation across all model repositories.
- **Strictness**: Requires adherence to naming conventions and index structures.

## Rationale
Enforcing a consistent schema ensures reproducibility. The support for multiple prediction sequences aligns with our rolling-origin forecasting workflow.
