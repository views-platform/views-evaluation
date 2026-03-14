# ADR-041: Evaluation Output Schema

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Evaluation Output Schema  |
| ADR Number          | 041   |
| Status              | Proposed   |
| Author              | Xiaolong   |
| Date                | 16.06.2025     |

## Context
Standardized reports are necessary for comparing ensemble models against constituent models and baselines.

## Decision
We define a standard output schema for evaluation reports. To prevent circular dependencies, `views-evaluation` returns a structured dictionary, and `views-pipeline-core` handles the persistence to disk.

### Formats
1. **JSON**: Machine-readable structured data.
2. **HTML**: Human-readable report with visualizations.

### Schema Overview (JSON)
The JSON structure includes metadata (Target, Level, Partition, Training/Testing periods) and an `Evaluation Results` list containing metric values for each model evaluated.

## Rationale
Saving reports within `views-pipeline-core` ensures full control over formatting and context while keeping `views-evaluation` focused on the mathematical core.
