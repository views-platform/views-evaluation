# Evaluation Output Schema

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Evaluation Output Schema  |
| ADR Number          | 005   |
| Status              | Proposed   |
| Author              | Xiaolong   |
| Date                | 16.06.2025     |

## Context
As part of our model evaluation workflow, we generate comprehensive reports summarizing model performance across a range of metrics and time periods. These reports are intended primarily for comparing ensemble models against their constituent models and baselines.

We use the `views-evaluation` package for computing evaluation metrics, while the report generation logic resides in the `views-pipeline-core` package. This separation of concerns avoids circular dependency, but requires a well-defined schema for passing data between the two components.

## Decision

We define a standard output schema for model evaluation reports using two formats:

1. **JSON file** – machine-readable output storing structured evaluation data.
2. **HTML file** – human-readable report with charts, tables, and summaries.

These files are stored in the `reports/` directory for each model within `views-models`.

To prevent a circular dependency between `views-evaluation` and `views-pipeline-core`, the `views-evaluation` package **does not** generate reports directly. Instead, it outputs intermediate results. These are then consumed by the reporting module in `views-pipeline-core` to generate final report files.

### Schema Overview (JSON)
Each report follows a standardized JSON structure that includes:
````
{
    "Target": "target",
    "Forecast Type": "point",
    "Level of Analysis": "cm",
    "Data Partition": "validation",
    "Training Period": [121,492],
    "Testing Period": [493,540],
    "Forecast Horizon": 36,
    "Number of Rolling Origins": 12,
    "Evaluation Results": [
        {
            "Type": "Ensemble",
            "Model Name": "ensemble_model",
            "MSE": mse_e,
            "MSLE": msle_e,
            "mean prediction": mp_e 
        },
        {
            "Type": "Constituent",
            "Model Name": "constitute_a",
            "MSE": mse_a,
            "MSLE": msle_a,
            "mean prediction": mp_a 
        },
        {
            "Type": "Constituent",
            "Model Name": "constitute_b",
            "MSE": mse_b,
            "MSLE": msle_b,
            "mean prediction": mp_b 
        }
        ...
    ]
}
````
Here, the 

The output file is name with the following name convention:
```
eval_validation_{conflict_type}_{timestamp}.json
```



## Consequences

**Positive Effects:**

- Avoids circular dependency between `views-evaluation` and `views-pipeline-core`.

- Provides consistent input for both HTML rendering and potential downstream systems (e.g., dashboards, APIs)

- Facilitates modularity and separation of concerns.


**Negative Effects:**

- Requires tight coordination between both packages to maintain schema compatibility

- Some redundancy between evaluation and report generation may occur

- May require schema migrations as new report sections are added



## Rationale

Generating reports within `views-pipeline-core` ensures full control over rendering, formatting, and contextual customization (e.g., comparing different model families). By letting `views-evaluation` focus strictly on metrics and alignment logic, we maintain cleaner package boundaries.


### Considerations

- This schema may evolve as we introduce new types of evaluation (e.g., correlation matrix).


- Reports are currently only generated for **ensemble models**, as comparison against constituent models is the primary use case.

-Future extensibility (e.g., visual version diffs) should be considered when evolving the format.



## Feedback and Suggestions
Any feedback or suggestion is welcomed

