# Integration Guide for `views-evaluation`

This guide explains how to use the `views-evaluation` library to evaluate conflict forecasting models.
It covers the architecture, the native API (recommended), the legacy API, identifier semantics, and
what the library does and does not do with your data.

---

## 1. Architecture Overview

The library is a pure-math evaluation engine with two core components:

```
  ┌───────────────────────────────────────────────┐
  │          EvaluationFrame (Core)                │
  │  Pure NumPy container: y_true, y_pred,        │
  │  identifiers {time, unit, origin, step}       │
  └───────────────────┬───────────────────────────┘
                      │
  ┌───────────────────▼───────────────────────────┐
  │         NativeEvaluator (Pure Math)            │
  │  Stateless engine: executes month-wise,       │
  │  sequence-wise, and step-wise schemas          │
  └───────────────────┬───────────────────────────┘
                      │
  ┌───────────────────▼───────────────────────────┐
  │        EvaluationReport (Results)              │
  │  Framework-agnostic results container;        │
  │  exposes to_dict(), to_dataframe(),           │
  │  get_schema_results()                          │
  └───────────────────────────────────────────────┘
```

Callers (e.g. views-pipeline-core) are responsible for constructing `EvaluationFrame` from their
own data formats. This library has no knowledge of Pandas, Polars, or any external data framework.

---

## 2. The Native API (Recommended)

### 2.1. Prerequisites

```bash
pip install views_evaluation
pip install pandas numpy  # only needed to prepare input DataFrames
```

### 2.2. Identifier Glossary

All evaluation logic operates on four identifiers that must be provided in the `EvaluationFrame`.
Understanding them is required:

| Identifier | Type    | Meaning                                                                 |
|------------|---------|-------------------------------------------------------------------------|
| `time`     | int     | Calendar month id (e.g. `500`). Direct from the DataFrame MultiIndex.   |
| `unit`     | int     | Spatial entity id (e.g. `country_id`, `priogrid_gid`). From MultiIndex. |
| `origin`   | int     | 0-indexed position of the prediction DataFrame in the input list.       |
|            |         | Origin 0 = first sequence, origin 1 = second sequence, etc.            |
|            |         | In a rolling-origin evaluation, this encodes *which forecast was made*. |
| `step`     | int     | 1-indexed positional ordinal within a sequence (step 1 = first month    |
|            |         | of that forecast, step 2 = second, …). Equivalent to lead time for      |
|            |         | regular sequences. Synthesised by the adapter; not from your input.     |

### 2.3. Formatting Your Input Data

**Ground truth** — a single `pandas.DataFrame`:
- Index: `MultiIndex` of `(month_id, entity_id)`
- Columns: exactly one column named with the target variable (e.g. `ged_sb_best` or `by_sb_best`)

**Predictions** — a `list` of `pandas.DataFrame`:
- Each DataFrame covers one forecast sequence (all lead times from one rolling origin)
- Index: same `MultiIndex` format as actuals
- Column: exactly one column named `f"pred_{target_name}"`
- Values for point evaluation: each cell is a list/array with **one** float, e.g. `[10.5]`
- Values for sample evaluation: each cell is a list/array with **multiple** floats, e.g. `[8.1, 9.5, 10.5]`
- Order matters: list position determines `origin`. Pass sequences in chronological order.

> **No transforms**: The native evaluator does not apply any inverse transformations. Pass data
> on the original scale. Target names do not need transformation prefixes (e.g. `ged_sb_best`,
> not `ln_ged_sb_best`).

### 2.4. Configuration Dictionary

```python
config = {
    # Exactly which step positions to evaluate (1-indexed, must match sequence length)
    'steps': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],

    # Declare which targets are regression vs classification
    'regression_targets': ['ged_sb_best'],
    # 'classification_targets': ['by_sb_best'],

    # Metrics to compute — choose the right category for your task and pred type
    'regression_point_metrics': ['MSE', 'RMSLE', 'Pearson'],
    # 'regression_sample_metrics': ['CRPS', 'MIS', 'Coverage'],
    # 'classification_point_metrics': ['AP'],
    # 'classification_sample_metrics': ['CRPS'],
}
```

`steps` declares the **exact** step positions to evaluate. If your sequences are 12 months long,
use `[1, 2, ..., 12]`. Sparse configs (e.g. `[1, 3, 6, 12]`) evaluate only those four positions.

### 2.5. Running the Evaluation

```python
import numpy as np
from views_evaluation import EvaluationFrame, NativeEvaluator

# --- 1. Construct EvaluationFrame from NumPy arrays ---
ef = EvaluationFrame(
    y_true=y_true_array,           # shape (N,)
    y_pred=y_pred_array,           # shape (N, S) where S >= 1
    identifiers={
        'time':   time_ids,        # shape (N,) — calendar month ids
        'unit':   unit_ids,        # shape (N,) — spatial entity ids
        'origin': origin_ids,      # shape (N,) — sequence index
        'step':   step_ids,        # shape (N,) — 1-indexed lead time
    },
    metadata={'target': 'ged_sb_best'},
)

# --- 2. Configure ---
config = {
    'steps': list(range(1, 13)),
    'regression_targets': ['ged_sb_best'],
    'regression_point_metrics': ['MSE', 'RMSLE', 'Pearson'],
}

# --- 3. Evaluate ---
evaluator = NativeEvaluator(config)
report = evaluator.evaluate(ef)

# --- 4. Access results ---
print(report.to_dict())                    # full nested dict
print(report.to_dataframe('step'))         # step-wise DataFrame
print(report.to_dataframe('month'))        # month-wise DataFrame
print(report.to_dataframe('time_series'))  # sequence-wise DataFrame
```

### 2.6. The `legacy_compatibility` Flag

`NativeEvaluator.evaluate(ef, legacy_compatibility=True)` caps step-wise evaluation to
the shortest sequence in the frame. If origin 0 has 12 steps and origin 1 has only 10 steps,
legacy mode evaluates steps 1–10 and leaves steps 11–12 empty. The default is `False` (evaluate
all steps with available data).

Set `legacy_compatibility=False` to evaluate all steps that have any data, regardless of whether
shorter sequences exist.

### 2.7. The `EvaluationReport` API

```python
report.target        # str: target variable name
report.task          # str: 'regression' or 'classification'
report.pred_type     # str: 'point' or 'sample'

report.to_dict()     # {'target': ..., 'task': ..., 'pred_type': ...,
                     #  'schemas': {'month': {...}, 'time_series': {...}, 'step': {...}}}

report.to_dataframe('month')        # pd.DataFrame, index = group keys
report.to_dataframe('time_series')  # pd.DataFrame
report.to_dataframe('step')         # pd.DataFrame
report.to_dataframe('raw')          # passthrough to internal results dict

report.get_schema_results('month')  # dict mapping key → typed metrics dataclass
```

---

## 3. What This Library Does NOT Do

- **Does not load or save data.** Construct `EvaluationFrame` from NumPy arrays; get an `EvaluationReport` out.
- **Does not perform data alignment or adaptation.** Callers (e.g. views-pipeline-core's `EvaluationAdapter`) are responsible for aligning actuals with predictions and synthesising identifiers.
- **Does not enforce k=12 or 36-month sequences.** The VIEWS standard (ADR-030) recommends
  k=12 rolling origins over 36-month evaluation windows, but this library accepts any sequence
  count and length.
- **Does not validate spatial or temporal alignment.** It verifies shape consistency and NaN/Inf
  rejection, but does not verify that sequences are chronologically ordered.
- **Does not produce output files.** Persistence is handled by `views-pipeline-core` per ADR-041.
