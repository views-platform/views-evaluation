# Integration Guide for `views-evaluation`

This guide explains how to use the `views-evaluation` library to evaluate conflict forecasting models.
It covers the architecture, the native API (recommended), the legacy API, identifier semantics, and
what the library does and does not do with your data.

---

## 1. Architecture Overview

The library has three layers:

```
  ┌─────────────────────────────────────────────┐
  │             Adapters (Bridge Layer)          │
  │  PandasAdapter — converts List[DataFrame]   │
  │  to EvaluationFrame; synthesises identifiers│
  └───────────────────┬─────────────────────────┘
                      │
  ┌───────────────────▼─────────────────────────┐
  │          EvaluationFrame (Core)              │
  │  Pure NumPy container: y_true, y_pred,      │
  │  identifiers {time, unit, origin, step}     │
  └───────────────────┬─────────────────────────┘
                      │
  ┌───────────────────▼─────────────────────────┐
  │         NativeEvaluator (Pure Math)          │
  │  Stateless engine: executes month-wise,     │
  │  sequence-wise, and step-wise schemas        │
  └───────────────────┬─────────────────────────┘
                      │
  ┌───────────────────▼─────────────────────────┐
  │        EvaluationReport (Results)            │
  │  Framework-agnostic results container;      │
  │  exposes to_dict(), to_dataframe(),         │
  │  get_schema_results()                        │
  └─────────────────────────────────────────────┘
```

`EvaluationManager` is a **legacy orchestrator** that wraps all four layers behind a single
`evaluate()` call. It is retained for backward compatibility and will be removed in Phase 3 of the
orchestrator migration (see `reports/2026-02-25_evaluation_frame_refactor/10_orchestrator_migration_plan.md`).

**New integrations should use the native API (§2). The legacy API is documented in §3.**

---

## 2. The Native API (Recommended)

### 2.1. Prerequisites

```bash
pip install views_evaluation
pip install pandas numpy  # only needed to prepare input DataFrames
```

### 2.2. Identifier Glossary

All evaluation logic operates on four identifiers that `PandasAdapter` synthesises from your input.
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
import pandas as pd
from views_evaluation.evaluation.adapters.pandas import PandasAdapter
from views_evaluation.evaluation.native_evaluator import NativeEvaluator

# --- 1. Prepare actuals ---
actuals_index = pd.MultiIndex.from_product(
    [range(500, 513), [101, 102]],
    names=['month_id', 'country_id']
)
actuals = pd.DataFrame(
    {'ged_sb_best': np.random.randint(0, 20, size=26)},
    index=actuals_index
)

# --- 2. Prepare predictions list (2 sequences, 12 steps each) ---
target = 'ged_sb_best'
pred_col = f'pred_{target}'
predictions_list = []

for origin_offset in range(2):
    months = range(500 + origin_offset, 512 + origin_offset)
    idx = pd.MultiIndex.from_product([months, [101, 102]], names=['month_id', 'country_id'])
    preds = pd.DataFrame({pred_col: [[v] for v in np.random.rand(len(idx)) * 20]}, index=idx)
    predictions_list.append(preds)

# --- 3. Configure ---
config = {
    'steps': list(range(1, 13)),
    'regression_targets': [target],
    'regression_point_metrics': ['MSE', 'RMSLE', 'Pearson'],
}

# --- 4. Adapt and evaluate ---
ef = PandasAdapter.from_dataframes(actual=actuals, predictions=predictions_list, target=target)

evaluator = NativeEvaluator(config)
report = evaluator.evaluate(ef)   # legacy_compatibility=True by default

# --- 5. Access results ---
print(report.to_dataframe('step'))         # step-wise DataFrame (MSE, RMSLE, Pearson per step)
print(report.to_dataframe('month'))        # month-wise DataFrame
print(report.to_dataframe('time_series'))  # sequence-wise DataFrame
print(report.to_dict())                    # full nested dict
```

### 2.6. The `legacy_compatibility` Flag

`NativeEvaluator.evaluate(ef, legacy_compatibility=True)` (default) caps step-wise evaluation to
the shortest sequence in the frame. If origin 0 has 12 steps and origin 1 has only 10 steps,
legacy mode evaluates steps 1–10 and leaves steps 11–12 empty. This reproduces a historic zip
truncation behaviour required for parity with the legacy system.

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

## 3. The Legacy API (`EvaluationManager`)

> **Deprecation notice:** `EvaluationManager` will be removed in Phase 3 of the orchestrator
> migration. New integrations must use the native API (§2). This section is retained for teams
> currently using the legacy path.

### 3.1. Differences from the Native API

- Accepts the same DataFrame inputs and config as §2.
- Applies **inverse transforms** based on target name prefixes:
  - `ln_` prefix: applies `exp(x) - 1` to both actuals and predictions
  - `lx_` prefix: applies a custom inverse log transform
  - `lr_` prefix: no transform (raw values)
  - No prefix: no transform
  This behaviour is **absent** from the native path, which always operates on data as provided.
- Returns a dict of `{schema: (dict, DataFrame)}` tuples, not an `EvaluationReport`.
- `legacy_compatibility` is hardcoded to `True` (cannot be changed).

### 3.2. Usage

```python
from views_evaluation.evaluation.evaluation_manager import EvaluationManager

manager = EvaluationManager()
config = {
    'steps': [1, 2, 3],
    'regression_targets': ['lr_ged_sb_best'],
    'regression_point_metrics': ['MSE', 'RMSLE', 'Pearson']
}

results = manager.evaluate(
    actual=actuals,         # same format as §2.3
    predictions=predictions_list,
    target='lr_ged_sb_best',
    config=config
)

# Access results (tuple format — not EvaluationReport)
step_df = results['step'][1]         # index 1 = DataFrame
step_dict = results['step'][0]       # index 0 = raw dict
```

---

## 4. What This Library Does NOT Do

- **Does not load or save data.** Pass DataFrames in; get an `EvaluationReport` (or dict) out.
- **Does not enforce k=12 or 36-month sequences.** The VIEWS standard (ADR-030) recommends
  k=12 rolling origins over 36-month evaluation windows, but this library accepts any sequence
  count and length.
- **Does not validate spatial or temporal alignment.** The adapter performs index intersection, but
  it does not verify that sequences are in chronological order or that all origins cover the same
  calendar range.
- **Does not produce output files.** Persistence is handled by `views-pipeline-core` per ADR-041.
