![GitHub License](https://img.shields.io/github/license/views-platform/views-evaluation)
![GitHub branch check runs](https://img.shields.io/github/check-runs/views-platform/views-evaluation/main)
![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/views-platform/views-evaluation)
![GitHub Release](https://img.shields.io/github/v/release/views-platform/views-evaluation)

<div style="width: 100%; max-width: 1500px; height: 400px; overflow: hidden; position: relative;">
  <img src="https://github.com/user-attachments/assets/1ec9e217-508d-4b10-a41a-08dface269c7" alt="VIEWS Twitter Header" style="position: absolute; top: -50px; width: 100%; height: auto;">
</div>

# **VIEWS Evaluation** 📊  

> **Part of the [VIEWS Platform](https://github.com/views-platform) ecosystem for large-scale conflict forecasting.**  

---

### ⚠️ **ATTENTION: Migration Notice (v0.4.0+)**

The evaluation ontology has been updated to be more explicit and task-specific. If your pipeline broke after updating, please update your configuration dictionary. The library now distinguishes between **regression** vs **classification** tasks, and **point** vs **sample** predictions.

**Key Changes:**
* `targets` is now **`regression_targets`** or **`classification_targets`**.
* `metrics` is now **`regression_point_metrics`**.
* All **`uncertainty`** keys have been renamed to **`sample`** (reflecting that we evaluate draws/samples from a distribution).

| Legacy Key | New Canonical Key |
|:--- |:--- |
| `targets` | `regression_targets` |
| `metrics` | `regression_point_metrics` |
| `regression_uncertainty_metrics` | `regression_sample_metrics` |
| `classification_uncertainty_metrics` | `classification_sample_metrics` |

> **⚠️ Legacy keys were REMOVED in 0.4.0 — they no longer work.**
>
> Earlier documentation stated that legacy keys still worked and emitted a
> `DeprecationWarning`. That was **incorrect**: the translation shim lived in
> `EvaluationManager`, which was deleted before 0.4.0 was published. There is no
> fallback and none will be restored.
>
> Since 0.5.0, a **legacy key** fails loudly at `NativeEvaluator.__init__` rather than
> being silently ignored:
>
> ```
> ValueError: Legacy evaluation config key(s) present: ['targets']. These were removed
> in 0.4.0 and are NOT translated automatically. Rename them: 'targets' ->
> 'regression_targets' or 'classification_targets'. See the README migration table.
> ```
>
> A **misspelled** evaluation key also fails, naming what it resembles — but the
> suspected key is only reported, never substituted:
>
> ```
> ValueError: Evaluation config key 'regression_target' is not recognised, but closely
> resembles 'regression_targets' — this looks like a typo. It has NOT been interpreted
> as 'regression_targets'; nothing is assumed on your behalf. ...
> ```
>
> Previously a legacy or misspelled metric key produced an **empty-but-successful
> report** with no error at all. Migrate using the table above.
>
> **An unrecognised key that resembles nothing is ignored, deliberately.**
> `views-pipeline-core` passes its whole combined model config through
> (`NativeEvaluator(context.configs)`), so the dict legitimately carries dozens of
> foreign keys — `batch_size`, `algorithm`, and so on. Rejecting unknown keys wholesale
> would break every model in the platform. See risk register C-33.
>
> This removal predates the deprecation policy now in force — see
> [ADR-022](documentation/ADRs/022_evolution_and_stability.md), which requires a
> `DeprecationWarning` in a published release plus one full release cycle before any
> public symbol or config key is removed.

---

## 📚 **Table of Contents**  

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Role in the VIEWS Pipeline](#role-in-the-views-pipeline)
4. [Features](#features)
5. [Installation](#installation)
6. [Architecture](#architecture)
7. [Project Structure](#project-structure)
8. [Contributing](#contributing)
9. [License](#license)
10. [Acknowledgements](#acknowledgements)  

---

## 🧠 **Overview**  

The **VIEWS Evaluation** repository provides a standardized framework for **assessing time-series forecasting models** used in the **VIEWS conflict prediction pipeline**. It ensures consistent, robust, and interpretable evaluations through **metrics tailored to conflict-related data**, which often exhibit **right-skewness and zero-inflation**.

The library is built on a **three-layer architecture** with a framework-agnostic NumPy core, ensuring that all mathematical evaluation logic is independent of Pandas or any other data-frame library.  

---

## 🚀 **Quick Start**

```python
from views_evaluation import EvaluationFrame, NativeEvaluator
import numpy as np

# 1. Construct EvaluationFrame with NumPy arrays
ef = EvaluationFrame(
    y_true=y_true_array,
    y_pred=y_pred_array,  # shape (N, S) where S >= 1
    identifiers={'time': times, 'unit': units, 'origin': origins, 'step': steps},
    metadata={'target': 'ged_sb_best'},
)

# 2. Configure and evaluate
config = {
    "steps": [1, 2, 3, 4, 5, 6],
    "regression_targets": ["ged_sb_best"],
    "regression_point_metrics": ["MSE", "RMSLE", "Pearson"],
}
evaluator = NativeEvaluator(config)
report = evaluator.evaluate(ef)

# 3. Access results
report.to_dataframe("step")          # pd.DataFrame
report.to_dict()                     # nested dict
report.get_schema_results("month")   # typed metrics dataclass
```

> For the full walkthrough including input formatting and sample evaluation, see [`documentation/integration_guide.md`](documentation/integration_guide.md).

---

## 🌍 **Role in the VIEWS Pipeline**

VIEWS Evaluation ensures **forecasting accuracy and model robustness** as the **official evaluation component** of the VIEWS ecosystem.

### **Pipeline Integration:**
1. **Model Predictions** →
2. **EvaluationFrame** (validated NumPy container) →
3. **NativeEvaluator** (metrics computation) →
4. **EvaluationReport** (structured results)  

### **Integration with Other Repositories:**  
- **[views-pipeline-core](https://github.com/views-platform/views-pipeline-core):** Supplies preprocessed data for evaluation.  
- **[views-models](https://github.com/views-platform/views-models):** Provides trained models to be assessed.  
- **[views-stepshifter](https://github.com/views-platform/views-stepshifter):** Evaluates **time-shifted forecasting models**.  
- **[views-hydranet](https://github.com/views-platform/views-hydranet):** Supports **spatiotemporal deep learning model evaluations**.  

---

## ✨ **Features**  
* **Comprehensive Evaluation Framework**: The `NativeEvaluator` provides structured, stateless evaluation of time series predictions across a 2×2 matrix of **regression/classification** tasks and **point/sample** prediction types.
* **Multiple Evaluation Schemas**:
  * **Step-wise evaluation**: groups and evaluates predictions by the respective steps from all models.
  * **Time-series-wise evaluation**: evaluates predictions for each time-series.
  * **Month-wise evaluation**: groups and evaluates predictions at a monthly level.
* **Support for Multiple Metrics** (see table below for details)

### **Available Metrics**

Metrics are organized by the 2×2 evaluation matrix: **task** (regression / classification) × **prediction type** (point / sample).

#### Regression Point Metrics

| Metric | Key | Description | Status |
|--------|-----|-------------|:------:|
| Mean Squared Error | `MSE` | Average of squared differences | ✅ |
| Mean Squared Log Error | `MSLE` | MSE computed on log-transformed values | ✅ |
| Root Mean Squared Log Error | `RMSLE` | Square root of MSLE | ✅ |
| Earth Mover's Distance | `EMD` | Wasserstein distance between distributions | ✅ |
| Pearson Correlation | `Pearson` | Linear correlation between predictions and actuals | ✅ |
| Mean Tweedie Deviance | `MTD` | Tweedie deviance (configurable power), ideal for zero-inflated data | ✅ |
| Mean Prediction | `y_hat_bar` | Average of all predicted values (diagnostic) | ✅ |
| Magnitude Calibration Ratio | `MCR_point` | Ratio of predicted to actual magnitude | ✅ |
| Sinkhorn Distance | `SD` | Regularized optimal transport distance | ❌ |
| pseudo-Earth Mover Divergence | `pEMDiv` | Efficient EMD approximation | ❌ |
| Variogram | `Variogram` | Spatial/temporal correlation structure score | ❌ |

#### Regression Sample Metrics

| Metric | Key | Description | Status |
|--------|-----|-------------|:------:|
| Continuous Ranked Probability Score | `CRPS` | Calibration and sharpness of probabilistic forecasts | ✅ |
| Threshold-Weighted CRPS | `twCRPS` | CRPS emphasizing values above a threshold | ✅ |
| Mean Interval Score | `MIS` | Prediction interval width and coverage | ✅ |
| Quantile Interval Score | `QIS` | Interval score at specified quantiles | ✅ |
| Coverage | `Coverage` | Proportion of actuals within prediction intervals | ✅ |
| Ignorance Score | `Ignorance` | Logarithmic scoring rule for probabilistic predictions | ✅ |
| Mean Prediction | `y_hat_bar` | Average of all predicted values (diagnostic) | ✅ |
| Magnitude Calibration Ratio | `MCR_sample` | Ratio of predicted to actual magnitude | ✅ |

#### Classification Point Metrics

| Metric | Key | Description | Status |
|--------|-----|-------------|:------:|
| Average Precision | `AP` | Area under precision-recall curve | ✅ |

#### Classification Sample Metrics

| Metric | Key | Description | Status |
|--------|-----|-------------|:------:|
| Continuous Ranked Probability Score | `CRPS` | Calibration and sharpness | ✅ |
| Threshold-Weighted CRPS | `twCRPS` | CRPS emphasizing values above a threshold | ✅ |
| Brier Score | `Brier` | Accuracy of probabilistic binary predictions | ❌ |
| Jeffreys Divergence | `Jeffreys` | Symmetric measure of distribution difference | ❌ |

> **Note:** Metrics marked ❌ are defined in the catalog but not yet implemented — requesting them raises a clear `ValueError`.

---

### 📝 **Configuration Schema**

The `NativeEvaluator` accepts a configuration dictionary (`EvaluationConfig` TypedDict) with the following keys:

| Key | Type | Description |
|:--- |:--- |:--- |
| `steps` | `List[int]` | List of forecast steps to evaluate (e.g., `[1, 3, 6, 12]`). |
| `regression_targets` | `List[str]` | List of continuous targets (e.g., `['ged_sb_best']`). |
| `regression_point_metrics` | `List[str]` | Metrics to compute for regression point predictions. |
| `regression_sample_metrics` | `List[str]` | Metrics to compute for regression sample predictions (e.g., `['CRPS']`). |
| `classification_targets` | `List[str]` | List of binary targets (e.g., `['by_sb_best']`). |
| `classification_point_metrics` | `List[str]` | Metrics to compute for classification probability scores. |
| `classification_sample_metrics` | `List[str]` | Metrics to compute for classification sample predictions. |
| `evaluation_profile` | `str` | Named hyperparameter profile (default: `"base"`). See `views_evaluation/profiles/`. |
| `metric_hyperparameters` | `Dict[str, Dict]` | Per-metric overrides that take precedence over the profile. |

#### **Example Configuration:**

```python
config = {
    "steps": [1, 3, 6, 12],
    "regression_targets": ["ged_sb_best"],
    "regression_point_metrics": ["MSE", "RMSLE", "Pearson"],
    "regression_sample_metrics": ["CRPS", "twCRPS", "MIS", "Coverage"],
    "evaluation_profile": "base",  # or "hydranet_ucdp"
    "metric_hyperparameters": {
        "twCRPS": {"threshold": 10.0},  # override profile default
    },
}
```

---

* **Data Integrity Checks**: Validates input arrays for shape consistency, NaN/infinity, and required identifiers.
* **Framework-Agnostic Core**: All evaluation operates on pure NumPy arrays via `EvaluationFrame`.
* **Metric Catalog & Profiles**: Hyperparameters are managed through named evaluation profiles with a Chain of Responsibility resolver (model overrides → profile → fail loud).  

---

## ⚙️ **Installation**  

### **Prerequisites**  
- Python **>= 3.11**  

### **From PyPI**
```
pip install views_evaluation
```

---
## 🏗 **Architecture**

The library follows a strict three-layer architecture (ADR-011):

```
Level 0 — Pure Core (NumPy + SciPy + sklearn; no dataframe libraries)
  EvaluationFrame            Canonical data container (y_true, y_pred, identifiers)
  NativeEvaluator            Stateless evaluation engine (month/sequence/step schemas)
  MetricCatalog              Genome registry mapping metrics → functions + required params
  native_metric_calculators  The metric kernels
  Profiles                   Named hyperparameter sets (base, hydranet_ucdp, ...)

Level 1 — Bridge / Emit
  EvaluationReport      Results container with dict / DataFrame / MetricFrame export
  MetricFrame           Typed, provenance-stamped evaluation-of-record (views-frames ADR-020)

Level 2 — Orchestration
  External to this repository (e.g. views-pipeline-core)
```

Each component appears in exactly one level; dependencies flow toward the core (ADR-011).

**Key design decisions:**
- **ADR-011**: No Pandas/Polars imports in Level 0 — math is framework-agnostic.
- **ADR-013**: Fail-loud — all structural failures raise exceptions with actionable messages, never silently degrade.
- **ADR-015**: Degenerate and empty results — a computation that cannot produce a result raises; it never returns a value standing in for one. A metric may return a number for a degenerate input only where that number is genuinely the answer (e.g. `MCR`'s `inf`), documented and tested.
- **ADR-022**: Evolution and stability — public API removals require a `DeprecationWarning` in a prior published release plus one full release cycle.
- **ADR-042**: Metric catalog — each metric declares its required hyperparameters ("genome"); values are resolved via Chain of Responsibility.  

---

## 🗂 **Project Structure**  

```plaintext
views-evaluation/
├── views_evaluation/
│   ├── __init__.py                        # Public API exports
│   ├── adapters/
│   │   └── __init__.py                     # Reserved for future framework bridges
│   ├── evaluation/
│   │   ├── config_schema.py               # EvaluationConfig TypedDict (authoritative key set)
│   │   ├── evaluation_frame.py            # Core data container
│   │   ├── evaluation_report.py           # Results container
│   │   ├── metric_frame.py                # Evaluation-of-record emit artifact (needs [frames])
│   │   ├── metric_catalog.py              # ADR-042 registry + resolver
│   │   ├── metrics.py                     # Typed metric dataclasses
│   │   ├── native_evaluator.py            # Core evaluation engine
│   │   └── native_metric_calculators.py   # Metric implementations
│   └── profiles/
│       ├── base.py                        # Standard hyperparameter values
│       └── hydranet_ucdp.py               # Domain-specific profile
├── tests/                                 # Green/Beige/Red suites (ADR-020)
├── documentation/
│   ├── ADRs/                              # Architecture Decision Records
│   ├── CICs/                              # Class Intent Contracts
│   ├── integration_guide.md               # Full API walkthrough
│   └── evaluation_concepts.md             # Domain concepts
├── pyproject.toml
└── README.md
```

---

## 🤝 **Contributing**  

We welcome contributions! Please follow the **[VIEWS Contribution Guidelines](https://github.com/views-platform/docs)**.  

---

## 📜 **License**  

This project is licensed under the [LICENSE](/LICENSE) file. 

---

## 💬 **Acknowledgements**  

<p align="center">
  <img src="https://raw.githubusercontent.com/views-platform/docs/main/images/views_funders.png" alt="Views Funders" width="80%">
</p>

Special thanks to the **VIEWS MD&D Team** for their collaboration and support.  

