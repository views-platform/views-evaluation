# Class Intent Contract: MetricCatalog

**Status:** Active  
**Owner:** Evaluation Core  
**Last reviewed:** 2026-04-04  
**Related ADRs:** ADR-042 (Metric Catalog), ADR-012 (Authority), ADR-013 (Observability)

---

## 1. Purpose

A genome registry and Chain of Responsibility resolver for evaluation metric hyperparameters. Declares what each metric requires (its genome) but provides NO default values. Values are supplied by named profiles and/or per-model overrides.

---

## 2. Non-Goals (Explicit Exclusions)

- This module does **not** compute metrics (that is the role of metric calculator functions).
- This module does **not** supply default hyperparameter values (that is the role of named profiles in `views_evaluation/profiles/`).
- This module does **not** validate data shapes or content (that is the role of `EvaluationFrame`).
- This module does **not** know about DataFrames or any external data framework.

---

## 3. Responsibilities and Guarantees

- **Genome Declaration:** Each `MetricSpec` guarantees an immutable declaration of which hyperparameters a metric requires (the `genome` tuple) and whether the metric is implemented.
- **Membership Declaration:** `METRIC_MEMBERSHIP` guarantees a complete mapping of `(task, pred_type)` pairs to valid metric name sets.
- **Chain of Responsibility Resolution:** `resolve_metric_params()` guarantees that hyperparameters are resolved in strict order: model overrides → named profile → fail loud. No silent defaults.
- **Fail-Loud on Missing Params:** Guarantees `ValueError` if a required parameter is missing from both overrides and profile.
- **Fail-Loud on None Values:** Guarantees `ValueError` if a resolved parameter is `None`.
- **Fail-Loud on Unknown Params:** Guarantees `ValueError` if model overrides contain parameters not in the metric's genome.
- **Fail-Loud on Unimplemented Metrics:** Guarantees `ValueError` with clear message if an unimplemented metric is requested.

---

## 4. Inputs and Assumptions

- **`resolve_metric_params(metric_name, model_overrides, profile)`:**
  - `metric_name` must exist in `METRIC_CATALOG`.
  - `model_overrides` is a dict of per-metric parameter overrides (may be empty).
  - `profile` is a named evaluation profile dict (e.g. `BASE_PROFILE`).
- **Metric functions:** Each function referenced by a `MetricSpec` must accept `(y_true, y_pred, **resolved_params)`.
- **Genome completeness:** All hyperparameters required by a metric function must be declared in the spec's `genome` tuple.

---

## 5. Outputs and Side Effects

- **`resolve_metric_params()`** returns a `Dict[str, Any]` of resolved hyperparameters ready to pass as `**kwargs` to the metric function. Empty dict for metrics with no genome.
- **No side effects.** The module is purely declarative; no state mutation, no I/O, no logging.

---

## 6. Failure Modes and Loudness

- `ValueError` if `metric_name` is unknown (not in `METRIC_CATALOG`).
- `ValueError` if metric is not implemented (`spec.implemented == False`).
- `ValueError` if a genome parameter is missing from both overrides and profile.
- `ValueError` if a resolved parameter is `None`.
- `ValueError` if overrides contain unknown parameters not in the genome.
- `ValueError` if overrides are provided for a metric with empty genome.
- `ValueError` if a probability/proportion parameter (`alpha`, `quantile`, `lower_quantile`, `upper_quantile`) is not in the open interval (0, 1).
- `ValueError` if `lower_quantile >= upper_quantile` for metrics requiring both (e.g. QIS).

All failures are immediate and explicit. No warnings, no fallbacks, no silent degradation.

---

## 7. Boundaries and Interactions

- **Upstream:** Consumed by `NativeEvaluator._calculate_metrics()`.
- **Internal:** Imports metric functions from `native_metric_calculators.py`.
- **Downstream:** Named profiles (`views_evaluation/profiles/`) supply values consumed by the resolver.
- **Isolation:** Must not import Pandas, Polars, or any external data framework. Only depends on `native_metric_calculators` and standard library.

---

## 8. Examples of Correct Usage

```python
from views_evaluation.evaluation.metric_catalog import METRIC_CATALOG, resolve_metric_params
from views_evaluation.profiles.base import BASE_PROFILE

# Resolve params for twCRPS using base profile
params = resolve_metric_params("twCRPS", {}, BASE_PROFILE)
# → {"threshold": 0.0}

# Override threshold for a specific model
params = resolve_metric_params("twCRPS", {"threshold": 2.0}, BASE_PROFILE)
# → {"threshold": 2.0}

# Metrics with no genome return empty dict
params = resolve_metric_params("MSE", {}, BASE_PROFILE)
# → {}
```

---

## 9. Examples of Incorrect Usage

- Hardcoding hyperparameter defaults inside metric function signatures — the catalog pattern requires all values to come from profiles or overrides.
- Calling `resolve_metric_params` with `None` as the profile — a real profile dict is always required.
- Adding a new metric to `METRIC_CATALOG` without adding its genome params to at least one profile — all callers will get `ValueError`.
- Passing overrides for metrics with empty genome (e.g. `resolve_metric_params("MSE", {"power": 1.5}, profile)`) — raises `ValueError`.

---

## 10. Test Alignment

- **Green:** `tests/test_metric_catalog.py` — registry snapshot integrity, resolver happy path, genome completeness checks.
- **Beige:** `tests/test_metric_catalog.py` — partial overrides, profile-only resolution, edge case param values.
- **Red:** `tests/test_metric_catalog.py` — unknown metrics, unimplemented metrics, missing params, None values, unknown overrides.
- **Red (bounds):** `tests/test_metric_catalog.py::TestResolveMetricParamsBoundsRed` — 7 tests for out-of-range alpha/quantile and crossed QIS quantiles.
- **Correctness:** `tests/test_metric_calculators.py::TestGoldenValues` — 17 golden-value tests for all implemented metrics.
- **Correctness (Brier/QS):** `tests/test_metric_calculators.py::TestBrierScore` + `TestQuantileScore` — 10 golden-value tests for the 3 Brier variants and 2 QS variants.

---

## 11. Evolution Notes

- New metrics are added by: (1) implementing the function in `native_metric_calculators.py`, (2) adding a `MetricSpec` to `METRIC_CATALOG`, (3) adding to `METRIC_MEMBERSHIP`, (4) adding genome values to relevant profiles, (5) adding a field to the typed metrics dataclass in `metrics.py`.
- The legacy dispatch dicts were removed in Phase 3. `METRIC_MEMBERSHIP` is the single source of truth.
- Profile structure is stable; new profiles are added by creating a new file in `profiles/`.
- Bounds validation added for probability/proportion parameters (2026-04-04, C-18): `alpha`, `quantile`, `lower_quantile`, `upper_quantile` must be in (0, 1). Cross-parameter validation for QIS quantile ordering.
- Explicit Brier variants added (2026-04-09): `Brier_sample`/`Brier_point` replaced by three task-explicit variants: `Brier_cls_point`, `Brier_cls_sample`, `Brier_rgs_sample`. The `_cls_`/`_rgs_` infix denotes the task context (classification vs. regression). `Brier_rgs_point` is intentionally omitted — a regression point estimate is not a probability. `Brier_cls_sample` averages probability samples (`mean(y_pred)`); `Brier_rgs_sample` binarises count samples (`mean(y_pred > threshold)`). Catalog size: 24 → 25.

---

## 12. Known Deviations

- **No profile completeness validation:** There is no mechanism to verify that a profile provides values for all metrics with non-empty genomes. A profile missing a metric's params will only fail at evaluation time, not at profile registration.
- **Golden-value coverage complete:** 17 tests in `tests/test_metric_calculators.py::TestGoldenValues` plus 10 Brier/QS golden-value tests cover all implemented metrics (C-07 closed 2026-04-02).
- **Breaking rename (2026-04-09):** `Brier_sample` and `Brier_point` were replaced by `Brier_cls_point`, `Brier_cls_sample`, and `Brier_rgs_sample`. Dataclass fields in `ClassificationPointEvaluationMetrics`, `ClassificationSampleEvaluationMetrics`, and `RegressionSampleEvaluationMetrics` were renamed/added accordingly. External consumers accessing `.Brier_sample` or `.Brier_point` must update.

---

## End of Contract

This document defines the **intended meaning** of the MetricCatalog module (`MetricSpec`, `METRIC_CATALOG`, `METRIC_MEMBERSHIP`, `resolve_metric_params`).

Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
