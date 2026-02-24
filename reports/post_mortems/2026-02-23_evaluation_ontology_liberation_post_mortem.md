# Post-Mortem: Evaluation Ontology Liberation

**Date:** 2026-02-23
**Authors:** Simon Polichinel von der Maase + Claude Sonnet 4.6
**Branch:** `feature/documentation-verification-suite`
**Merged into:** pending PR to `development`
**Related documents:**
- `reports/investigations/2026-02-21_evaluation_ontology_liberation_plan.md` — the architectural manifesto this session executed
- `reports/technical_debt_backlog.md`
- ADR `documentation/ADRs/001_evaluation_metrics.md`

---

## 1. Background

The immediate trigger for this session was a crash in HydraNet evaluation:

```
ValueError: Target by_sb_best is not a valid target
```

`EvaluationManager.transform_data()` inspected column name prefixes (`ln_`, `lx_`, `lr_`) to decide which inverse mathematical transformation to apply before computing metrics. Any target whose prefix was not on the internal whitelist raised a hard `ValueError`. HydraNet's binary classification target `by_sb_best` had no recognised prefix and crashed every evaluation run.

The manifesto written the day before (2026-02-21) had already diagnosed this correctly — the `ValueError` was only the most visible symptom of a broader problem: `EvaluationManager` had accumulated domain knowledge it should never have had.

---

## 2. What We Did

### 2.1 Branch setup and review (morning)

- Checked out `feature/documentation-verification-suite` via a git worktree
- Diffed the branch against `development` — found a significant body of new documentation, a new test suite (`conftest.py`, `test_adversarial_inputs.py`, `test_documentation_contracts.py`, `test_data_contract.py`, `test_evaluation_schemas.py`, `test_metric_correctness.py`), and targeted source changes
- Merged `development` into the feature branch (clean merge, 4 files: MTD metric, updated README and tests)
- Ran the full test suite: **58/58 passing**
- Ran ruff linting: **1 fix** — unused variable `step` (F841) in `_split_dfs_by_step`
- Committed and pushed the lint fix

### 2.2 Manifesto analysis

Read the full `2026-02-21_evaluation_ontology_liberation_plan.md` and extended its analysis. The manifesto identified four sites of overreach; the extended analysis surfaced five structural weaknesses in the proposed remediation:

1. **`convert_to_array` underweighted** — every metric function is coupled to the array-per-cell DataFrame format, making them non-pure and independently untestable
2. **`pred_` convention inadequately scrutinised** — it bleeds into every metric function and should be isolated at the `EvaluationManager` level
3. **Migration hook has a silent failure mode** — the proposed `prepare_predictions_for_evaluation` hook could silently produce wrong metrics if forgotten
4. **`calculate_ap` threshold migration needs enforcement, not documentation** — a required config field, not a convention
5. **Cross-repo coordination complexity underestimated** — no versioning strategy proposed for the multi-repo migration

### 2.3 Architectural agreement

A substantial design discussion followed. The key decisions agreed:

**The core contract:**
> Models always return predictions on the original scale. No transformations happen at evaluation time. Ever.

This made the `prepare_predictions_for_evaluation` hook from the manifesto's Phase 2 unnecessary — there is nothing to hook into because the transformations do not happen at this stage at all.

**Config-driven dispatch, never inference:**
- Task type (regression / classification) — declared explicitly in config
- Prediction type (point / uncertainty) — detected structurally from data shape (array length), which is legitimate because it reads structure not semantics

**Fail loud, fail fast:**
- `AP` applied to a regression target must raise immediately, not silently apply `threshold=25`
- Missing config keys raise `KeyError` at the top of `evaluate()`, before any data is touched
- No defaults that mask developer intent

**The 2×2 matrix:**

|  | Point | Uncertainty |
|---|---|---|
| **Regression** | MSE, RMSLE, Pearson, MTD, ... | CRPS, MIS, Coverage, Ignorance |
| **Classification** | AP | CRPS, Brier, Jeffreys |

Both regression and classification can have distributional predictions — HydraNet samples posteriors over both expected counts and event probabilities simultaneously.

**Config schema:**
```python
{
    "steps":                              [1, 2, 3, ...],
    "regression_targets":                 ["lr_ged_sb_best"],
    "regression_point_metrics":           ["MSE", "RMSLE", "Pearson", "MTD"],
    "regression_uncertainty_metrics":     ["CRPS", "MIS", "Coverage"],
    "classification_targets":             ["by_ged_sb_best"],
    "classification_point_metrics":       ["AP"],
    "classification_uncertainty_metrics": ["CRPS", "Brier"],
}
```

Legacy keys `targets` and `metrics` accepted with a loud deprecation warning, translated to `regression_targets` and `regression_point_metrics` respectively.

### 2.4 Implementation — views-evaluation v0.4.0

All changes landed on `feature/documentation-verification-suite`, commit `19266b9`.

**`metric_calculators.py`**
- `calculate_ap()` — removed `threshold=25` entirely. Function now expects pre-binarised actuals (0/1) and probability scores. Thresholding is the model pipeline's responsibility.
- Four canonical dispatch dicts replacing the two old ones:
  - `REGRESSION_POINT_METRIC_FUNCTIONS`
  - `REGRESSION_UNCERTAINTY_METRIC_FUNCTIONS`
  - `CLASSIFICATION_POINT_METRIC_FUNCTIONS`
  - `CLASSIFICATION_UNCERTAINTY_METRIC_FUNCTIONS`
- Old `POINT_METRIC_FUNCTIONS` and `UNCERTAINTY_METRIC_FUNCTIONS` retained as deprecated union aliases

**`metrics.py`**
- Four new dataclasses mirroring the dispatch dicts:
  - `RegressionPointEvaluationMetrics`
  - `RegressionUncertaintyEvaluationMetrics`
  - `ClassificationPointEvaluationMetrics`
  - `ClassificationUncertaintyEvaluationMetrics`
- Old `PointEvaluationMetrics` and `UncertaintyEvaluationMetrics` retained for backward compat

**`evaluation_manager.py`**
- `transform_data()` — `else: raise ValueError` replaced with `logger.warning` + identity pass-through. Method marked deprecated. HydraNet unblocked.
- `__init__()` — `metrics_list` parameter removed. Metrics come from config. **Breaking API change.**
- `_normalise_config()` — new static method. Translates legacy keys with loud warning.
- `_validate_config()` — new static method. Raises `KeyError` immediately on incomplete config.
- `evaluate()` — rewired. Reads task type from config, detects pred type from data shape, dispatches to correct quadrant, validates every declared metric exists in the selected dict before touching any data.
- Three evaluation methods (`step_wise_evaluation`, `time_series_wise_evaluation`, `month_wise_evaluation`) — refactored to accept explicit `metrics_list`, `metric_functions`, `metrics_cls` parameters instead of deriving them from `self.metrics_list` and `is_uncertainty`.

**Tests**
- All 58 existing tests updated to new config schema and `EvaluationManager()` API
- 12 new tests added: config normalisation, legacy key warnings, missing key errors, cross-task metric rejection, four canonical dict membership tests
- Final result: **70/70 passing, ruff clean**
- `pyproject.toml` bumped to `0.4.0`

### 2.5 Integration fixes — views-pipeline-core

Three successive errors after switching to the feature branch:

**Error 1: `TypeError: EvaluationManager.__init__() takes 1 positional argument`**

`_evaluate_prediction_dataframe` in `model.py` was still calling `EvaluationManager(metrics_to_use)`. Fixed by Simon: `EvaluationManager()`, with `tasks` simplified to target lists only and `self.configs` passed directly to `evaluate()`.

**Error 2: `KeyError` on `self.configs["targets"]`**

Line 2707 still read actuals using the legacy `"targets"` key after the config had migrated to `regression_targets`/`classification_targets`. Fixed by Simon:
```python
all_targets = (
    self.configs.get("regression_targets", []) +
    self.configs.get("classification_targets", [])
)
df_actual = df_viewser[all_targets]
```

**Error 3: `ValueError: Predictions[0] must contain exactly one column, but found 15`**

`evaluate()` was being called with the full wide prediction DataFrame (all targets as columns). `validate_predictions` correctly rejected it — the evaluator contract requires exactly one `pred_{target}` column per DataFrame. Fixed by Simon: slice both actuals and predictions to the specific target before calling `evaluate()`:
```python
df_actual[[target]],
[df[[f"pred_{target}"]] for df in raw_preds],
```

---

## 3. Why We Did It

**Immediate:** Unblock HydraNet, which had been unable to run evaluation due to the `ValueError` crash on unrecognised prefixes.

**Architectural:** Remove the fundamental design flaw — an evaluator that carries domain knowledge (transformation spaces, binarisation thresholds, target semantics) it should never have had. This created a closed-world assumption: any model that didn't conform to the evaluator's internal whitelist was simply rejected.

**Preventative:** Eliminate an entire class of silent errors. `AP` with `threshold=25` applied to a binary classification target (where all values are ≤ 1) produces an AP score of 0 or undefined with no warning. Under the new architecture this is a hard error at config validation time.

---

## 4. Who

**Simon Polichinel von der Maase** — lead architect, final decisions on all design questions, all `views-pipeline-core` fixes, final testing against HydraNet.

**Claude Sonnet 4.6** — analysis, architectural debate, implementation of the `views-evaluation` refactor, test suite updates.

---

## 5. What We Learned

### On the architecture

**The 2×2 matrix is the right abstraction.** Separating task type (what the target represents) from prediction type (what format the predictions are in) cleanly handles every combination the pipeline currently produces and anticipates future ones. It also makes the evaluator's responsibilities precisely statable.

**"Models return on the original scale" is a stronger contract than it first appears.** It eliminates not just the `transform_data` problem but the entire category of evaluation-time transformation logic. The manifesto's Phase 2 `prepare_predictions_for_evaluation` hook became unnecessary once this was stated clearly.

**Silent errors are worse than crashes.** The old `threshold=25` default in `calculate_ap` had been in production producing meaningless numbers for binary targets without anyone noticing. A crash at least surfaces the problem.

### On tooling

**Git worktrees and editable installs do not mix well for active development.** The worktree was useful for the initial branch review (diff, read, explore) but became friction once we moved to implementation — the editable install pointed at the main checkout, not the worktree. For future sessions: use worktrees for read-only review, work directly in the main checkout for implementation.

### On process

**Read the full source before proposing fixes.** The manifesto's original plan missed three of the four overreach sites because it was written after reading only the crash traceback, not the full `evaluation_manager.py`. The extended analysis in this session found the `calculate_ap` threshold, the `convert_to_array` coupling, and the `pred_` convention issue — all by reading the complete file.

**The config is the contract.** Moving from implicit inference (prefix → transformation → metric space) to explicit declaration (config → task type → metric functions) made every failure mode visible and every valid combination statable. This is a pattern worth applying elsewhere in the pipeline.

---

## 6. What Remains (Deferred)

These items are explicitly out of scope for this session and tracked for Phase 2:

| Item | Blocker |
|---|---|
| Remove `transform_data` from `_process_data` | All legacy model repos must first be confirmed to return predictions on original scale |
| Remove deprecated `POINT_METRIC_FUNCTIONS` / `UNCERTAINTY_METRIC_FUNCTIONS` aliases | Downstream callers must be identified and migrated |
| Remove deprecated `PointEvaluationMetrics` / `UncertaintyEvaluationMetrics` dataclasses | Same as above |
| Investigate `lx_` formula bug (`exp(x) - exp(100)`) | Separate investigation — likely produces astronomically wrong numbers for any active `lx_` target |
| `calculate_ap` threshold migration for models passing continuous predictions | Requires per-model config update and testing |
| `convert_to_array` / metric function decoupling | Phase 3 — metric functions should accept plain arrays, not DataFrames with array-valued cells |
