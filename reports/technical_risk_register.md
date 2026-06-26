# Technical Risk Register — views-evaluation

**Last updated:** 2026-06-26
**Total open concerns:** 9
**Governing ADR:** ADR-023

---

## Causal Clusters

Root-cause groupings of the open concerns (added 2026-06-26, strategic review). Fixing the root addresses the listed symptoms together.

- **Cluster A — Fail-Loud violations / silent degradation** → **C-02, C-20, C-22.** The codebase espouses ADR-013 Fail-Loud, yet has output paths that emit empty/omitted/`nan` values unflagged. A single audit of metric/output paths (make loud, or document explicit sentinels à la MCR) addresses all three. **Highest priority** — holds both borderline-Tier-1 entries (C-02, C-20).
- **Cluster B — EvaluationManager-deletion fallout / unowned config validation** → **C-02.** Phase-3 removal of `EvaluationManager` left config validation unowned. **C-02 is the keystone** (also in Cluster A). _(The dangling-docstring symptom, formerly C-21, was demoted to the tech-debt backlog on 2026-06-26.)_
- **Cluster C — scipy/sklearn in the Level-0 core** → **C-05, C-19.** `native_metric_calculators.py` imports scipy/sklearn at module level in a nominally pure-numpy core: ADR-011 purity violation (C-05) + undeclared packaging dep (C-19). Latent — no current trigger.
- **Cluster D — MetricFrame evaluation-of-record integrity** → **C-24, C-25, C-26.** The newly-emitted cross-repo artifact (`to_metric_frame` / `MetricFrame`) has three exposures on a young, not-yet-hardened surface: contract drift breaking consumers (C-24), a misleading provenance version-label (C-25), and float32 / `schema_version` fidelity gaps (C-26). Consumers (views-reporting, pipeline-core) are actively building against it.

---

## Open Concerns

### C-02 — NativeEvaluator does not validate config at init
- **Tier:** 2 (High) — upgraded from 3 (2026-04-04); borders Tier 1 via the silent-no-op path below
- **Description:** `NativeEvaluator.__init__` only validates the profile name. Missing or malformed config keys cause failures at evaluation time, not at construction. Two distinct failure modes exist: (a) missing target lists raise a loud `ValueError` ("Target X not found"); but (b) a **missing or misspelled metric-list key silently produces empty results with no error signal** — `_resolve_task_and_metrics` reads `config.get(f"{task}_{pred_type}_metrics", [])` (`native_evaluator.py:52`), so a typo like `regression_sample_metric` returns `[]`, and `evaluate()` returns an `EvaluationReport` with empty per-group dicts that looks successful. `config_schema.py` is a `TypedDict(total=False)` with zero runtime enforcement, and the documented validator (`EvaluationManager._validate_config`) was deleted in Phase 3, leaving config validation unowned.
- **Trigger:** A caller passes a config whose metric-list key is misspelled or omitted (e.g. `regression_sample_metric`); `evaluate()` returns an empty-but-successful-looking report instead of failing, and the misconfiguration goes unnoticed. (Also: missing `steps`/target lists surface as errors only deep inside `evaluate()`.)
- **Location:** `native_evaluator.py:28-39` (init), `native_evaluator.py:41-54` (`_resolve_task_and_metrics`, line 52), `config_schema.py` (type-only)
- **Source:** repo-assimilation (2026-03-31), upgraded 2026-04-04 (risk register review), silent-no-op path added 2026-06-24 (repo-assimilation)
- **Note:** This directly contradicts the Fail-Loud principle (ADR-013) upheld elsewhere in the codebase. See also C-21 (stale `EvaluationManager` validator reference). Part of causal clusters A + B.

---

### C-05 — sklearn/scipy in pure-math core
- **Tier:** 3 (Medium)
- **Description:** `native_metric_calculators.py` imports `sklearn.metrics` and `scipy.stats` at module level. Only 4 metric functions use these (AP, EMD, Pearson, MTD). This contradicts the zero-external-dep goal for Level 0 (ADR-011).
- **Trigger:** When someone packages views-evaluation as a minimal-dep wheel, or adds a CI/import-lint check asserting Level-0 imports only numpy — the module-level `sklearn`/`scipy` imports fail it.
- **Source:** repo-assimilation (2026-03-31); trigger sharpened 2026-06-26 (strategic review)
- **Mitigation path:** Replace with pure-numpy implementations or move affected metrics to a Level 1 module.
- **Note:** See also C-19 — the `scipy` half of these imports is also an *undeclared packaging* dependency, a separate concern from the ADR-011 purity violation tracked here. Part of causal cluster C.

---

### C-13 — No deprecation protocol for public API symbols
- **Tier:** 2 (High)
- **Description:** `EvaluationManager` and `PandasAdapter` were deleted in a single PR (PR #16) with no deprecation warning in a prior release. Downstream consumers (views-pipeline-core) had no advance signal. The Phase 4 wrapper existed as a bridge but was deleted in the same commit that merged the purge.
- **Trigger:** Any future public API symbol deletion (function, class, or module) that a downstream consumer depends on.
- **Source:** 2026-04-03 incident investigation (6/6 integration tests crashed with `ModuleNotFoundError`)
- **Mitigation path:** Before deleting any `__all__`-exported symbol, add a `DeprecationWarning` in the previous release. Require one release cycle between deprecation and removal.

---

### C-19 — `scipy` is an undeclared runtime dependency
- **Tier:** 3 (Medium)
- **Description:** `native_metric_calculators.py:6` does `from scipy.stats import wasserstein_distance, pearsonr` at module load (used by EMD and Pearson), but `pyproject.toml` declares no `scipy` dependency — only `scikit-learn`, `numpy`, and optional `pandas`. The import succeeds today only because `scipy` is pulled in transitively by scikit-learn (observed: scipy 1.15.1). The dependency is real and load-time, not lazy, so the packaging contract under-declares what the library needs.
- **Trigger:** When the dependency tree is resolved in an environment where scikit-learn is present without `scipy` (a future sklearn that vendors its math, or a constrained/locked install that satisfies sklearn from a wheel without the scipy transitive), importing `views_evaluation` raises `ImportError` at module load.
- **Location:** `views_evaluation/evaluation/native_metric_calculators.py:6`; `pyproject.toml` `[tool.poetry.dependencies]`
- **Source:** repo-assimilation (2026-06-24)
- **Mitigation path:** Declare `scipy` explicitly in `pyproject.toml`. Distinct from C-05, which concerns Level-0 purity rather than declaration. See also C-05.

---

### C-20 — Step-wise schema trusts positional `step` semantics and silently truncates under `legacy_compatibility`
- **Tier:** 2 (High)
- **Description:** The step-wise schema groups rows by the caller-supplied `step` identifier (`native_evaluator.py:119`) under the convention that step = 1-indexed positional lead time (step 1 = first month in a sequence). This is documented as a "known semantic risk, matches legacy." Compounding it, when `evaluate(..., legacy_compatibility=True)` is used, steps beyond the shortest origin sequence are **silently dropped** (`native_evaluator.py:111-122`): `max_allowed_step` is set to the minimum per-origin sequence length and longer-horizon steps are skipped with no warning. A consumer comparing models with heterogeneous sequence lengths can silently lose long-horizon evaluation rows.
- **Trigger:** When a caller evaluates data whose origins have unequal sequence lengths with `legacy_compatibility=True`, the step-wise report omits steps beyond the shortest sequence without any signal; or when `step` encodes something other than positional lead time, step groups are mislabeled.
- **Location:** `views_evaluation/evaluation/native_evaluator.py:102-128`
- **Source:** repo-assimilation (2026-06-24)
- **Mitigation path:** Log dropped steps under truncation; document/validate the positional `step` contract at the `EvaluationFrame` boundary.
- **Note:** Part of causal cluster A (silent degradation).

---

### C-22 — Pearson returns `nan` on constant input with suppressed warning
- **Tier:** 3 (Medium) — re-tiered from 4 on 2026-06-26 (silent `nan` reachable in normal use, not only adversarial input)
- **Description:** `calculate_pearson_native` (`native_metric_calculators.py:88-94`) calls `scipy.stats.pearsonr`, which on a constant `y_true` or `y_pred` returns `nan` and emits a `ConstantInputWarning` (observed during the test run for `test_nan_metric_result_is_finite_checkable`). The `nan` flows into the `EvaluationReport` unflagged. Small or single-unit groups (a month with one unit, a short sequence) can be constant, so this is reachable in normal use, not only adversarial input.
- **Trigger:** When a per-group view (e.g. a single-unit month or a constant-truth sequence) yields constant `y_true`/`y_pred`, the Pearson metric silently records `nan` in the report rather than signalling the degenerate case.
- **Location:** `views_evaluation/evaluation/native_metric_calculators.py:88-94`
- **Source:** repo-assimilation (2026-06-24)
- **Mitigation path:** Decide on an explicit contract for degenerate Pearson groups (raise, or document `nan` as the defined sentinel like `MCR` does at lines 120-122).
- **Note:** Part of causal cluster A (silent degradation). Re-tiered 4→3 on 2026-06-26.

---

### C-24 — MetricFrame cross-repo emit contract can drift and silently break consumers
- **Tier:** 2 (High)
- **Description:** views-evaluation now owns a cross-repo emit contract — `EvaluationReport.to_metric_frame()` and the `MetricFrame` value object — that views-reporting and pipeline-core consume: the axes `(eval_type, target, metric, group_id, partition, level)`, the cross-group aggregate row keyed by `MEAN_GROUP_ID` (`"mean"`), the canonical metric tokens, the `SCHEMA_TO_EVAL_TYPE` vocabulary, and the `save`/`load` serialization format. A change to any of these silently breaks downstream (views-reporting renders "not calculated"; the `EvaluationSource` loader returns wrong rows or fails). Only the **metric-token** half is guarded by a CI drift-guard test (`tests/test_metric_frame.py`); the axes, vocabulary, `MEAN_GROUP_ID`, and serialization format are **unguarded**.
- **Trigger:** When someone renames/re-tokenizes a metric, changes the MetricFrame axes or `MEAN_GROUP_ID`, edits `SCHEMA_TO_EVAL_TYPE`, or changes `MetricFrame.save`/`load` format, without a matching update in views-reporting / pipeline-core.
- **Location:** `views_evaluation/evaluation/metric_frame.py`; `views_evaluation/evaluation/evaluation_report.py` (`to_metric_frame`)
- **Source:** review-rr strategic (2026-06-26), blind-spot analysis
- **Mitigation path:** Extend the drift-guard test to cover axes / `MEAN_GROUP_ID` / `eval_type` vocabulary / a serialization round-trip; pin the persistence convention jointly with pipeline-core + reporting; enforce `schema_version` at load. See views-reporting C-41 (token drift, consumer side) and C-26 (serialization).
- **Note:** Part of causal cluster D (MetricFrame evaluation-of-record). This repo's most consequential cross-repo surface — consumers are actively coding against it.

---

### C-25 — Provenance version-label integrity in the evaluation-of-record
- **Tier:** 3 (Medium)
- **Description:** `to_metric_frame()` stamps `scoring_code_version` from the installed package version (`default_scoring_code_version` → `importlib.metadata`), which is currently a stale `"0.4.0"` — PR #20 reverted the version bump (see C-11/C-14). A dev build of views-evaluation and the released 0.4.0 therefore stamp **identical** provenance, so the evaluation-of-record cannot identify which code produced it — undermining the auditability the MetricFrame exists to provide.
- **Trigger:** When an emitted MetricFrame is used for audit/repro and two different code states both carry `scoring_code_version="0.4.0"`; or when the 0.5.0 release (issue #23) lands and historical dev-build frames remain indistinguishable from it.
- **Location:** `views_evaluation/evaluation/metric_frame.py` (`default_scoring_code_version`); `pyproject.toml` (`version`)
- **Source:** review-rr strategic (2026-06-26), blind-spot analysis
- **Mitigation path:** Cut the 0.5.0 release (issue #23) so the label is meaningful; optionally append a git SHA when available. See also C-11 / C-14.
- **Note:** Part of causal cluster D.

---

### C-26 — MetricFrame emitted-artifact fidelity: float32 cast + unenforced `schema_version`
- **Tier:** 3 (Medium)
- **Description:** Two durability gaps in the persisted evaluation-of-record. (a) Metric values are computed in float64 but cast to **float32** for the views-frames envelope (`assert_frame_envelope` requires float32), so the stored record loses precision relative to the computed metric. (b) `MetricFrame` carries a `schema_version` marker (`"1.0.0"`) but `save`/`load` does **not enforce** it — an old frame loaded under a changed format is not rejected loudly (the cross-repo wire-contract half of views-frames **C-46** is open and is this repo's responsibility).
- **Trigger:** When a metric's precision beyond ~7 significant figures matters for a downstream decision; or when `MetricFrame.save`/`load` format changes and a previously-persisted frame is loaded without a `schema_version` mismatch error.
- **Location:** `views_evaluation/evaluation/evaluation_report.py` (`to_metric_frame`, float32 cast); `views_evaluation/evaluation/metric_frame.py` (`save`/`load`, `SCHEMA_VERSION`)
- **Source:** review-rr strategic (2026-06-26), blind-spot analysis
- **Mitigation path:** Document float32 as the defined precision of the evaluation-of-record (accept), or persist float64 alongside; enforce `schema_version` on load (views-frames C-46 open half). See also C-24.
- **Note:** Part of causal cluster D.

---

## Demoted Concerns

Moved out of the register (no correctness/reliability dimension) — tracked in `reports/technical_debt_backlog.md` §5. IDs retired, never reused.

| ID | Tier | Description | Disposition | Date |
|----|------|-------------|-------------|------|
| C-21 | 4 | Stale docstring references deleted `EvaluationManager` validator | → tech-debt backlog §5.1 (real risk stays as C-02) | 2026-06-26 |
| C-23 | 4 | Vestigial empty `adapters` package | → tech-debt backlog §5.2 | 2026-06-26 |

---

## Closed Concerns

| ID | Tier | Description | Resolution | Date |
|----|------|-------------|------------|------|
| C-01 | 2 | Duplicate dispatch registries | Dispatch dicts and `calculate_ap` alias removed in Phase 3. `METRIC_MEMBERSHIP` is the single source of truth. | 2026-04-01 |
| C-03 | 3 | Rectangular sample invariant not enforced by EvaluationFrame | Added `y_pred.ndim != 2` check to `EvaluationFrame._validate()`. Tests: `test_y_pred_1d_raises`, `test_y_pred_3d_raises`. | 2026-03-31 |
| C-04 | 3 | Pandas declared as hard runtime dependency | Both modules removed in Phase 3. Lazy imports remain in optional `to_dataframe()` only. | 2026-04-01 |
| C-06 | 4 | ~77 PHASE-3-DELETE tests as maintenance burden | All PHASE-3-DELETE test files deleted in Phase 3 (10 files, ~63 tests). | 2026-04-01 |
| C-07 | 4 | Golden-value test coverage incomplete | 17 golden-value tests in `TestGoldenValues` + 8 Brier/QS golden-value tests. AP uses sklearn as oracle. | 2026-04-02 |
| C-08 | 4 | EvaluationManager config validation diverges from MetricCatalog | EvaluationManager removed in Phase 3. NativeEvaluator is the single evaluator. | 2026-04-01 |
| C-09 | 4 | `deprecation_msgs.py` appears unused | File removed. No internal or external references found. | 2026-03-31 |
| C-11 | 3 | pyproject.toml version not bumped for breaking change | Bumped to 0.5.0 in commit `aba663c`. ⚠️ **Regressed (2026-06-26):** version reverted to 0.4.0 (PR #20); re-tracked in open issue #23 (cut 0.5.0). | 2026-04-02 |
| C-12 | 4 | `to_dataframe()` requires undeclared pandas optional dependency | Added `pandas = {version = "^1.5.3", optional = true}` and `[tool.poetry.extras] dataframe = ["pandas"]`. | 2026-04-02 |
| C-14 | 4 | Editable install metadata stale | Ran `pip install -e ".[dataframe]"` to refresh. `pip show` now reports 0.5.0. ⚠️ **Stale (2026-06-26):** now reports 0.4.0 (version reverted, PR #20); see #23. | 2026-04-04 |
| C-15 | 4 | `to_dataframe()` is the only remaining dataclass consumer | Accepted as design — `to_dataframe()` owns the dataclass-to-DataFrame path internally. No action needed unless metric catalog grows beyond 30 entries. | 2026-04-04 |
| C-16 | 3 | Metric function exceptions propagate without context | Wrapped `spec.function()` call in `_calculate_metrics()` with try/except that re-raises as `ValueError` naming the metric, task, and pred_type. Test: `test_metric_function_error_includes_metric_name`. | 2026-04-04 |
| C-17 | 4 | `max_allowed_step = 999` hardcoded sentinel | Changed to `float('inf')`. Steps >= 1000 now correctly evaluated. Test: `test_step_values_above_999_not_silently_dropped`. | 2026-04-04 |
| C-18 | 4 | No bounds validation on metric hyperparameters | Added bounds validation in `resolve_metric_params()` for `alpha`, `quantile`, `lower_quantile`, `upper_quantile` — all must be in (0, 1). Cross-validation: `lower_quantile < upper_quantile`. 7 tests in `TestResolveMetricParamsBoundsRed`. | 2026-04-04 |
| C-10 | 4 | `_guard_shapes` dead Pandas handling branches | Removed 22 lines of dead code (pandas extraction, list-in-cell handling). EvaluationFrame._validate() now rejects object-dtype arrays, making these branches unreachable. | 2026-04-04 |
