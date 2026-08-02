# Changelog

All notable changes to `views_evaluation` are recorded here.

This file exists because [ADR-022](documentation/ADRs/022_evolution_and_stability.md)
requires it: §3.1 mandates that behaviour changes making previously-accepted input fail
be announced in release notes with the specific inputs that now fail, and §6 requires the
0.5.0 notes to retrospectively record the 0.4.0 removals. It did not exist until 0.5.0 —
which is itself the reason 0.4.0 shipped breaking changes that reached users unannounced.

The project follows Semantic Versioning. **Since 1.0.0 (2026-08-02) the public API is
stable and any breaking change requires a MAJOR bump.** Entries below 1.0.0 were released
under the `0.x` caveat of ADR-022 §5, where breaking changes could ship in a MINOR bump
provided they were announced here.

---

## [1.0.0] — 2026-08-02

**The API is now stable.** Under `0.x`, ADR-022 §5 allowed breaking changes in a MINOR
bump provided they were announced. That latitude ends here: from 1.0.0, **any breaking
change to the public API requires a MAJOR bump**. The public API is every symbol in
`views_evaluation/__init__.py:__all__` plus the documented behaviour of those symbols —
constructor and method signatures, return types, raised exception types, and the config
keys declared in `EvaluationConfig` — and, regardless of `__all__`, the `MetricFrame`
on-disk format and axis vocabulary, which are a cross-repo contract (ADR-022 §1).

ADR-022 §5 says 1.0.0 should be cut "when downstream consumers require a stability
guarantee — not before". Both consumers now build against this library's evaluation-of-
record: `views-pipeline-core` invokes `NativeEvaluator` directly, and `views-reporting`
reads the emitted `MetricFrame`. That is the condition, and it is met.

**There are no API changes in 1.0.0.** It contains the dependency and CI changes below
and nothing else; 0.5.0's surface carries over unaltered. Anyone on 0.5.0 can upgrade
without touching code, provided they satisfy the `views-frames` floor.

### Changed — breaking for the `frames` extra

- **`views-frames` floor raised from `^1.4` to `>=1.10.2,<2`.** This aligns
  views-evaluation with `views-pipeline-core`, which already pins `^1.10.2`, so the
  platform resolves one `views-frames` rather than negotiating a range.
  *Migration:* if you install the `frames` extra, upgrade `views-frames` to at least
  1.10.2. Anyone on 1.4–1.10.1 must move; the core package (no extra) is unaffected.
  Recorded here per ADR-022 §3.1 — narrowing an accepted dependency range is a breaking
  change for the resolvers it excludes, even though no API changed.

### Fixed — CI was not testing the cross-repo contract

- **CI now runs `poetry install --all-extras`.** It ran a bare `poetry install`, which
  installs no optional dependencies, and both `tests/test_metric_frame.py` (44 tests) and
  `tests/test_evaluation_report.py` (22 tests) guard with a **module-level**
  `pytest.importorskip`. Both files were therefore skipped whole on every pull request:
  361 tests pass locally, 293 ran in CI. The 68-test gap included every guard on
  `MetricFrame`, which ADR-022 §1 designates a cross-repo contract and public API
  "regardless of `__all__`" — and the emit path its consumers depend on.

  The skip was silent by construction: a green run reporting "293 passed" was
  indistinguishable from one that verified the contract. Every "CI green" claim made
  while shipping 0.5.0 was true and meant less than it appeared to. Guarded against
  recurrence by `tests/test_falsification_ci_coverage_gap.py`, which derives the required
  extras from `pyproject.toml` rather than hardcoding them.

- **CI now verifies the extras actually arrived**, with an explicit
  `poetry run python -c "import views_frames, pandas"` step after install. Asserting
  `--all-extras` appears in the workflow proves intent, not effect: `poetry install`
  exits 0 even when it resolves without an optional package, and every downstream guard
  is a module-level `importorskip` that turns absence into silence. Without this step the
  68-test gap above could reopen while the guard against it stayed green. Found by
  falsification audit, guarded by `tests/test_falsification_extras_actually_installed.py`.

### Release checklist (ADR-022 §7)

- [x] **Does this release do anything rule 2 governs — remove an `__all__` symbol, remove
  a supported config key, narrow an accepted input, or change a raised exception type?**
  No. 1.0.0 makes no API change of any kind; the surface is 0.5.0's, unaltered. The
  `views-frames` floor is a dependency constraint, not an accepted input, and is governed
  by rule 3 as a breaking change for excluded resolvers.
- [x] **Does this release make previously-accepted input fail? If so, are the release
  notes explicit, and have known consumers been notified?** No input changes. The
  dependency floor is enumerated above with its migration. **Both consumers verified
  directly:** `views-pipeline-core` pins `views-frames ^1.10.2` (compatible) and
  `views-reporting` is moving to `>=1.10.2,<2.0.0` on its own branch. Checking consumer
  *resolution* rather than assuming it is the specific lesson of C-38.
- [x] **Does the version bump match the change class?** Yes. `0.5.0` → `1.0.0` is MAJOR,
  which is required for the stability commitment itself and permitted for the dependency
  narrowing.
- [x] **Do the release notes list every breaking change with its migration?** Yes — one
  breaking change (the `views-frames` floor), with its migration stated.
- [x] **Does `MetricFrame`'s format or axis vocabulary change?** No. Unchanged from 0.5.0.
  From 1.0.0 it is additionally covered by the MAJOR-bump guarantee, and — as of this
  release — its 44 guards actually run in CI, which was not true when 0.5.0 shipped.

---

## [0.5.0] — 2026-08-02

The Fail-Loud Doctrine release. Paths that previously returned a value for input the
library could not actually evaluate now raise instead. Governed by
[ADR-015](documentation/ADRs/015_degenerate_and_empty_results.md), whose eight rulings are
the authoritative list. This section states the caller-visible consequence of the six that
changed behaviour. Two are not listed below, for different reasons. **R1** (`MCR` returns
`inf`/`nan` on a zero-truth group) is genuinely unchanged from 0.4.0 — a documented
sentinel, nothing to migrate. **R3** (`MetricFrame.values` may contain `NaN`) governs a
class that **did not exist before 0.5.0**, so it is a ruling about new API rather than a
change to old behaviour; `MetricFrame` is listed under Added below. Neither gives a caller
anything to migrate.

### Breaking — configuration now validated at construction

`NativeEvaluator.__init__` now validates its config, so a structurally invalid config
fails at construction instead of part-way through an evaluation. **The 0.4.0 behaviour
was not uniform**, and the "Was" column below states it per row, verified by replaying
`v0.4.0` rather than assumed. Three rows produced a silent empty report; the rest already
failed, but later, or with a message that did not name the real problem.

| Input | Was, in 0.4.0 | Now |
|---|---|---|
| Legacy `metrics` / `*_uncertainty_metrics` | **silently ignored** → task declared with no metrics → empty report | raises at `__init__`, naming the rename |
| Legacy `targets` | ignored, then raised at `evaluate()`: `Target {t} not found in config` — which blames the frame, not the config key that caused it | raises at `__init__`, naming the rename |
| A key one small edit from a real one (`regression_target`) | **silently ignored**, then behaved as whichever list it should have been is missing | raises at `__init__`, naming what it resembles — **never substituting it** |
| `steps` absent | **step-wise schema silently empty** (`{}`, not missing); month-wise and time-series-wise still fully populated, so the report looked complete | raises at `__init__` |
| `steps` empty | silently matched no group → `{}` | raises at `__init__` |
| `steps` containing `0`, negatives, or non-integers | **worse than empty: fabricated step labels.** `steps` was pre-initialised as `{f"step{str(s).zfill(2)}": {} for s in config_steps}`, so `[0]` produced `{'step00': {}}`, `[-1]` produced `{'step-1': {}}` and `[1.5]` produced `{'step1.5': {}}` — keys that look evaluated and scored nothing (the C-20 shape) | raises at `__init__` |
| `steps` a scalar, tuple, or numpy array rather than a list | scalar: bare `TypeError: 'int' object is not iterable` at `evaluate()`. numpy array: bare `ValueError: The truth value of an array … is ambiguous` | raises `ValueError` at `__init__`, naming the key and the expected type |
| No `regression_targets` and no `classification_targets` | raised at `evaluate()`: `Target {t} not found in config` | raises at `__init__`, naming the actual problem |
| A declared target list with no metric list for that task | empty per-group dicts → empty report | raises at `__init__` |
| A metric invalid for its `(task, pred_type)` cell | raised at `evaluate()`: `Metric '{m}' is not valid for ({task}, {pred_type})` | **a second, different check added at `__init__`** — it validates each metric against the cell its *config key* declares, so a bad config fails before any computation. The `evaluate()`-time check is unchanged and still fires on the cell the *frame* resolves to; the two are not the same check and neither replaces the other |
| A non-dict config | `AttributeError` immediately at `__init__`, from `config.get(...)` | raises `ValueError` naming the expected type |

**Migration**, per row. For the three key-name rows: correct the key — the library reports
what it suspects you meant and stops, it does not interpret a typo on your behalf
(ADR-015 forbids silent repair). For the four `steps` rows: supply `steps` as a `list` of
positive integers, e.g. `[1, 2, 3]`; it is 1-indexed, and `list` specifically, because
`EvaluationConfig` declares `List[int]`. For the two target/metric rows: declare a
non-empty `regression_targets` and/or `classification_targets`, and at least one metric
list for each task you declare. For the invalid-metric row: check the metric is valid for
the `(task, pred_type)` cell its key names — `METRIC_MEMBERSHIP` is the authority, and the
error message names the cell. For the non-dict row: pass a `dict`.

In every case the config is wrong and the caller fixes it; nothing is defaulted or repaired.

**An unrecognised key that resembles nothing is still ignored, deliberately.**
`views-pipeline-core` passes its whole combined model config through
(`NativeEvaluator(context.configs)`), so the dict legitimately carries dozens of foreign
keys. Rejecting unknown keys wholesale would break every model in the platform. See risk
register C-33.

### Breaking — evaluation and emit

- **A resolved `(task, pred_type)` with no configured metrics now raises at `evaluate()`**
  (ADR-015's general rule — no ruling in its table of eight covers this case). Fires when
  a config declares only `<task>_point_metrics` but the frame
  turns out to be sample-type (`n_samples > 1`). Was: empty per-group dicts.
  *Migration:* declare metrics for the prediction type your frame actually carries.
- **`to_metric_frame()` raises when no schema has any metric value** (ADR-015 R6). Was: a
  structurally valid, zero-row evaluation-of-record — an artifact that looked evaluated and
  was not. `MetricFrame` itself still accepts zero rows so `load()` can read back whatever
  `save()` wrote.
  *Migration:* there is nothing to change in the emit call — the raise means the evaluation
  that preceded it produced no values, so fix that. The message names the target and which
  schemas were present-but-empty. Usual causes: the frame carries no rows for the requested
  `steps`, or the configured metrics do not apply to the frame's `(task, pred_type)` cell.
- **`Ignorance` raises on an observation outside the configured bin range, both tails**
  (ADR-015 R8). Was: `IndexError` above the top edge (masked when a prediction was also out
  of range) and a silently wrong bin below the bottom edge.
  *Migration:* widen `bins`, or clip observations before scoring.
- **`EvaluationFrame` raises on `y_true.ndim != 1`.** Checked after the dtype gate, so a
  non-numeric array is still reported as a dtype problem.
  *Migration:* pass observations as a flat array. A column vector of shape `(n, 1)` — the
  common case, from a single-column DataFrame selection — becomes `y_true.ravel()`. Note
  `y_pred` is unaffected and remains 2-D `(n_rows, n_samples)`.

### Changed — not breaking

- **`legacy_compatibility=True`: truncated and data-less steps are now omitted from the
  step-wise report** rather than returned as empty placeholder dicts (ADR-015 R7, register
  C-20). A step that could not be evaluated is no longer presented as one that was.
  This was briefly implemented as a raise and **reversed the same day**: the flag is itself
  a request to truncate, so raising blamed the caller for what they asked for. The raise was
  latent — 0 of 256 (model, run_type) combinations truncate on the default path.
- **`Pearson` returns `nan` for constant input**, with the `scipy` `ConstantInputWarning`
  suppressed in scope. Also briefly a raise and reversed the same day: a "predict zero
  everywhere" baseline has constant predictions by construction, and ADR-041 requires
  baseline comparison, so raising aborted the run for a case the design mandates.

### Added

- **`MetricFrame` / `MetricFrameMetadata`** — the evaluation-of-record, emitted via
  `EvaluationReport.to_metric_frame()`. Exported from `views_evaluation` only when the
  optional `views-frames` extra is installed. Its on-disk format and axis vocabulary are a
  **cross-repo contract** and are treated as public API regardless of `__all__`
  (ADR-022 §1). Contract: `documentation/CICs/MetricFrame.md`.
- **ADR-015** (Degenerate and Empty Results) — when a non-raising path is permitted.
- **ADR-022** (Evolution and Stability) — activated from `Proposed (Deferred)`. Defines the
  public API surface, the one-release-cycle deprecation contract, the
  `DeprecationWarning`-versus-fail-loud boundary, and the release checklist below.
- **ADR-041** moved `Proposed` → `Accepted` and was rewritten; it had asserted that this
  library never persists to disk, which `MetricFrame.save()` falsifies.
- **`tests/test_documentation_contracts.py`** — asserts documentation makes no support
  claim the code does not honour. This is ADR-022 §7's CI enforcement point.
- **`scipy`** declared explicitly in `pyproject.toml`. It was a load-time import satisfied
  only transitively via scikit-learn.

### Fixed

- `Brier` is implemented (three variants, shipped 2026-04-09) and was documented as
  unimplemented in ADR-031 and the tech-debt backlog until this release.

### Known limitations

- **Downstream pins are current.** `views-pipeline-core` now pins
  `views-evaluation = "^0.5.0"` (widened in pipeline-core PR #379, merged before this
  release was cut); `views-reporting`'s `>=0.4.0,<1.0.0` already accepted it. Published
  `views-pipeline-core` 2.3.0 continues resolving 0.4.0, which remains on PyPI, so
  nothing already deployed moves.

  *An earlier version of this section said pipeline-core pinned `^0.4.0` and "will reject
  this release on install". That was already false when it published — the pin had been
  widened hours earlier. Corrected 2026-08-02.*
- ADR-022 §3.2 requires known downstream consumers be notified **before** the release is
  cut. Nothing in this repository can evidence that a notification happened, which is
  registered as **C-36** — the release gates are prose with no enforcement point.

---

## [0.4.0] — 2026-05-18

Recorded retrospectively, as required by ADR-022 §6. **These removals shipped with no
deprecation warning in any prior release**, and no release notes were published at the
time. All six downstream integration tests in `views-pipeline-core` crashed with
`ModuleNotFoundError`. This is the incident that activated ADR-022 (register C-13).

**On the dates.** The removal commit `fe6dd4c` was authored **2026-04-01** and merged to
**`development`** on **2026-04-03** via PR #16; the downstream crash followed within the
hour. It reached **`main` only on 2026-05-18**, in `57794af` — the very commit this
version tags. `main`'s first-parent history runs straight from PR #12 to PR #19 with
nothing in between.

So consumers broke **six weeks before the release**, because `views-pipeline-core` was
tracking the `development` branch, not a published version. The release did not cause the
incident; it shipped the cause. *(An earlier draft of this section said the removals
"landed on `main` on 2026-04-01". That was wrong — `main` was untouched until the release
itself — and it also split one event, PR #16, into two.)*

### Removed — breaking, unannounced at the time

- **`EvaluationManager`** — the deprecated orchestrator. Replaced by `NativeEvaluator`.
- **`PandasAdapter`** (`views_evaluation/adapters/pandas.py`). Callers convert to
  `EvaluationFrame` themselves; conversion belongs in the orchestrator (ADR-011).
  Note the `views_evaluation.adapters` **package itself was not removed** — an empty
  `__init__.py` remains importable to this day, tracked as the demoted C-23.
- **The legacy config-key translation shim.** It lived inside `EvaluationManager` and was
  deleted with it. The README continued to promise, until 0.5.0, that legacy keys still
  worked and emitted a `DeprecationWarning`. They did not, and no warning was ever emitted
  by the published package (register C-29).

**The shim will not be restored.** It has been absent from the published package since
2026-05-18, and after ADR-015 R4 a legacy key fails loudly rather than silently, which is
the better outcome. See ADR-022 §6 for the full retroactive position.

---

## Release checklist (ADR-022 §7)

Worked through for **0.5.0** before publish. This section is the artifact the checklist
was missing; register C-36 records that nothing yet *requires* it to be completed.

- [x] **Does this release do any of the things ADR-022 rule 2 governs — remove an `__all__`
  symbol, remove a supported config key, narrow an accepted input, or change a raised
  exception type? If so, did a `DeprecationWarning` ship at least one release ago?**
  No symbol or config key is removed in 0.5.0; the 0.4.0 removals are recorded above
  retroactively, and the deprecation contract did not exist when they shipped.
  **But yes on the other two**, and the answer is not "no" as a narrower reading of this
  item would have allowed: 0.5.0 narrows accepted input in every row of the config table,
  and changes a raised exception type (a scalar `steps` moved from a bare `TypeError` to
  `ValueError`). No `DeprecationWarning` shipped a release ago for any of them.
  **That is compliant, not a lapse**, because ADR-022 §4's boundary table rules that a
  structural failure — invalid config, broken invariant — must raise and must never warn
  (ADR-013, ADR-015). There is no warn-first path available for these. They are governed by
  §3 instead, which is the next item.
  *(This checklist item asked only about symbols and config keys until 2026-08-02. Under
  that wording this release could have answered a clean "no" and skipped the analysis
  above — the item under-covered the rule it enforces. Widened in ADR-022 §7.)*
- [ ] **Does this release make previously-accepted input fail? If so, are the release notes
  explicit, and have known consumers been notified?** Yes, it does — enumerated above with
  the specific inputs, so the release-notes half is met. **The notification half is NOT
  met.** Compatibility was *verified* (the full `views-pipeline-core` suite passes against
  this branch), which is not the same thing as notifying. Deliberately left **unchecked**:
  a ticked box asserting completion of something the text says was not completed is the
  precise failure this checklist exists to prevent. Open as C-36.
- [x] **Does the version bump match the change class?** Yes. Breaking behaviour changes
  under `0.x` require at least MINOR; `0.4.0` → `0.5.0` is MINOR.
- [x] **Do the release notes list every breaking change with its migration?** Yes — the
  tables above, cross-checked against ADR-015's eight binding rulings, and each breaking
  change carries a `*Migration:*` line. *(This answered "yes" before it was true: two of
  the four evaluation/emit changes had no migration, and the per-row migrations for the
  config table were left implicit. Both fixed 2026-08-02 — a checklist answered
  optimistically is the same failure the checklist exists to prevent.)*
- [x] **Does `MetricFrame`'s format or axis vocabulary change?** `MetricFrame` is **new** in
  0.5.0, so there is no prior format to change. Its shape has been verified end-to-end
  against `views-reporting`'s real `MetricFrameFileSource`, but that is verification, not
  the agreement the checklist asks for. Drift risk after this point is tracked as C-24.
