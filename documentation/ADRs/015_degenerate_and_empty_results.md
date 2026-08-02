# ADR-015: Degenerate and Empty Results (The "No Empty Success" Rule)

**Status:** Accepted  
**Date:** 2026-08-02  
**Deciders:** Project maintainers  
**Consulted:** repo-assimilation (2026-08-02), review-rr strategic (2026-08-02)  
**Informed:** All contributors, downstream consumers (views-pipeline-core, views-reporting)  

---

## Context

ADR-013 established the Fail-Loud rule: structural failures must be raised explicitly and never silently degraded. It did not, however, say **what counts as a failure when there is nothing to compute**.

That gap is not theoretical. A strategic review of the technical risk register on 2026-08-02 found that seven of fifteen open concerns shared a single root cause: with no doctrine for degenerate or empty results, each site improvised independently. The register groups them as **Causal Cluster A**, and it holds the register's only Tier-1 entry.

The improvisations that resulted, all empirically verified on 2026-08-02:

| Situation | Improvised behaviour |
|---|---|
| Misspelled metric-list config key | Returns `{'month100': {}, 'month101': {}}` — an empty report that looks successful |
| Missing `steps` config key | Omits the entire step-wise schema |
| `Ignorance`, observation above the top bin edge | `IndexError` on ordinary country-month data |
| `Ignorance`, observation below the bottom edge | Negative-indexes into the **wrong bin**, returns a plausible number |
| `Pearson`, constant input | Returns `nan` into the report, unflagged |
| `to_metric_frame()` on an empty report | Emits a structurally valid **zero-row** evaluation-of-record |
| `legacy_compatibility=True` | Returns `{}` for steps the caller explicitly requested |
| `MCR`, zero-truth group | Returns `inf` or `nan` — **documented and tested** |

The last row differs in kind from the others, and that difference is the substance of this ADR.

The urgency is downstream. `EvaluationReport.to_metric_frame()` produces the **evaluation-of-record** consumed by views-reporting and views-pipeline-core. An empty-but-valid audit artifact is indistinguishable from a real one, so an upstream config typo becomes a clean-looking, permanently archived record of nothing.

---

## Decision

> **A computation that cannot produce a result must raise. It must never return a value that represents the absence of a result.**

A returned number is permitted **only when the number is itself the answer**, not when it stands in for one.

### The test for a permitted exception

A non-raising return is justified only if **all** of the following hold. Any exception must satisfy this test in writing, in this ADR.

1. **The degenerate case is a property of the data, not a broken invariant.** This is the load-bearing criterion. Ask: *did something go wrong, or is the input simply like this?* A malformed config is a fault. A group where every unit is zero is a fact about conflict data. Faults raise; facts may be recorded.
2. **The value is contracted** — documented in the function's docstring and in the relevant Intent Contract (ADR-021).
3. **The value is tested** — a test asserts the sentinel for the degenerate input.
4. **A consumer can distinguish it** from a computed ordinary result, or the distinction does not change any decision.

Failing any of the four, the correct behaviour is to raise.

> **Note on criterion 1 (revised 2026-08-02).** This criterion originally read *"the value
> is mathematically defined for the input, not a placeholder chosen because no answer
> exists."* That phrasing was wrong, and it produced a wrong ruling — see the reversal of
> ruling 2 below. It invited the question *"is this number a real answer?"*, which sorts
> cases by the **arithmetic** of the degenerate result rather than by **what caused it**.
> That is the wrong axis. `MCR`'s `inf` and `Pearson`'s `nan` arise from the same kind of
> event — data with no variation — and any test that separates them is sorting on an
> irrelevance. What actually matters is whether the evaluator is looking at a fault it
> should refuse to proceed past, or at data that is simply shaped that way.

### What this rule forbids

Restating the governing doctrine so it cannot be reinterpreted:

- **No silent fixes** to keep a run moving.
- **No repaired values.** Clamping an out-of-range input into range, substituting a neighbouring bin, or defaulting a missing parameter are all forbidden. *It is not the evaluator's job to fix values.*
- **No magic numbers or strings** introduced to make a degenerate case tractable.
- **No default values** standing in for absent configuration.
- **No fallbacks.**
- **No warnings in place of raising.** ADR-013 already forbids downgrading structural failures to warnings; the Logging & Observability Standard §9 lists it as a prohibited anti-pattern. This ADR does not reopen that.

### Rulings

Each path enumerated in the Context is ruled on individually below. These rulings are binding.

| # | Path | Ruling | Basis |
|---|---|---|---|
| 1 | `MCR` on a zero-truth group | **Documented sentinel** — `inf` / `nan` retained | Passes all four criteria; see R1 |
| 2 | `Pearson` on constant input | **Documented sentinel** — `nan` retained | ⚠ **Reversed 2026-08-02**; see R2 |
| 3 | `MetricFrame.values` permitting `NaN` | **Permitted, unchanged** | See R3 |
| 4 | Unknown / misspelled config key | **Raise at construction** | Direct application of the rule |
| 5 | Missing `steps` config key | **Raise at construction** | Direct application of the rule |
| 6 | Zero-row `to_metric_frame()` emit | **Raise at emit** | Direct application of the rule |
| 7 | `legacy_compatibility` truncating requested steps | **Omit the key** (revised) | ⚠ Reversed 2026-08-02; see R7 |
| 8 | `Ignorance` observation outside the bin range | **Raise**, both tails | See R8 |

---

## Rationale

**R1 — `MCR` keeps its sentinel.** `MCR = mean(y_pred) / mean(y_true)`. A zero-truth group is a property of conflict data, not a fault — most units are zero most of the time — so criterion 1 holds. `inf` (predicted conflict where none occurred) and `nan` (nothing predicted, nothing observed) are both interpretable calibration statements for that group. Documented in `calculate_mcr_native`'s docstring (`native_metric_calculators.py`) and asserted by `tests/test_metric_catalog.py::TestMCR::test_zero_true_positive_pred_returns_inf` and `::test_zero_both_returns_nan`. All four criteria hold. **`Pearson` (R2) is the same case and is ruled the same way.**

**R2 — `Pearson` keeps its `nan` sentinel.** ⚠ **This ruling was reversed on 2026-08-02, the same day it was made.** Both the original reasoning and the reversal are recorded here, because the mistake is instructive and should not be repeated.

*Original ruling (wrong):* Pearson raises. Correlation requires variance in both series; on constant input it is *undefined*, so `nan` denotes the **absence** of a result rather than a result. Under criterion 1 as first written — "is the value mathematically defined?" — that failed, while `MCR`'s `inf` passed. The asymmetry was stated as: MCR's `inf` is an answer, Pearson's `nan` is a shrug.

*Why it was wrong:* the criterion was sorting on the wrong axis. It asked what the degenerate *number* is, when what matters is what **caused** it. Both cases arise from the same event — data with no variation. Splitting them apart put two identical situations in different categories on the strength of an arithmetic coincidence.

*What made the error concrete:* a "predict zero everywhere" baseline has a constant prediction series. ADR-041 states that reports exist to compare *"ensemble models against constituent models and baselines"*, so evaluating such a model is routine, not exotic. Under the original ruling, evaluating any constant baseline with `Pearson` in the metric list **aborted the entire run** — every metric, every schema — because one metric was undefined on one group. Verified 2026-08-02.

That blast radius is the real signal. Fail-loud exists to stop the evaluator proceeding past a **fault**. A baseline model is not a fault; a constant prediction series is a *finding about the model*, and an all-zero month is a *fact about conflict data*, where most units are zero most of the time. Halting an entire evaluation over it is disproportionate and makes a standard workflow impossible.

*Corrected ruling:* `Pearson` returns `nan` for a constant series, documented in `MCR`'s style and tested. Consumers treat it as "not computable for this group"; `to_metric_frame()`'s `nanmean` already excludes it from aggregates. The `ConstantInputWarning` is suppressed at the call site — with the case contracted and handled, the warning is noise.

This leaves `MCR` and `Pearson` in the **same** category for the **same** stated reason, which is the coherent outcome. Criterion 1 was rewritten accordingly (see the note under The test above).

**R3 — `MetricFrame` keeps permitting `NaN`.** This is a container invariant, not a computation. `MetricFrame.load()` must faithfully reconstruct whatever `save()` wrote, including historical frames; and the cross-group aggregate row deliberately carries `nanmean` semantics. Forbidding `NaN` in the container would break the round-trip and change a published cross-repo envelope that views-reporting already consumes, without preventing a single silent failure — because the failures are prevented at the *emit* site (ruling 6) where the context to diagnose them exists. Criterion 4 governs: the distinction is preserved upstream, so the container need not police it.

**R7 — Truncated steps are OMITTED, not raised on.** ⚠ **This ruling was reversed on 2026-08-02**, two rulings in this ADR having now been reversed for the same underlying reason. Both versions are recorded.

*Original ruling (wrong):* raise. `legacy_compatibility=True` returned truncated steps as **empty placeholder dicts** — present in the report, looking evaluated, scoring nothing, emitting no MetricFrame rows. Since `config['steps']` is a request list rather than a hint, silently not fulfilling it was judged a contract violation regardless of the flag.

*Why it was wrong:* the defect was real, but the remedy attacked the wrong thing. **`legacy_compatibility=True` is itself an explicit request to truncate** — its entire documented purpose is "cap step-wise evaluation at the shortest sequence". Raising because the caller *also* listed the steps treats one explicit instruction as a violation of another. The caller is not being silently denied; they asked for exactly this.

*What made the error concrete:* `views-pipeline-core` has one evaluation call path, and it passes both together unconditionally (`managers/evaluation/stage.py`):

```python
# legacy_compatibility=True preserves step-wise truncation to shortest
# sequence, matching the deleted EvaluationManager wrapper (C-29).
report = evaluator.evaluate(ef=ef, legacy_compatibility=True)
```

Every views-models config sets `steps = list(range(1, 37))`, so a caller whose sequences are unequal asks for 36 steps while the shortest supplies fewer — and the call raised.

**How often that happens, derived from the real partition arithmetic (2026-08-02):**

`base_origin = test[0] - 1`; sequence *i* spans months `base+i+1 … base+i+36`, intersected against the test window. `core_config_sniffer` *enforces* `test_len == time_steps + MAX_SHIFT_COUNT` = 36 + 12 = **48 months**, and `MAX_SHIFT_COUNT = 12` gives **13 sequences**. Sequence 12 ends exactly on the last actual, so all 13 are full length.

| eval_type | sequences | truncates? |
|---|---|---|
| `standard` (default), `live` | 13 | **No** — equal lengths. Checked across all 128 models × 2 run types: **0 of 256** |
| `long` | 37 | Yes — shortest sequence is 12, so 24 of 36 steps drop. **256 of 256** |

`eval_type=long` appears nowhere outside pipeline-core's own arg-parser tests.

**So the raise was latent, not live.** An earlier version of this ADR stated it broke "every model in the platform, deterministically" — that was **wrong**, asserted from a fabricated unequal-sequence fixture rather than derived from the partition configs. Corrected here. The ruling is still reversed, for the reason below, which does not depend on frequency.

*Corrected ruling:* a step that truncation or the data cannot supply is **omitted from the report**. This fixes the original C-20 defect — an absent key cannot masquerade as an evaluated one — without breaking the caller. Applies with the flag off too: a configured step with no data is omitted rather than emitted empty. Confirmed compatible with the consumer: `log_wandb_log_dict` iterates `step_wise.keys()` and assumes no fixed step set.

*The lesson, generalised:* this ADR's exception test asks whether a degenerate case is a **fault or a data property**. Ruling 7 failed a prior question — *whose* fault. Before ruling that something must raise, identify who is being told off and whether they actually did anything wrong. Here the caller had followed the documented contract exactly.

**R8 — `Ignorance` raises on out-of-range observations.** An observation outside the configured bins means the profile's `bins` do not fit the target. Two alternatives were considered and rejected under the doctrine: **clamping** into the edge bins is a silent fix that redefines the metric precisely at the tails, where conflict data matters most; **widening the base profile's ceiling** is a magic number that relocates the cliff without removing it and does nothing for the underflow. The configuration is wrong and must be corrected by the researcher.

---

## Considered Alternatives

### Alternative A: No exceptions — every degenerate case raises

Rejected. It would break `MCR`'s documented, tested contract and force the cross-repo `MetricFrame` envelope to change, for no gain in silence-prevention. The four-criteria test achieves the same guarantee while preserving genuinely defined answers.

### Alternative B: Keep all existing sentinels, document them

Rejected. It preserves `Pearson`'s `nan` — a value denoting no-answer — flowing unflagged into reports for degenerate groups. That is the failure mode C-22 identifies, and documenting a shrug does not make it an answer.

### Alternative C: Warn instead of raise

Rejected on constitutional grounds, not preference. ADR-013 forbids downgrading structural failures to warnings, and the Logging & Observability Standard §9 lists it as a prohibited anti-pattern.

---

## Consequences

### Positive

- Cluster A's root cause is removed; ten register concerns become closable.
- A misconfigured evaluation cannot masquerade as a successful one, in the report or in the archived evaluation-of-record.
- Future contributors have a written test for permissible exceptions instead of a precedent to imitate.
- Degenerate-group failures name the group, so they are diagnosable rather than mysterious.

### Negative

- **Breaking behaviour changes in a published library.** Configs that returned an empty-but-successful report will raise (rulings 4, 5); `legacy_compatibility` callers requesting steps beyond the shortest sequence will raise (ruling 7); `Ignorance` raises on observations outside its bins (ruling 8). Downstream consumers must be told before 0.5.0 ships — governed by ADR-022's deprecation policy. (Ruling 2 was *reversed* precisely because its breaking change — aborting any evaluation of a constant baseline — was disproportionate to the fault it was catching.)
- Some evaluations that previously "worked" will now stop. That is intended, but it transfers cost to callers with degenerate data, who must fix their configuration or their profile.
- The four-criteria test requires judgement. It is deliberately narrow, but it is not mechanical.

---

## Implementation Notes

- **Exception type:** `ValueError` throughout, consistent with existing config and metric failures (`resolve_metric_params` in `metric_catalog.py`; `_validate_config`, `_resolve_task_and_metrics` and `_calculate_metrics` in `native_evaluator.py`).
- **Cite symbols, not line numbers.** References in this ADR name functions, classes and test node IDs rather than line ranges, because line numbers drift the moment anything is inserted above them — including within the very commit that writes the citation.
- **Message quality:** every raise must name **what was wrong**, **what was expected**, and **what the caller should change**. A message that only reports a symptom fails this ADR's intent.
- **Level-0 logging exemption stands.** The Logging & Observability Standard §5.1 exempts Level-0 pure-math components from maintaining loggers; exceptions propagate to the orchestrator. Raises added under this ADR in Level-0 files must **not** introduce loggers. The Level-1 emit path (`MetricFrame`, `to_metric_frame`) does log at `ERROR` before raising.
- **Detection belongs as early as the information exists.** Config failures raise at `NativeEvaluator.__init__`, not at `evaluate()`. Emit failures raise at `to_metric_frame()`, not inside `MetricFrame._validate` — the emit site holds the context needed for a useful message, and the container must remain able to represent anything `load()` reads back.

---

## Validation & Monitoring

- Every ruling that changes behaviour carries a **Red** test per ADR-020.
- The suite must finish with **zero warnings**. A `ConstantInputWarning` reaching the runner means a degenerate case is being *encountered* rather than *contracted*; under ruling 2 it is suppressed at the call site precisely because it is handled.
- **Before ruling that a degenerate case must raise, identify WHOSE fault it is.** A raise tells the caller they did something wrong. If the caller followed the documented contract — or if the behaviour they are being blamed for is one they explicitly requested — the fault is not theirs and the raise is wrong. Ruling 7 was reversed on exactly this point: it raised at a caller who had asked for truncation, for truncating.
- **Execute the real caller's path, not a reconstruction of it.** Both reversals in this ADR were found by running an actual consumer call, never by this repo's test suite. Ruling 7 was reviewed, tested, and merged before a falsification audit ran `views-pipeline-core`'s exact invocation. See register C-33 / C-35.
- **Before ruling that a degenerate case must raise, identify the blast radius.** Rulings here apply per-metric per-group, but a raise propagates to the whole `evaluate()` call — every metric, every schema. Ruling 2 was reversed on exactly this point. Ask what legitimate workflow the raise makes impossible, and check it against a real one (for ruling 2, it was baseline comparison per ADR-041).
- Intent Contracts (ADR-021) must record each new failure mode; a documentation-contract test enforces that they stay in step.
- Sentinels permitted under this ADR must retain their asserting tests. Deleting such a test is a violation of criterion 3, not a test-cleanup decision.

---

## Open Questions

- **The positional-`step` contract.** `step` is assumed to mean 1-indexed positional lead time and is enforced nowhere; the adapter that synthesised it was removed in Phase 3. This is a cross-repo contract question, deliberately **not** settled here. Tracked as register C-20's remaining half.
- **Profile-domain validation.** Ruling 8 raises when an observation falls outside a profile's bins, but nothing validates at *registration* that a profile's bins suit its intended target. A profile-level domain declaration would move the failure earlier still. Out of scope here.

---

## References

- ADR-013 (Observability and Explicit Failure) — the parent rule this ADR completes
- ADR-014 (Boundary Contracts and Validation) — where validation belongs
- ADR-020 (Multi-Perspective Testing) — Red/Beige/Green obligations
- ADR-021 (Intent Contracts for Classes) — where failure semantics are recorded
- ADR-022 (Evolution and Stability) — governs announcing the breaking changes above
- `documentation/standards/logging_and_observability_standard.md` §3, §5.1, §9
- `reports/technical_risk_register.md` — Causal Cluster A; C-02, C-20, C-22, C-27, C-28, C-30
