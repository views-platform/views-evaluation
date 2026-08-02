# Logging & Observability Standard

**Status:** Active  
**Governing ADRs:** ADR-013 (Observability and Explicit Failure), ADR-020 (Multi-Perspective Testing)  

---

## 1. Purpose

This document defines operational standards for:

- Logging behavior
- Log levels
- Error propagation patterns
- Alerting and observability expectations

This standard operationalizes:

> Structural failures must be raised explicitly and logged persistently. (ADR-013)

It does not redefine architectural principles.

---

## 2. Core Principles

### 2.1 Fail Loud and Persist

- Structural failures must:
  - be logged at `ERROR` or higher
  - be raised as exceptions
- Logging is not a substitute for raising.
- Raising is not a substitute for logging.

Silent degradation is prohibited.

---

### 2.2 Logs Must Support Understanding

Logs must:
- provide sufficient context to reconstruct state
- include relevant identifiers (run_id, model_id, stage, etc.)
- avoid ambiguity

Logs must not:
- rely on implicit assumptions
- require tribal knowledge to interpret

---

### 2.3 Logs Must Not Leak Sensitive Data

- Secrets must never be logged.
- Credentials must never be logged.
- Sensitive raw inputs must not be logged unless explicitly approved.

---

## 3. Log Levels (Normative Definitions)

We adopt the following level semantics:

### DEBUG
- Development diagnostics.
- Detailed internal state.
- Must not be required to understand production failures.

### INFO
- High-level lifecycle events.
- Start/finish of major stages.
- Model identifiers and configuration summaries.

### WARNING
- Unexpected but recoverable conditions.
- Degraded behavior that does not violate invariants.
- Must not mask structural errors.

Warnings must not be used to hide invariant violations.

### ERROR
- Structural failure within a component.
- Operation failed and cannot proceed correctly.
- Must be raised and logged.

### CRITICAL
- System-wide failure.
- Corruption, irrecoverable state, or orchestration breakdown.
- Immediate attention required.

---

## 4. Error Propagation Pattern

Structural errors must follow this minimal pattern:

1. Construct a clear, descriptive error message.
2. Log the error (`ERROR` or `CRITICAL`).
3. Raise the appropriate exception with the same message.

Example:

```python
err_msg = "Run type not specified; cannot proceed."

logger.error(err_msg)

raise ValueError(err_msg)
````

Spacing conventions are not mandated.
Clarity and consistency are.

---

## 5. Logging Scope Expectations

### 5.1 Required Logging

The following must be logged:

* Pipeline stage transitions
* Model training start/finish
* Data loading and validation outcomes
* Configuration summaries
* All structural failures

> **Scope note (revised 2026-08-02):** logging responsibility in this repository splits by architectural level (ADR-011). The split is stated positively below so it cannot be re-derived by guesswork as new components are added — the previous wording named three classes and predated `MetricFrame`, which left Level 1 uncovered by omission rather than by decision.
>
> **Level 0 — exempt. Must NOT maintain loggers.**
> `evaluation_frame.py`, `native_evaluator.py`, `evaluation_report.py`, `metric_catalog.py`, `native_metric_calculators.py`.
> These are pure math and pure registry. They compute and validate; they do not act on the world. They rely on exception propagation per ADR-013, and logging responsibility sits at the orchestration layer (e.g. `views-pipeline-core`). Adding a logger here is a violation, not an improvement: it would duplicate what the orchestrator already records and put I/O in the numeric core.
>
> **Level 1 — must log at `ERROR` before raising.**
> `metric_frame.py`, and the `EvaluationReport.to_metric_frame()` emit path.
> These produce the **evaluation-of-record**: a persisted, cross-repo audit artifact written to disk by `MetricFrame.save()`. There is no orchestrator between this code and the filesystem, and a failure here means an audit record was not written or was written wrong. That must leave a trace independent of whether the caller catches the exception.
>
> **The `to_metric_frame()` exception.** The emit path's `ImportError` guard physically resides in `evaluation_report.py`, a Level-0 file. It logs anyway, because logging follows the **emit path**, not the file. This is deliberate and is not a precedent for logging elsewhere in that module — no other raise in `evaluation_report.py` logs.
>
> **Rule for new components:** a component logs if it performs or persists an observable external effect. If it only computes a value and hands it back, it does not.

### 5.2 Optional Logging

* Intermediate tensor shapes (DEBUG)
* Performance metrics during experimentation
* Detailed internal diagnostics

---

## 6. Log Structure and Context

Log entries should include:

* Timestamp
* Level
* Module or component name
* Relevant identifiers (run_id, model_name, etc.)

Structured logging (JSON or key-value format) is recommended where possible.

---

## 7. Alerting

Alerting is an operational layer built on logging.

At minimum:

* `ERROR` and `CRITICAL` logs must be alertable.
* `CRITICAL` logs must escalate.
* Alert routing must avoid noise amplification.

Alert configuration (Slack, email, orchestration tools) is operational and may evolve.

---

## 8. Testing Requirements

Logging behavior must be testable where meaningful.

Tests should verify:

* Errors are both logged and raised.
* Log level separation works as expected.
* Alerts trigger on configured severity thresholds.

Logging tests must not rely on manual inspection.

---

## 9. Anti-Patterns (Prohibited)

* Swallowing exceptions without logging
* Logging and continuing after invariant violation
* Downgrading errors to warnings to “keep things running”
* Using `print()` for structural diagnostics
* Logging entire objects without context

---

## 10. Evolution

This document may evolve independently of ADRs.

If logging semantics change in a way that affects system meaning,
ADR-013 must be revisited.


