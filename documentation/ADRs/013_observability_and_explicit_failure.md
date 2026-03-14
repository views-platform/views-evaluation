# ADR-013: Observability and Explicit Failure (The "Fail-Loud" Rule)

**Status:** Accepted  
**Date:** 2026-02-25  
**Deciders:** Project maintainers, Gemini CLI  

---

## Context

This repository supports systems where silent failure, degraded semantics, or partial execution can cause cascading downstream impact. Stack traces alone are insufficient for traceability in distributed or long-running pipelines.

To preserve architectural integrity and post-hoc auditability, failures must be both:
- **explicitly raised**, and
- **persistently recorded**.

## Decision

The repository adopts the following invariant:

> **Structural failures must be both logged persistently and raised explicitly.**

### 1. Explicit Failure
- Invariant violations must raise exceptions.
- Structural failures must not be downgraded to warnings.
- Errors must not be silently swallowed.
- Fallback behavior must not hide semantic failure.

### 2. Persistent Observability
- Raised structural failures must be logged at `ERROR` level or higher.
- Critical system-wide failures must be logged at `CRITICAL`.
- Logging must occur before or at the point of raising.

## Scope

This ADR applies to:
- data validation failures,
- configuration inconsistencies,
- semantic ambiguity,
- broken invariants,
- orchestration breakdowns.

It does not prescribe formatting, spacing, or specific logging utilities.

## Consequences

### Positive
- Persistent traceability of structural failures.
- Reduced debugging entropy.
- Strong alignment with the fail-loud invariant (ADR-012).

### Negative
- Slight increase in boilerplate.
- Requires discipline in error handling.
