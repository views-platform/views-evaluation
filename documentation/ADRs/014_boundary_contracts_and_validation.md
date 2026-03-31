# ADR-014: Boundary Contracts and Configuration Validation (The "Handshake" Rule)

**Status:** Accepted  
**Date:** 2026-02-25  
**Deciders:** Project maintainers  
**Consulted:** —  
**Informed:** All contributors  

---

## Context

Complex systems fail most often at boundaries: between modules, between configuration and runtime, and between data producers and consumers. Ambiguous configuration and hidden defaults introduce silent semantic drift.

To preserve architectural integrity and fail-loud guarantees (ADR-012), all internal and external boundaries must be explicit and validated.

## Decision

This repository adopts the following invariants:

> **All architectural boundaries must declare explicit contracts. All configuration must be validated at entry. No semantic defaults may exist silently.**

### 1. Boundary Contracts
Every boundary between components (e.g., Adapter → Core) must define:
- Explicit input schema (e.g., `EvaluationFrame` shape).
- Explicit output schema.
- Declared invariants.

### 2. Validation at Entry
All configuration and external inputs must be validated at the system boundary (e.g., in `EvaluationManager` or `Adapters`).
- Before execution begins.
- Before orchestration proceeds.

The system must fail early if required fields are missing, types are incorrect, or invariants are violated.

### 3. Failure Semantics
Validation failures must be logged and raised explicitly (ADR-013). Warnings are insufficient for structural configuration errors.

## Consequences

### Positive
- Eliminates hidden configuration drift.
- Reduces boundary fragility.
- Strengthens fail-loud guarantees.

### Negative
- Requires explicit schemas or validation logic.
- Increases up-front configuration clarity requirements.
