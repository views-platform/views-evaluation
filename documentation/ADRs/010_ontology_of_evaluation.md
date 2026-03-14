# ADR-010: Evaluation Ontology

**Status:** Accepted  
**Date:** 2026-02-25  
**Deciders:** Project maintainers, Gemini CLI  

---

## Context

Pre-Feb 2026, the Evaluation repo operated on an implicit ontology where "Actuals" and "Predictions" were DataFrames with implicit "step" and "sequence" structures. This led to "lists-in-cells" and fragile row-wise iteration.

Without an explicit ontology, systems tend to accumulate implicit concepts and overloaded responsibilities (e.g., `EvaluationManager` doing both alignment and math).

## Decision

This repository defines a **closed set of conceptual categories** (entities) that are allowed to exist.

### Core Ontological Categories

1.  **`EvaluationFrame` (Canonical Data)**:
    - **Semantic Role**: The synchronized, pre-aligned "Pure NumPy" representation of a task.
    - **Invariant**: Must contain only primitive NumPy arrays. No objects, no DataFrames.
    - **Authority**: Authoritative. It is the only thing the math core sees.

2.  **`NativeEvaluator` (Pure Math Engine)**:
    - **Semantic Role**: A stateless engine that performs math and regrouping.
    - **Non-goal**: It does NOT know how to load, save, or align DataFrames.

3.  **`Adapters` (Framework Bridges)**:
    - **Semantic Role**: Bridges between external frameworks (Pandas, Polars, etc.) and the `EvaluationFrame`.
    - **Constraint**: Must be isolated from the math core.

4.  **`MetricCalculators` (Mathematical Kernels)**:
    - **Semantic Role**: Functional kernels that compute specific values (MSE, CRPS, etc.).
    - **Constraint**: Must be framework-agnostic and vectorized where possible.

### Explicit Non-Entities

The following are **NOT allowed** as first-class concepts in the Evaluation Core:
- DataFrames, Series, or Index levels.
- Lists inside cells.
- Implicit or inferred semantics (see ADR-012).

## Consequences

### Positive
- Clear review criteria for new abstractions.
- Performance is baked into the ontology (via contiguous NumPy arrays).

### Negative
- Requires upfront conversion (adaptation) even for small tasks.
- Some refactors may be blocked until concepts are clarified.
