# ADR-011: Topology and Dependency Rules (The "Uncle Bob" Rule)

**Status:** Accepted  
**Date:** 2026-02-25  
**Deciders:** Project maintainers  
**Consulted:** —  
**Informed:** All contributors  

---

## Context

In complex evaluation systems, architectural fragility often emerges not from incorrect logic, but from uncontrolled dependencies between components.

The Evaluation repository pre-Feb 2026 suffered from "Pandas-heavy" coupling. Higher-level logic (EvaluationManager) depended on Pandas `MultiIndex` internals for alignment, which constrained our ability to scale probabilistic forecasts (N, S) due to memory/performance limits of Pandas' "lists-in-cells."

Without explicit topology rules, we risk high-level math modules beginning to depend on implementation details (e.g., NumPy indexing vs Xarray coordinates).

## Decision

This repository enforces a **strict, directional dependency structure**.

> **The Evaluation Core must have ZERO knowledge of external data frameworks (Pandas, Polars, Dask, etc.).**

Dependency direction is part of the system’s structural integrity.

Violations are architectural defects.

## Layering Principle

The Evaluation Core is the lowest-level layer (most stable). 

- **Level 0: Evaluation Core** (Pure NumPy, `EvaluationFrame`, `NativeEvaluator`). No external imports except `numpy` and `scipy`.
- **Level 1: Adapters** (Framework-specific bridges like `PandasAdapter`). May depend on Level 0.
- **Level 2: Orchestration** (e.g., `EvaluationManager`, Pipeline Core). May depend on Level 1 and Level 0.

Dependency direction must always flow **toward the Core**.

## Forbidden Patterns

- Math kernels importing `pandas` or `polars`.
- `EvaluationFrame` containing anything other than NumPy arrays.
- Higher-level modules (e.g., `EvaluationManager`) passing DataFrames directly into metric functions.

If a dependency feels “convenient but wrong,” it probably is.

## Consequences

### Positive
- Improved modularity: We can replace Pandas with Polars in the future without touching the math core.
- Performance: Forces developers to think in NumPy arrays rather than DataFrame cells.

### Negative
- Requires a "middle-man" (Adapter) to convert DataFrames to `EvaluationFrame`.
- Small amount of boilerplate for simple scripts.
