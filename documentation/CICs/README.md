# Class Intent Contracts README

This directory contains **Intent Contracts** as defined in ADR-021.

An Intent Contract is a human-readable, unambiguous declaration of:

- what a non-trivial class is meant to do,
- what it must never do,
- its invariants,
- and its failure semantics.

Intent Contracts are architectural artifacts.
They are not implementation documentation.

---

## When Is an Intent Contract Required?

An Intent Contract is mandatory for:

- Core domain classes
- Architectural boundary classes
- Orchestration components
- State-owning components
- Classes that enforce invariants
- Classes that modify semantics or transformation

Trivial value objects and pure utility functions do not require one.

---

## Structure of an Intent Contract

Each contract must define:

1. Purpose
2. Responsibility Boundary
3. Invariants
4. Explicit Non-Responsibilities
5. Failure Semantics
6. Observable Effects (if applicable)

Contracts must be clear enough that:

- Tests (ADR-020) can be derived from them.
- Architectural violations can be detected.
- Silicon-based agents cannot reinterpret intent (ADR-001).

---

## Active Contracts

- `EvaluationFrame.md` — Canonical NumPy data container
- `NativeEvaluator.md` — Pure math evaluation engine
- `EvaluationReport.md` — Structured result container
- `MetricCatalog.md` — Genome registry and parameter resolver
- `PandasAdapter.md` (PHASE-3-DELETE) — DataFrame bridge

---

## Governance Relationship

Intent Contracts are governed by:

- ADR-021 (Intent Contracts for Classes)
- ADR-012 (Authority over Inference)
- ADR-020 (Multi-Perspective Testing)
- ADR-042 (Metric Catalog)

If a class changes meaning, its Intent Contract must be updated.
