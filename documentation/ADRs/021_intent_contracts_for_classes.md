# ADR-021: Intent Contracts for Classes

**Status:** Accepted  
**Date:** 2026-02-25  
**Deciders:** Project maintainers, Gemini CLI  

---

## Context

As the repository evolves, classes tend to accumulate implicit responsibilities and undocumented assumptions. Tests verify current behavior, but they do not preserve *intent*.

To prevent semantic drift, non-trivial classes require an explicit declaration of intent.

## Decision

All **non-trivial and substantial classes** (e.g., `EvaluationFrame`, `NativeEvaluator`, `PandasAdapter`) must have an explicit **intent contract**.

An intent contract is a short, human-readable description of:
- **Purpose**: what the class is for.
- **Non-goals**: what the class explicitly does *not* do.
- **Inputs and assumptions**: what it expects to be true.
- **Outputs and guarantees**: what it promises in return.
- **Failure behavior**: how it fails when assumptions are violated.

The contract must live as a clearly marked docstring or markdown file referenced from the code.

## Relationship to Tests

Intent contracts and tests must agree.
- Tests should reflect the declared intent.
- Changes to intent require updating the contract.
- Changes that violate the declared intent are bugs, not refactors.

## Consequences

### Positive
- Preserves architectural intent over time.
- Makes refactoring safer and more principled.
- Reduces cognitive load for reviewers.

### Negative
- Requires additional upfront thought and writing.
