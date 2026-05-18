# ADR-020: Multi-Perspective Testing (Green/Beige/Red)

**Status:** Accepted  
**Date:** 2026-02-25  
**Deciders:** Project maintainers  
**Consulted:** —  
**Informed:** All contributors  

---

## Context

Failure in a forecasting evaluation system is not just a crash; it is **silent semantic drift** or over-confidence in misaligned data. Standard unit tests covering only "happy paths" are insufficient for architectural shifts.

## Decision

This repository treats **testing as mandatory critical infrastructure**. All non-trivial refactors must be covered by a **three-perspective parity campaign**:

### 1. 🟩 Green Team Tests (Happy Path)
- **Goal**: Confirm bit-wise parity for standard, clean workloads.
- **Standard**: Zero divergence permitted for deterministic metrics.

### 2. 🟫 Beige Team Tests (Realistic Messiness)
- **Goal**: Catch failures caused by ragged sequences, missing months, or sparse targets.
- **Requirement**: Reproduce or explicitly document legacy behavior (e.g., Finding 4's truncation logic).

### 3. 🟥 Red Team Tests (Adversarial/Hostile)
- **Goal**: Prove fail-loud correctness under coordinate mismatches, NaNs in index levels, or inconsistent sample lengths.
- **Mindset**: "How can we break the adapter or the core?"

## Enforcement Rules
- No new path (e.g., `NativeEvaluator`) becomes the default until **total parity** is proven across all three teams.
- Divergence is permitted **only** when a legacy bug is identified and documented.

## Consequences

### Positive
- Prevents silent "alignment traps" during architectural shifts.
- Increases trustworthiness of probabilistic metrics.

### Negative
- Slower implementation phase due to mandatory parity harness.
- Higher up-front development cost.
