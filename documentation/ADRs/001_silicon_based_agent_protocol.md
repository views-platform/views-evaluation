# ADR-001: Silicon-Based Agent Protocol

**Status:** Accepted  
**Date:** 2026-02-25  
**Deciders:** Project maintainers, Gemini CLI  

---

## Context

This repository is modified with the assistance of **silicon-based agents** (SBAs) like Gemini CLI. These agents optimize for local plausibility, but the Evaluation repo requires global correctness and bit-wise parity for all metrics.

Without explicit guardrails, SBAs introduce risks like silent truncation, inventing framework-coupling (e.g., adding `pandas` into a math kernel), or misinterpreting architectural intent.

## Decision

Silicon-based agents are treated as **untrusted contributors**. They are permitted to assist in modification only under the following mandatory constraints:

1.  **Verification-First Mandate**: No SBA-generated refactor may be committed without empirical verification. SBAs must run the **Green/Beige/Red** parity suite (ADR-020) after every non-trivial change.
2.  **Architectural Guardrails**: SBAs are strictly forbidden from adding framework imports (`pandas`, `xarray`, `polars`) into Level 0 modules (`views_evaluation/evaluation/`).
3.  **Fail-Loud Implementation**: SBAs must prioritize raising explicit exceptions over adding warnings or implicit fallbacks.
4.  **Shadow Runs**: For critical refactors, SBAs must implement a Shadow Run harness (running legacy and native paths side-by-side) before the legacy path is removed.

Carbon-based review (by project maintainers) remains the final gate for all SBA-assisted changes.

## Consequences

### Positive
- Prevents "architectural erosion" during high-velocity automation.
- Forces agents to prove correctness rather than just propose code.

### Negative
- Slower agent-driven iteration (due to mandatory verification loops).
- Increases token-cost and tool-call overhead for SBAs.
