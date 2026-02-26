# ADR-000: Use of Architecture Decision Records (ADRs)

**Status:** Accepted  
**Date:** 2026-02-25  
**Deciders:** Project maintainers, Gemini CLI  

---

## Context

The Views Evaluation repository sits at the intersection of evolving research (new metrics, probabilistic scaling) and production stability (Pipeline Core integration). 

Significant decisions in such systems are often made under uncertainty and revisited later, leading to regressions or duplicated debate. Without a shared record of *why* decisions were made, we risk:
- Accidental reversals of critical design choices (e.g., re-introducing Pandas into the math core).
- Losing institutional memory as contributors and agents change.

## Decision

We will use **Architecture Decision Records (ADRs)** to document all significant technical, architectural, and conceptual decisions.

- ADRs are stored in the repository under `documentation/ADRs/`.
- ADRs are numbered sequentially and represent a decision, not just a discussion.
- ADRs and code must agree; code that violates an ADR is considered an architectural defect.
- If a decision changes, it is **superseded** by a new ADR, never erased.

## Consequences

### Positive
- Clearer decision-making and fewer repeated debates.
- Easier onboarding for both carbon-based and silicon-based contributors.
- Better long-term coherence through the "Pure Math Engine" refactor.

### Negative
- Small upfront cost in writing and discipline to maintain.
- Forces explicitness where ambiguity may feel easier.
