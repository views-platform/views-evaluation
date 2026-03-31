# ADR-023: Technical Risk Register

**Status:** Accepted  
**Date:** 2026-03-31  
**Deciders:** Project maintainers  
**Consulted:** —  
**Informed:** All contributors  

---

## Context

As the views-evaluation codebase matures through its EvaluationFrame refactor and metric catalog implementation, structural risks have been identified through repo-assimilation and expert review. Without a centralized, living register of these risks, concerns are scattered across reports, post-mortems, and tribal knowledge.

A formalized risk register ensures that architectural concerns are:
- tracked with consistent metadata,
- prioritized by severity,
- linked to their source of discovery,
- and revisited systematically.

---

## Decision

This repository maintains a **Technical Risk Register** at `reports/technical_risk_register.md` as a first-class governance artifact.

### Concern Format

Each entry uses:
- **ID:** `C-xx` for concerns, `D-xx` for disagreements
- **Tier:** 1 (critical) through 4 (informational)
- **Trigger:** The specific circumstance under which the risk becomes actionable
- **Source:** How the concern was identified (e.g. repo-assimilation, expert review, falsification audit)

### Tier Definitions

| Tier | Severity | Response |
|------|----------|----------|
| 1 | Critical — blocks release or causes data corruption | Must be resolved before next release |
| 2 | High — significant architectural risk | Must have a mitigation plan within one sprint |
| 3 | Medium — known weakness, bounded impact | Track and address opportunistically |
| 4 | Low/Informational — minor or cosmetic | Document and revisit during tech debt cleanup |

### Lifecycle

- Concerns are opened during expert reviews, tech debt audits, repo-assimilation, and falsification audits.
- Concerns are closed when the risk is resolved, mitigated, or explicitly accepted with rationale.
- The register header tracks the total count for quick reference.

---

## Consequences

### Positive
- Centralized visibility of all known risks
- Consistent prioritization and tracking
- Prevents risks from being forgotten between conversations

### Negative
- Requires discipline to keep updated
- Risk of register staleness if not reviewed regularly

---

## References

- `reports/technical_risk_register.md`
- Repo-assimilation output (2026-03-31)
- `reports/technical_debt_backlog.md` (related but focuses on actionable debt, not structural risks)
