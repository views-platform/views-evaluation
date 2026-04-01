# Instantiation Checklist

Use this checklist to track the base_docs governance adoption for views-evaluation.

---

## Before You Start

- [x] Decide which adoption phase you're targeting
- [x] Identify your project's ontological categories (ADR-010)

---

## ADR Adaptation

### All adopted ADRs
- [x] Update Status from `--template--` to `Proposed` or `Accepted`
- [x] Fill in Date, Deciders, Consulted, Informed fields

### Per-ADR adaptation notes
- [x] **ADR-000:** Updated path reference to `documentation/ADRs/`
- [x] **ADR-010 (base 001):** Defined project's ontological categories (EvaluationFrame, NativeEvaluator, etc.)
- [x] **ADR-011 (base 002):** Defined 3-level layering and forbidden dependency patterns
- [x] **ADR-012 (base 003):** Adapted forbidden behavior examples to evaluation domain (no sniffing, no type inference)
- [x] **ADR-020 (base 005):** Adapted test taxonomy for forecasting evaluation domain
- [x] **ADR-021 (base 006):** No domain adaptation needed (criteria are universal)
- [x] **ADR-001 (base 007):** Adapted silicon agent rules to views-evaluation tooling
- [x] **ADR-014 (base 009):** Adapted boundary examples to Adapter-Core and Config-Runtime boundaries
- [x] **ADR-023:** Created technical risk register ADR

---

## CICs

- [x] Replace placeholder active contracts list in `CICs/README.md` with project contracts
- [x] Create intent contracts for non-trivial classes:
  - [x] EvaluationFrame.md
  - [x] NativeEvaluator.md
  - [x] EvaluationReport.md
  - [x] PandasAdapter.md (removed in Phase 3)
  - [x] MetricCatalog.md

---

## Contributor Protocols

- [x] Review and adapt `contributor_protocols/silicon_based_agents.md` for project tooling
- [x] Review and adapt `contributor_protocols/carbon_based_agents.md` for project team
- [x] Adapt `contributor_protocols/hardened_protocol_template.md` for numerical computation domain

---

## Standards

- [x] Review `standards/logging_and_observability_standard.md` — adapted scope for Level 0 pure-math exception propagation
- [x] Review `standards/physical_architecture_standard.md` — includes critical bundling assessment

---

## Risk Register

- [x] Created `reports/technical_risk_register.md` seeded with 9 concerns from repo-assimilation
- [x] Created ADR-023 governing the risk register

---

## Final Verification

- [x] No files still have Status `--template--` (except ADR-022 which is intentionally deferred)
- [ ] No phantom references to non-existent files
- [ ] All cross-ADR references resolve correctly
- [ ] Run `validate_docs.sh` to check internal consistency
