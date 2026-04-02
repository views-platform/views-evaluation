# Architectural Decision Records (ADRs)

## Overview
This folder houses the Architectural Decision Records (ADRs) for the Views Evaluation repository. ADRs capture the "why" behind our system design, ensuring that architectural intent is preserved as the system evolves.

## The Foundation Suite
We follow a hierarchical numbering scheme to organize decisions from the most foundational to the most concrete.

### 00x: Foundational Principles
*Process and philosophy governing how we build and interact.*
- **000**: [Use of ADRs](000_use_of_adrs.md)
- **001**: [Silicon-Based Agent Protocol](001_silicon_based_agent_protocol.md)

### 01x: Architectural Invariants
*The "Rules of the House" that ensure system integrity.*
- **010**: [Evaluation Ontology](010_ontology_of_evaluation.md)
- **011**: [Topology and Dependency Rules](011_topology_and_dependency_rules.md)
- **012**: [Authority over Inference](012_authority_over_inference.md)
- **013**: [Observability and Explicit Failure](013_observability_and_explicit_failure.md)
- **014**: [Boundary Contracts and Validation](014_boundary_contracts_and_validation.md)

### 02x: Engineering Discipline & Quality
*Standards for implementation and verification.*
- **020**: [Multi-Perspective Testing](020_multi_perspective_testing.md)
- **021**: [Intent Contracts for Classes](021_intent_contracts_for_classes.md)
- **022**: [Evolution and Stability](022_evolution_and_stability.md)
- **023**: [Technical Risk Register](023_technical_risk_register.md)

### 03x: Domain Strategy & Methodology
*The mathematical and strategic core of conflict evaluation.*
- **030**: [Evaluation Strategy](030_evaluation_strategy.md)
- **031**: [Evaluation Metrics](031_evaluation_metrics.md)
- **032**: [Metric Calculation Schemas](032_metric_calculation_schemas.md)

### 04x: Data & Integration Contracts
*Schemas for I/O and external systems.*
- **040**: [Evaluation Input Schema](040_evaluation_input_schema.md)
- **041**: [Evaluation Output Schema](041_evaluation_output_schema.md)

---

## Governance Structure

- **Ontology (010)** defines what exists.
- **Topology (011)** defines structural direction.
- **Authority (012)** defines who owns meaning.
- **Observability (013)** enforces failure semantics.
- **Boundary Contracts (014)** define interaction rules.
- **Testing (020)** verifies system integrity.
- **Intent Contracts (021)** bind class-level behavior.
- **Evolution (022)** (deferred) — rules for stability.
- **Risk Register (023)** tracks structural concerns.
- **Silicon Agent Protocol (001)** constrains automated modification.

Together with domain ADRs (030–042), these define the invariant layer of the system.

---

## Contributing
To add a new ADR:
1. Identify the appropriate group for the decision.
2. Use the [ADR Template](adr_template.md).
3. Ensure the ADR follows the "Fail-Loud" principle (ADR-013).
