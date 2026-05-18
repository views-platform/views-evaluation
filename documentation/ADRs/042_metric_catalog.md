# ADR-042: Metric Catalog and Named Evaluation Profiles

**Status:** Accepted  
**Date:** 2026-03-11  
**Deciders:** Project maintainers  
**Consulted:** Claude (silicon-based agent)  
**Informed:** All contributors  

## Context

views-evaluation's metric functions accept hyperparameters (e.g. `threshold` for twCRPS,
`alpha` for Coverage) but these were hardcoded as function-signature defaults. The
NativeEvaluator called metrics with zero hyperparameters, meaning every metric silently
used its compile-time default. This is the exact implicit-configuration anti-pattern that
views-r2darts2's ReproducibilityGate was built to prevent.

Additionally, metric hyperparameters should vary across target+unit-of-analysis combinations.
For example, twCRPS threshold for a PRIO-GRID model predicting fatalities differs from a
country-month model. Without a shared configuration mechanism, each model developer
independently picks values, leading to silent inconsistency across models.

## Decision

### Overview

Introduce a **Metric Catalog** (genome registry) and **Named Evaluation Profiles** following
the views-r2darts2 catalog pattern. The catalog declares what hyperparameters each metric
requires (its "genome") but provides NO default values. Values are supplied by named profiles
and/or per-model overrides, resolved via Chain of Responsibility.

### Components

1. **MetricSpec** (frozen dataclass): declares `function`, `genome` (tuple of required param
   names), and `implemented` flag. No defaults field.

2. **METRIC_CATALOG**: maps metric names to MetricSpec instances.

3. **METRIC_MEMBERSHIP**: maps (task, pred_type) tuples to sets of valid metric names.

4. **resolve_metric_params()**: Chain of Responsibility resolver.
   Resolution order: model overrides → profile → fail loud.

5. **Named Profiles** in `views_evaluation/profiles/`:
   - `base.py`: system-wide standard evaluation protocol
   - Researchers add target+unit profiles (e.g. `sb_best_pgm.py`)

6. **No defaults in function signatures**: all metric functions with genome params use
   keyword-only arguments without defaults.

### Resolution Chain

```
model config overrides → named evaluation profile → ValueError (fail loud)
```

A model's config specifies:
- `evaluation_profile`: name of the profile (e.g. "base", "sb_best_pgm")
- `metric_hyperparameters`: optional per-metric overrides

## Consequences

**Positive Effects:**
- Every hyperparameter is explicit and auditable — no silent defaults
- Cross-model consistency via shared profiles
- Domain-appropriate variation via target+unit profiles
- Follows the same pattern as views-r2darts2 (ModelCatalog, LossCatalog, OptimizerCatalog)
- Clean Architecture: views-evaluation (core) declares the contract, consumers fulfill it

**Negative Effects:**
- Breaking change for direct callers of metric functions (must now pass params explicitly)
- Slight increase in config complexity (evaluation_profile key)

## Rationale

### Clean Architecture (Uncle Bob)
The catalog belongs in views-evaluation (core). It declares requirements. Values are policy;
policy belongs at the boundary. The Dependency Rule is preserved: views-evaluation defines
the contract, views-models fulfills it.

### Design Patterns (Gang of Four)
- **Strategy Pattern**: each metric function is a strategy. The catalog formalizes the registry.
- **Chain of Responsibility**: param resolution walks model overrides → profile → fail.
  This matches the domain — target+unit groupings are real, meaningful clusters.
- The catalog is a flat registry, not a class hierarchy. Metrics are stateless functions.

### Why not defaults in code?
Hardcoded defaults lead to silent failures. If 24 models all use Coverage with alpha=0.1
because nobody bothered to configure it, and the research team later decides alpha=0.05 is
more appropriate, the change requires updating 24 model configs. Profiles solve this: change
the profile once, all models using it get the update.

## Additional Notes

- Legacy dispatch dicts (REGRESSION_POINT_NATIVE, etc.) were removed in Phase 3.
  METRIC_MEMBERSHIP is the single source of truth for (task, pred_type) → metric mapping.
- The base profile ships with views-evaluation and provides values that match the previous
  function-signature defaults, ensuring zero behavioral change for existing integrations.
- Profile values for twCRPS threshold and QIS quantile levels are subject to alignment
  decisions with views-metric-lab.

## Feedback and Suggestions

Open questions for the research team:
- twCRPS default threshold: 0.0 (current) or 0.1 (views-metric-lab)?
- Should additional target+unit profiles be defined upfront or added on demand?
