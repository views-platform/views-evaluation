# Physical Architecture Standard

**Status:** Active  
**Governing ADRs:** ADR-010 (Ontology), ADR-011 (Topology)  

---

## 1. The 1-Class-1-File Standard

**Every non-trivial class must live in its own file named after the class in `snake_case`.**

- **Correct:** `EvaluationFrame` lives in `evaluation_frame.py`.
- **Correct:** `NativeEvaluator` lives in `native_evaluator.py`.
- **Exception:** Trivial data containers directly related to a class may coexist in the same file.

---

## 2. Directory Ontology

Files must be located in directories that match their functional category:

```
views_evaluation/
├── evaluation/          # Core evaluation logic (Level 0)
│   ├── evaluation_frame.py
│   ├── native_evaluator.py
│   ├── metric_catalog.py
│   ├── native_metric_calculators.py
│   ├── evaluation_report.py
│   ├── metrics.py
│   └── config_schema.py
├── adapters/            # Reserved for future framework bridges
│   └── __init__.py
└── profiles/            # Named evaluation profiles
    ├── base.py
    └── hydranet_ucdp.py
```

---

## 3. Current State Assessment — Bundling

### Compliant

| File | Contents | Verdict |
|------|----------|---------|
| `evaluation_frame.py` | `EvaluationFrame` (1 class) | Compliant |
| `native_evaluator.py` | `NativeEvaluator` (1 class) | Compliant |
| `metric_catalog.py` | `MetricSpec` + `METRIC_CATALOG` + `METRIC_MEMBERSHIP` + `resolve_metric_params` | Cohesive module — spec, registries, and resolver form a single concept |
| `evaluation_report.py` | `EvaluationReport` (1 class) | Compliant |
| `config_schema.py` | `EvaluationConfig` (1 TypedDict) | Compliant |

### Defensible Exception

| File | Contents | Verdict |
|------|----------|---------|
| `metrics.py` | 5 dataclasses: `BaseEvaluationMetrics` + 4 typed 2x2 containers | Defensible — trivial data containers sharing a base class. Splitting into 5 files would create fragmentation without improving discoverability. |

### Identified Challenge

| File | Contents | Concern |
|------|----------|---------|
| `native_metric_calculators.py` | 437 lines: `_guard_shapes` (shared guard), 15+ implemented metric functions spanning 4 categories, 4 placeholder stubs, 4 legacy dispatch dicts, 1 legacy alias | **Bundling challenge** |

**Analysis of `native_metric_calculators.py`:**

This file bundles heterogeneous concerns:

1. **Shared utility** (`_guard_shapes`) — used by all metrics, should arguably be its own module or remain as a private helper.

2. **Four metric families:**
   - Regression point: MSE, MSLE, RMSLE, EMD, Pearson, MTD, y_hat_bar, MCR
   - Regression sample: CRPS, twCRPS, MIS, QIS, QS_sample, Coverage, Ignorance
   - Classification point: AP, Brier_point, QS_point
   - Classification sample: Brier_sample

3. **Placeholder stubs** for unimplemented metrics (SD, pEMDiv, Variogram, Jeffreys).

4. **Legacy dispatch dicts removed in Phase 3.** `METRIC_MEMBERSHIP` is now the single source of truth.

**Why this is a challenge:**
- The file mixes 4 distinct metric families. Adding a new regression-sample metric requires editing a 437-line file that also contains classification metrics.
- The legacy dispatch dicts at the bottom duplicate the `METRIC_MEMBERSHIP` registry (risk C-01).
- The file has the highest line count of any source module and the most heterogeneous responsibility set.

**Why splitting is not straightforward:**
- All metric functions share `_guard_shapes`. Splitting would either duplicate it or create a shared utility module.
- The `MetricCatalog` imports all 22 functions from this single module. Splitting would require updating the catalog's import block.
- Functions are stateless and flat — they are not classes, so the 1-class-1-file rule does not directly apply.

**Recommendation (for future consideration):**
If and when the file exceeds ~600 lines or the metric count exceeds ~30, consider splitting into:
```
evaluation/
├── metric_calculators/
│   ├── __init__.py          (re-exports all functions)
│   ├── _guard.py            (_guard_shapes)
│   ├── regression_point.py
│   ├── regression_sample.py
│   ├── classification.py
│   └── placeholders.py
```

This is a **future evolution path**, not a current mandate. The current bundling is tolerable but approaches the threshold where it creates friction.

---

## 4. Import Conventions

- **Explicit imports:** Avoid `from module import *`.
- **Circular dependency guard:** Follow ADR-011 layering. Level 0 modules must not import from Level 1 or Level 2.
- **Lazy imports for Pandas:** Pandas is imported inside methods (e.g. `to_dataframe()`) rather than at module level in Level 0/1 code.

---

## 5. Enforcement

Compliance with this standard is assessed during:
- Code review
- Repo-assimilation audits
- Tech debt cleanup cycles

PRs that introduce new multi-class files or significantly expand existing bundled files should document the justification.

---

**"The structure of the files is as rigorous as the logic of the code."**
