# The Hardened Protocol: Contributor Governance for Numerical Evaluation

This document defines the mandatory engineering and mathematical standards for the `views-evaluation` repository. Adherence to this protocol is required for all contributions that affect metric computation, data transformation, or evaluation logic to guarantee scientific integrity and reproducibility.

---

## 1. Core Principles

### A. The Authority of Declarations (ADR-012)
**"Never infer; only trust declarations."**
All meaningful semantics (task types, prediction types, metric hyperparameters, step identifiers) must be explicitly declared in configuration or the `EvaluationFrame`.
- **Prohibited:** Type-sniffing from cell contents, step inference from row position without explicit assignment, scaling inference from target name prefixes.
- **Requirement:** If a parameter affects metric computation (e.g. twCRPS threshold, Coverage alpha), it must be a declared gene in the `MetricCatalog` genome and resolved via Chain of Responsibility.

### B. The Fail-Loud Mandate (ADR-013)
**"A crash is a successful defense of scientific integrity."**
Silent failures, implicit fallbacks, and "best-effort" corrections are forbidden.
- **Requirement:** Violations of data, configuration, or semantic invariants must raise explicit `ValueError` immediately.
- **Prohibited:** Using `np.nan_to_num`, silent clipping, "sensible defaults" for critical metric parameters, or downgrading errors to warnings.

### C. The Numerical Airlock (EvaluationFrame._validate)
All data entering the evaluation system must pass through the `EvaluationFrame` validation boundary.
- **Requirement:** Reject NaN and Inf values in observations and predictions at construction time.
- **Requirement:** Reject NaN/None in all identifier arrays at construction time.
- **Requirement:** Enforce shape consistency: `y_true` (N,), `y_pred` (N, S), all identifiers (N,).

### D. The Metric Genome Contract (ADR-042)
**"No silent defaults."**
Every metric hyperparameter must be declared in the `MetricSpec.genome` tuple and resolved explicitly.
- **Requirement:** New metrics must declare all required hyperparameters in their genome.
- **Requirement:** Metric functions must use keyword-only arguments without defaults for genome parameters.
- **Prohibited:** Hardcoding default values in metric function signatures.

---

## 2. Contributor Requirements

### Adding a New Metric
1. **Implement the function** in `native_metric_calculators.py` with keyword-only args for genome parameters.
2. **Register in catalog:** Add a `MetricSpec` to `METRIC_CATALOG` in `metric_catalog.py`.
3. **Declare membership:** Add the metric name to the appropriate set in `METRIC_MEMBERSHIP`.
4. **Add to profile:** Add genome parameter values to `BASE_PROFILE` (and other relevant profiles).
5. **Add dataclass field:** Add the metric as `Optional[float] = None` to the appropriate typed metrics dataclass in `metrics.py`.
6. **Write tests:** Include at minimum one golden-value test and one red-team test.

### Modifying an Existing Metric
1. **Update the CIC** if the change affects behavior described in the intent contract.
2. **Verify parity** by running the full Green/Beige/Red test suite.
3. **Update golden-value tests** if numerical output changes.

---

## 3. Mandatory Testing Taxonomy (ADR-020)

Every Pull Request affecting metric computation must include tests covering:

### Green Team (Stability & Correctness)
- **Goal:** Ensure the metric produces correct values for known inputs.
- **Examples:** Golden-value tests against analytical solutions, CRPS parity with `properscoring`, bit-identical results across schemas.

### Beige Team (Configuration & Human Error)
- **Goal:** Catch failures caused by common configuration mistakes or missing parameters.
- **Examples:** Missing genome parameters in profile, requesting unimplemented metrics, mismatched task/pred_type combinations.

### Red Team (Adversarial)
- **Goal:** Expose failure modes by deliberately trying to make the system produce wrong results silently.
- **Examples:** NaN injection in predictions, Inf in observations, ragged sample arrays, zero-variance inputs.

---

## 4. Operational Invariants

- **Shape Guard Defense-in-Depth:** All metric functions call `_guard_shapes()` even though `EvaluationFrame._validate()` has already checked. This is deliberate double-checking, not redundancy to remove.
- **Profile Consistency:** All profiles must provide values for all metrics with non-empty genomes that may be requested in evaluations using that profile.
- **Schema Reproducibility:** Month-wise, time-series-wise, and step-wise schemas must produce identical results regardless of the order of input rows (grouping is by identifier value, not position).

---

**"In this repository, we value explicit correctness over convenient execution."**
