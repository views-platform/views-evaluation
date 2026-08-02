# Class Intent Contract: EvaluationFrame

**Status:** Active  
**Owner:** Evaluation Core  
**Last reviewed:** 2026-08-02  
**Related ADRs:** ADR-010 (Ontology), ADR-011 (Topology), ADR-012 (Authority)

---

## 1. Purpose

The canonical, framework-agnostic internal representation of a forecasting evaluation task. It encapsulates synchronized NumPy arrays for observations, predictions, and identifiers.

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** perform data alignment or index intersection.
- This class does **not** handle I/O (loading from or saving to disk).
- This class does **not** depend on Pandas, Polars, or Xarray.
- This class does **not** calculate metrics (that is the role of MetricCalculators).

---

## 3. Responsibilities and Guarantees

- **Shape Integrity**: Guarantees that all internal arrays (`y_true`, `y_pred`, and all identifiers) have the same number of rows ($N$).
- **Sample Consistency**: Guarantees that `y_pred` is a dense 2D array of shape $(N, S)$, where $S \ge 1$.
- **Pure NumPy**: Guarantees that no Python objects (lists, dicts) are stored inside data cells.
- **State Immutability**: Provides methods to select subsets of data by creating *new* instances rather than mutating state.

---

## 4. Inputs and Assumptions

- **Pre-aligned Data**: Assumes that the adapter has already performed necessary joins and truth-duplication.
- **Homogeneous Types**: Assumes that all predictions in a single frame share the same task type and sample count.
- **Required Identifiers**: Expects at least `time`, `unit`, `origin`, and `step` identifiers to be present for regrouping.

---

## 5. Outputs and Side Effects

- **Group Indices**: `get_group_indices(key)` produces mappings of unique identifier values to integer row indices.
- **Sub-frames**: `select_indices(indices)` produces new `EvaluationFrame` instances for specific slices of data.
- **Properties**: `n_rows` (int, number of observations), `n_samples` (int, number of prediction columns), `is_sample` (bool, `True` when `n_samples > 1`).

---

## 6. Failure Modes and Loudness

- Raises `ValueError` if input arrays have mismatched lengths during initialization.
- Raises `ValueError` if required identifiers (`time`, `unit`, `origin`, `step`) are absent.
- Raises `ValueError` if any identifier array contains `NaN` or `None` (as per ADR-012).
- Raises `ValueError` if `y_true` or `y_pred` contain `NaN` or infinity (as per ADR-013).
- Raises `ValueError` if `y_true` or `y_pred` is object-dtype (Pure NumPy contract, ADR-011).
- Raises `ValueError` if **`y_true` is not 1-D** `(N,)`. Added 2026-08-02 (C-31) as the symmetric partner of the `y_pred` check below; an un-squeezed `(N, 1)` column previously constructed successfully, reported a plausible `n_rows`, and failed much later inside `_guard_shapes` as a *metric* error rather than an input-contract one.
- Raises `ValueError` if `y_pred` is not 2-D `(N, S)` (C-03).

Shape is validated **after** dtype, so a non-numeric array is reported as a dtype problem rather than a shape one.

**Loudness note:** this class is Level 0 and maintains **no logger** — exceptions propagate to the orchestrator (logging standard §5.1).

---

## 7. Boundaries and Interactions

- **Upstream**: Created by **Adapters**.
- **Downstream**: Consumed by **NativeEvaluator** and **MetricCalculators**.
- **Isolation**: Must not import anything outside of `numpy` and standard typing.

---

## 8. Step Semantics

The `step` identifier represents **positional lead time** (1-indexed), not an absolute calendar month. Step 1 is the first month of each forecast origin's prediction window, step 2 is the second, and so on. This is assigned positionally by adapters (e.g., views-pipeline-core's `EvaluationAdapter`) based on the order of unique time values within each origin sequence.

**Consequence:** Step 1 in origin A and step 1 in origin B typically refer to *different* calendar months. When `NativeEvaluator` groups data by step, it collects the "diagonals" of the parallelogram — all first-month-ahead predictions together, all second-month-ahead together, etc. This is the correct semantic for forecast-horizon evaluation.

Do not confuse `step` with the `time` identifier, which represents the absolute calendar month.

---

## 9. Examples of Correct Usage

```python
ef = EvaluationFrame(
    y_true=np.array([0, 1]),
    y_pred=np.array([[0.1, 0.2], [0.8, 0.9]]),  # 2 samples
    identifiers={
        'time':   np.array([100, 100]),  # calendar month id
        'unit':   np.array([1, 2]),      # spatial entity id
        'origin': np.array([0, 0]),      # 0-indexed sequence position
        'step':   np.array([1, 1]),      # 1-indexed lead time within sequence
    },
    metadata={'target': 'ged_sb_best'},
)
month_groups = ef.get_group_indices('time')
sub_ef = ef.select_indices(month_groups[100])
```

---

## 10. Examples of Incorrect Usage

- Constructing an `EvaluationFrame` directly with ragged sample arrays (varying S per row). External adapters should guard against this, but direct construction validates only ndim.
- Passing DataFrames or Series instead of NumPy arrays — the class has zero knowledge of Pandas.
- Omitting required identifier keys (e.g. passing only `time` and `unit` without `origin` and `step`).
- Storing derived or mutable state on an `EvaluationFrame` instance after construction.

---

## 11. Test Alignment

- **Green:** `tests/test_evaluation_frame.py::TestEvaluationFrameGreen` — construction, properties, grouping, selection.
- **Beige:** `tests/test_evaluation_frame.py::TestEvaluationFrameBeige` — single-row frames, large sample counts, multi-unit grouping.
- **Red:** `tests/test_evaluation_frame.py::TestEvaluationFrameRed` — shape mismatches, NaN/Inf/None in data and identifiers, missing keys.
- **Adversarial:** `tests/test_adversarial_inputs.py::TestAdversarialNativeInputs` — NaN/Inf boundary rejection.

---

## 12. Known Deviations

- ~~**Rectangular sample invariant not enforced**~~ — **resolved.** `_validate` now rejects any `y_pred` that is not 2-D, so a ragged array (which NumPy materialises as object-dtype) cannot be constructed: it is caught by either the object-dtype gate or the ndim gate. (Risk register C-03, closed 2026-03-31; deviation cleared here 2026-08-02.)
- **Integer identifier NaN not checked:** Validation checks float and object identifiers for NaN/None, but integer-typed identifiers are not checked (NumPy integers cannot represent NaN, so this is safe in practice but not explicitly documented).
- **No immutability enforcement:** The contract claims "State Immutability" via new-instance methods, but `y_true`, `y_pred`, and `identifiers` are publicly mutable attributes. Nothing prevents `ef.y_true[0] = 999` after construction.

---

## End of Contract

This document defines the **intended meaning** of `EvaluationFrame`.

Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
