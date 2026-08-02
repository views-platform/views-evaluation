# Class Intent Contract: MetricFrame

**Status:** Active  
**Owner:** Evaluation Core  
**Last reviewed:** 2026-08-02  
**Related ADRs:** views-frames ADR-020 (contract home), ADR-041 (Output Schema), ADR-013 (Observability), ADR-015 (Degenerate/Empty Results), ADR-011 (Topology)  

---

## 1. Purpose

The **evaluation-of-record**: a typed, transportable, provenance-stamped container of evaluation metric values, and the artifact `views-reporting` and `views-pipeline-core` consume instead of scraping WandB.

It is a string-keyed value object — **not** a spatiotemporal `(time, unit)` frame — keyed by the axes `(eval_type, target, metric, group_id, partition, level)`. It exposes the shared views-frames "frame envelope" surface (`values` float32 with an explicit trailing axis, `n_rows`, and a `save`/`load` round-trip) so a consumer can validate it with `views_frames.conformance.assert_frame_envelope` rather than re-asserting a drifting local copy.

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** compute metrics — it carries values already computed by `NativeEvaluator`.
- This class does **not** decide *what* to emit; `EvaluationReport.to_metric_frame()` owns flattening and aggregation.
- This class does **not** render reports, format for humans, or produce JSON/HTML summaries (ADR-041).
- This class does **not** know about DataFrames.
- This class does **not** validate that its contents are *meaningful* — only that they are structurally well-formed. A frame of all-NaN values is valid here; whether that should ever have been emitted is the emit site's concern.

---

## 3. Responsibilities and Guarantees

- **Envelope Conformance:** Guarantees `values` is a `float32` array of shape `(N, 1)` — an explicit trailing axis, never a bare 1-D array — and that every axis in `AXES` is present as a 1-D array of length `N`. This is what makes `assert_frame_envelope` pass.
- **Complete Key Space:** Guarantees all six axes are present. `partition` and `level` may be empty strings (reporting does not key on them), but the columns exist so the key space is complete.
- **Provenance Split (views-frames ADR-020, register C-47):** Guarantees generic identity (`model`, `run_id`, `data_version`, `run_type`, `timestamp`, `seed`) lives in the reused `views_frames.FrameMetadata`, while evaluation-specific identity (`scoring_code_version`, `evaluation_timestamp`) stays in `MetricFrameMetadata` and **never leaks into the generic header**.
- **Round-Trip Fidelity:** Guarantees `load(save(x))` reconstructs an equivalent frame, including numpy-typed metadata scalars.
- **Fail-Loud Construction:** Guarantees structural violations raise at construction, never later (ADR-013).
- **Level-1 Observability:** Guarantees every structural failure is logged at `ERROR` **before** raising. Unlike Level-0 components, this class maintains its own logger — it persists an audit record and there is no orchestrator between it and the filesystem (logging standard §5.1).

---

## 4. Inputs and Assumptions

- **`values`**: `np.ndarray`, dtype `float32`, shape `(N, 1)`. `NaN` **is permitted** — see §6.
- **`identifiers`**: dict containing exactly the keys in `AXES`, each a 1-D length-`N` array of strings.
- **`metadata`**: optional `MetricFrameMetadata`; defaults to an empty one.
- **Assumes** the caller has already decided what belongs in the frame. Construction is a structural gate, not a semantic one.
- **Requires** the optional `views-frames` dependency (`pip install views-evaluation[frames]`). The module is import-gated in `views_evaluation/__init__.py` on `find_spec`, so the core API stays importable without it (ADR-011 minimal core).

---

## 5. Outputs and Side Effects

- **`n_rows`** (property): row count.
- **`save(directory)`**: **writes three files to disk** — `values.npy`, `identifiers.npz`, `metadata.json`. Creates the directory if absent. This is the one place `views-evaluation` performs I/O (ADR-041 §2).
- **`load(directory)`** (classmethod): reconstructs a frame written by `save`. Loads with `allow_pickle=False`.
- **`MetricFrameMetadata.to_dict()` / `.from_dict()`**: flatten and reconstruct provenance, routing generic keys to `FrameMetadata`.

---

## 6. Failure Modes and Loudness

All raise `ValueError` at construction, each logged at `ERROR` first:

- `values` is not a numpy array.
- `values` is not `float32`.
- `values` is not 2-D.
- Any axis in `AXES` is missing from `identifiers`.
- Any identifier is not 1-D.
- Any identifier length does not match `values.shape[0]`.

`_json_default` raises `TypeError` (logged) for a metadata value it cannot serialise.

### NaN is permitted — deliberately

`values` may contain `NaN`, meaning *"this metric was not calculated for this group"*. This is **not** a fail-loud violation:

- `load()` must faithfully reconstruct whatever `save()` wrote, including historical frames.
- The `group_id="mean"` aggregate row carries `nanmean` semantics by design.
- Accepted sentinels upstream (`MCR`'s `inf`/`nan`, `Pearson`'s `nan`) legitimately reach here.

ADR-015 **ruling 3** rules on this explicitly: the guard against a *vacuous* record belongs at the emit site (`to_metric_frame` raises when no schema produced any value), not in the container, because the emit site has the context to say *why* it was empty. Tightening the container would break the round-trip and change a published cross-repo envelope.

### Zero rows are permitted

A zero-row frame is constructible and loadable, for the same round-trip reason. `to_metric_frame()` will not *emit* one.

---

## 7. Boundaries and Interactions

- **Upstream:** produced solely by `EvaluationReport.to_metric_frame()`.
- **Downstream:** consumed by `views-reporting` (renders the `group_id="mean"` rows) and `views-pipeline-core`.
- **Substrate:** reuses `views_frames.FrameMetadata` and is validated by `views_frames.conformance.assert_frame_envelope`.
- **Level:** this is a **Level 1** component. It may perform I/O; Level 0 may not.
- **Isolation:** must not import pandas, and must not reach back into `NativeEvaluator` or the metric kernels.

---

## 8. The Cross-Repo Contract

The following are **load-bearing for other repositories**. Changing any of them is a breaking change requiring coordination with views-reporting and views-pipeline-core (register **C-24**, governed by ADR-022):

| Element | Value |
|---|---|
| Axes | `(eval_type, target, metric, group_id, partition, level)` |
| Aggregate row key | `MEAN_GROUP_ID` = `"mean"` |
| `eval_type` vocabulary | `month-wise`, `time-series-wise`, `step-wise` (via `SCHEMA_TO_EVAL_TYPE`) |
| Metric tokens | must remain valid names in `METRIC_MEMBERSHIP` |
| Serialization | `values.npy` + `identifiers.npz` + `metadata.json` |
| `values` dtype | `float32` |

Only the **metric-token** half is currently CI-guarded (`tests/test_metric_frame.py::TestCanonicalTokenDriftGuard`). The axes, vocabulary, `MEAN_GROUP_ID` and serialization format are **unguarded** — that gap is register **C-24**.

---

## 9. Examples of Correct Usage

```python
report = evaluator.evaluate(ef)
mf = report.to_metric_frame(
    model_id="purple_alien", run_id="abc123", data_version="v7",
    partition="calibration", level="pgm",
)
assert_frame_envelope(mf)          # views-frames conformance
mf.save("/path/to/run/evaluation")
later = MetricFrame.load("/path/to/run/evaluation")
```

---

## 10. Examples of Incorrect Usage

- Constructing a `MetricFrame` by hand instead of via `to_metric_frame()` — the emit path owns aggregation and provenance.
- Passing `float64` values, or a 1-D `values` array — both raise; the trailing axis is part of the envelope.
- Putting `scoring_code_version` or `evaluation_timestamp` into `FrameMetadata` — they belong in `MetricFrameMetadata` (C-47).
- Inferring `model_id`/`run_id`/`data_version` inside this library. All identity is **injected** by the caller; `run_id` may legitimately be `None` at emit time, when the WandB run does not yet exist.
- Aggregating `values` across groups with `mean` rather than `nanmean` — permitted `NaN`s will poison the result.
- Treating `schema_version` as enforced on `load()`. It is written but **not** checked (register **C-26**).

---

## 11. Test Alignment

- **Green:** `tests/test_metric_frame.py::TestToMetricFrameGreen` — emit for all four `(task, pred_type)` cells, envelope conformance, save/load round-trip, vocabulary mapping, provenance split, mean aggregate rows.
- **Beige:** `TestToMetricFrameBeige` — empty individual schema, NaN values, `run_id=None`, default `scoring_code_version`, zero-row container conformance.
- **Red:** `TestMetricFrameConstructionRed` — non-float32, wrong ndim, missing axis, length mismatch, non-array, 2-D identifier.
- **Red (loudness):** `TestMetricFrameLogAndRaiseRed` — log-and-raise pairing, identical log/exception message, and the Level-0 exemption asserted in the opposite direction.
- **Red (emit):** `TestVacuousEmitRed` — the ADR-015 ruling-6 guard.
- **Drift guard:** `TestCanonicalTokenDriftGuard` — metric tokens against views-reporting's canonical set.

---

## 12. Known Deviations

- **`schema_version` is not enforced on `load()`** — an old frame under a changed format is not rejected loudly. Register **C-26**; the cross-repo wire-contract half of views-frames C-46 is open and is this repo's responsibility.
- **float64 → float32 cast at emit.** Metric values are computed in float64 and narrowed for the envelope, so the persisted record is less precise than the computed metric. Register **C-26**.
- **`scoring_code_version` identifies the code that ran, when it can.** It is the installed distribution's version, plus `+g<sha>` when the package is running from a git worktree (e.g. `1.0.0+g5469690`). A wheel install has no worktree and stamps a bare version — a property of the install, not a failure. The SHA exists because `importlib.metadata` reports the *installed distribution*, not the executing code: under an editable install those drift as soon as the source moves ahead of the last `pip install`, and a bare version cannot distinguish the two. Register **C-25**.
- **Most of the cross-repo contract in §8 is unguarded by CI.** Register **C-24**.

---

## End of Contract

This document defines the **intended meaning** of `MetricFrame`.

Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract — and, because §8 is consumed by other repositories, must be coordinated with them.
