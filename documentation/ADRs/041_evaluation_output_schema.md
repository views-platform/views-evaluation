# ADR-041: Evaluation Output Schema

**Status:** Accepted  
**Date:** 2025-06-16 · **Revised:** 2026-08-02  
**Deciders:** Xiaolong; revised by project maintainers  
**Consulted:** views-frames ADR-020 (MetricFrame contract home)  
**Informed:** All contributors, views-pipeline-core, views-reporting  

## Context

Standardized outputs are necessary for comparing ensemble models against constituent models and baselines.

The 2025 version of this ADR asserted that *"views-evaluation returns a structured dictionary, and views-pipeline-core handles the persistence to disk"*, and listed JSON and HTML report formats. **Both claims became false and stayed false for months.** `EvaluationReport.to_metric_frame()` and `MetricFrame.save()` shipped in PR #22, and `save()` writes three files to disk directly. No HTML has ever been produced by this library. The ADR was also left at `Proposed`, so the output surface consumers actually build against was governed by nothing. Corrected here; the detection gap is register **C-34**.

## Decision

`views-evaluation` produces **three output forms**, and it **does** write to disk for one of them.

### 1. In-memory result structure (the mathematical output)

`EvaluationReport` holds `{schema: {group_id: {metric: value}}}` for the three schemas of ADR-032. Exposed via:

- `to_dict()` — nested plain dict; the pandas-free, extra-free path.
- `get_schema_results(schema)` — mapped onto the typed metrics dataclasses.
- `to_dataframe(schema)` — pandas DataFrame; requires the optional `dataframe` extra.

`views-evaluation` does **not** serialise these to JSON or render HTML. That remains the orchestrator's job, and the original rationale still holds for them: formatting and run context belong where the run is orchestrated.

### 2. The evaluation-of-record (`MetricFrame`) — persisted by this library

`EvaluationReport.to_metric_frame()` emits a typed, provenance-stamped `MetricFrame`: a `float32` value column plus the six string axes `(eval_type, target, metric, group_id, partition, level)`, with one row per `(group, metric)` and a cross-group aggregate row keyed `group_id="mean"`.

**`MetricFrame.save(directory)` writes `values.npy`, `identifiers.npz` and `metadata.json`.** This library therefore *does* persist — narrowly, in its own format, for the audit record only. This does not reintroduce a circular dependency: the format is self-contained and `views-pipeline-core` chooses the location.

**The contract's home is views-frames ADR-020**, which owns the `FrameMetadata` provenance vocabulary and the `assert_frame_envelope` conformance checker this artifact satisfies. This ADR is the local record of what views-evaluation emits and does not restate that contract.

### 3. Failure output

There is none by design. A computation that cannot produce a result raises (**ADR-015**); a vacuous emit raises rather than persisting an empty record. There is no partial-output or error-report format.

## Consequences

### Positive
- The output surface consumers build against is now governed by an Accepted ADR rather than a Proposed one.
- The disk-writing boundary is stated explicitly instead of being denied.

### Negative
- Two output paths (`to_dict` for orchestrators, `MetricFrame` for the audit record) must be kept semantically consistent; nothing enforces that they agree.
- The authoritative contract lives in another repo, so a reader must follow a cross-repo link. Tracked as part of **C-24**.

## Rationale

Splitting "results the orchestrator formats" from "the record that must survive for audit" is why this library persists one and not the other. Report rendering has many valid presentations and belongs with the orchestrator; the evaluation-of-record has exactly one correct form and must be reproducible byte-for-byte, which argues for the producer owning its serialisation.

## References

- views-frames **ADR-020** — MetricFrame contract home, `FrameMetadata`, `assert_frame_envelope`
- **ADR-015** — why there is no partial or error output
- **ADR-032** — the three schemas whose results this carries
- `documentation/CICs/EvaluationReport.md`, `documentation/CICs/MetricFrame.md`
- Register **C-24** (contract drift), **C-26** (float32 / `schema_version`), **C-34** (the drift that produced this revision)
