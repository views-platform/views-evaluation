# ADR-022: Rules for Evolution and Stability

**Status:** Accepted  
**Date:** 2026-02-25 (deferred) · **Activated:** 2026-08-02  
**Deciders:** Project maintainers  
**Consulted:** 2026-04-03 incident investigation; review-rr strategic (2026-08-02)  
**Informed:** All contributors, downstream consumers (views-pipeline-core, views-reporting)  

---

## Context

The preceding ADRs establish the ontology, topology, and semantic authority at a point in time. What they do not yet define is how the system is allowed to change over time (versioning, breaking changes, backward compatibility).

### Why this ADR was activated (2026-08-02)

This ADR was deliberately deferred in 2026-02 and listed its own reconsideration triggers. **The second trigger fired.**

On **2026-04-03**, `EvaluationManager` and `PandasAdapter` were deleted in a single PR (#16) with no deprecation warning in any prior release. **All six downstream integration tests in views-pipeline-core crashed with `ModuleNotFoundError`.** The Phase 4 bridge wrapper that would have softened the removal was deleted in the same commit that merged the purge. Tracked as risk register **C-13** (Tier 2).

Two further facts made deferral untenable:

1. **The breakage reached users.** Published `v0.4.0` (2026-05-18) ships without `EvaluationManager`, `PandasAdapter`, and the legacy-config-key shim — while its README still tells users legacy keys work and emit a `DeprecationWarning`. Users are currently reading a promise the package does not keep.
2. **ADR-015 adds more breaking changes.** Rulings 2, 4, 5 and 7 turn previously-silent successes into raises. Shipping those without a policy would repeat 2026-04-03 deliberately.

C-13's register entry states the reason a policy, not a register entry, is the fix: *as a register entry it has no enforcement point and will re-fire on every release.*

## Decision

The deferral is **withdrawn**. The following rules govern evolution of this repository.

### 1. What constitutes public API

The public API is **every symbol exported in `views_evaluation/__init__.py:__all__`**, plus the documented behaviour of those symbols: constructor signatures, method signatures, return types, raised exception types, and the config keys declared in `EvaluationConfig`.

The `MetricFrame` on-disk format and axis vocabulary are additionally a **cross-repo contract** and are treated as public API regardless of `__all__`.

Anything else — module paths, private helpers, internal dict shapes — is not public and may change without notice.

### 2. Deprecation contract

Before an `__all__`-exported symbol is **removed**:

1. A `DeprecationWarning` must ship in a **published release**, naming the replacement.
2. **At least one full release cycle** must pass between the deprecating release and the removing release.
3. The deprecation must be recorded in the release notes of both releases.

The same applies to removing a supported config key, narrowing an accepted input, or changing a raised exception type.

### 3. Behaviour changes that make previously-accepted input fail

ADR-015 introduces raises where the library previously returned a value. These are **not** symbol removals, so rule 2's warning-first mechanism does not apply — there is no symbol to warn on, and emitting a warning instead of raising is forbidden by ADR-013 and ADR-015.

Such changes are governed instead by:

1. They must be **announced in release notes** as breaking, with the specific inputs that now fail.
2. Downstream consumers known to be affected must be **notified before** the release is cut, not after.
3. The release notes must state the migration: what the caller changes to keep working.

### 4. `DeprecationWarning` versus fail-loud — the boundary

These two rules can appear to conflict. They do not, and the distinction is this:

| Situation | Correct response | Governed by |
|---|---|---|
| A **structural failure** — invalid config, undefined computation, broken invariant | **Raise.** Never warn | ADR-013, ADR-015 |
| A **still-supported but scheduled-for-removal** API that works correctly today | **`DeprecationWarning`**, then remove a cycle later | This ADR, rule 2 |

A `DeprecationWarning` announces a *future* removal of something that currently works. It is never a substitute for raising on something that is already wrong. Where a deprecated path would also produce an incorrect result, ADR-013 wins: raise.

### 5. Versioning

The project follows Semantic Versioning, with the standard `0.x` caveat made explicit:

- **While `0.x`:** the public API carries no stability guarantee. Breaking changes may ship in a MINOR bump — but rules 2 and 3 still apply in full. Deferred stability is not deferred *communication*.
- **From `1.0.0`:** breaking changes require a MAJOR bump.
- A release that removes a symbol or breaks behaviour **must** bump at least MINOR; it may never ship as PATCH.

`1.0.0` should be cut when downstream consumers require a stability guarantee — not before.

### 6. Retroactive position on the 0.4.0 removals

The `EvaluationManager` / `PandasAdapter` / legacy-key removals shipped in 0.4.0 without notice. That cannot be undone, and the shim will **not** be restored: it has been absent from the released package since 2026-05-18, and after ADR-015 ruling 4 a legacy key fails loudly rather than silently, which is the better outcome.

What is owed instead:

- The README migration notice must be corrected to describe what the package actually does. *(This is the fix for C-29.)*
- The 0.5.0 release notes must retrospectively record the 0.4.0 removals as breaking.

### 7. Enforcement

This policy is checked at two points:

- **Release checklist** (below) — worked through before any `poetry publish`, and the
  worked-through answers recorded in `CHANGELOG.md`, which is this repository's
  release-notes artifact for every "release notes" obligation named above. It was created
  for 0.5.0; nothing yet *requires* the next release to repeat the exercise, which is
  registered as **C-36**.
- **CI** — `tests/test_documentation_contracts.py` asserts that documentation makes no support claim the code does not honour, which is the specific failure that produced C-29.

#### Release checklist

- [ ] Does this release do any of the things rule 2 governs — remove an `__all__` symbol, remove a supported config key, narrow an accepted input, or change a raised exception type? If so, did a `DeprecationWarning` ship at least one release ago? *(This item asked only about symbols and config keys until 2026-08-02, under-covering the rule it enforces: 0.5.0 both narrowed an accepted input and changed an exception type, and a "no" was defensible against the old wording.)*
- [ ] Does this release make previously-accepted input fail? If so, are the release notes explicit, and have known consumers been notified?
- [ ] Does the version bump match the change class (rule 5)?
- [ ] Do the release notes list every breaking change with its migration?
- [ ] Does `MetricFrame`'s format or axis vocabulary change? If so, has it been agreed with views-reporting and views-pipeline-core?

## Consequences

### Positive
- Downstream consumers get advance signal instead of `ModuleNotFoundError` in CI.
- The policy has a named enforcement point, so it stops depending on whoever happens to remember.
- The fail-loud / deprecate boundary is written down, so ADR-013 and this ADR cannot be read as contradicting.
- `0.x` flexibility is preserved without using it as an excuse for silent breakage.

### Negative
- A removal now takes at least two release cycles, which slows cleanup.
- Requires release-notes discipline that this repository has not previously practised.
- Rule 3 depends on knowing who the consumers are; that knowledge is currently informal.

## References

- ADR-013 (Observability and Explicit Failure) — why warnings never replace raising
- ADR-015 (Degenerate and Empty Results) — the breaking changes rule 3 governs
- `reports/technical_risk_register.md` — C-13 (this policy's origin), C-29 (the documentation half)
- 2026-04-03 incident investigation — 6/6 downstream integration tests crashed
- `views_evaluation/__init__.py`, `__all__` — the public surface this policy governs
