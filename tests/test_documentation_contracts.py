"""
Documentation-contract tests: assert that the docs describe the code as it is.

Why this file exists
--------------------
The original ``tests/test_documentation_contracts.py`` was deleted in the Phase 3
purge (commit ``fe6dd4c``, 2026-04-01) along with nine other test files. The others
were ``PHASE-3-DELETE`` tests for code that genuinely no longer existed (tracked and
closed as register C-06). This one had **no replacement**, and the consequences were
measurable:

  * The README promised legacy config keys still worked and emitted a
    ``DeprecationWarning``. No such code existed — and that claim **shipped in
    published 0.4.0** (register C-29).
  * The README listed ``evaluation_manager.py``, deleted four months earlier.
  * ``CICs/NativeEvaluator.md`` §3 said ``legacy_compatibility`` defaulted to ``True``
    while §11 and the code said ``False``.
  * ``CICs/NativeEvaluator.md`` §4 omitted the required ``unit`` identifier.

Every check below is narrow and falsifiable, and each is anchored to one of those
real regressions. This file must not become a prose or style checker — the original
was deleted partly because it had drifted into unfalsifiable territory, and a test
nobody trusts gets deleted again.

Per ADR-020 the checks are Red: they assert that a specific documented claim matches
a specific code fact, and fail loudly when it stops doing so.
"""
import inspect
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
DOCS = REPO_ROOT / "documentation"
CICS = DOCS / "CICs"

# Symbols removed in the Phase 3 purge (commit fe6dd4c, 2026-04-01). One structure so
# that retiring a further symbol means editing one place, not two: the prose check below
# matches file PATHS (a project-structure tree citing a module that is gone) while the
# docstring check matches CLASS NAMES (source claiming a deleted class still does work).
_DELETED_SYMBOLS = (
    # (class name, module paths that would appear in a file listing)
    ("EvaluationManager", ("evaluation_manager.py",)),
    ("PandasAdapter", ("adapters/pandas.py", "adapters.pandas")),
)
_DELETED_MODULE_PATHS = [p for _, paths in _DELETED_SYMBOLS for p in paths]
_DELETED_CLASS_NAMES = [name for name, _ in _DELETED_SYMBOLS]

# Files where a historical mention is legitimate: change logs, decision records and
# the risk register all necessarily discuss things that no longer exist.
_HISTORICAL_ALLOWLIST = {
    "documentation/ADRs",             # ADRs record why things were removed
    "documentation/CICs/EvaluationReport.md",   # Evolution Notes
    "documentation/CICs/NativeEvaluator.md",    # Evolution Notes
    "documentation/contributor_protocols",
    "reports",                        # register + post-mortems are history by nature
}

# Prose may legitimately name a removed symbol or a superseded metric when it is
# *explaining* the removal or rename. These markers signal that framing, matched
# case-insensitively against the whole block (see `_blocks`), not a single line.
#
# ⚠ This couples the checks to prose wording. `config_schema.py`'s note passes the
# deleted-class check only because it opens "Until 0.4.0 this docstring named ...".
# If you reword such a note, keep one of these markers somewhere in the same paragraph —
# or add the new phrasing here — otherwise the check fails for a purely cosmetic edit.
_HISTORICAL_MARKERS = (
    "until 0.", "prior to 0.", "before 0.",
    "was removed", "were removed", "was deleted", "were deleted",
    "removed in phase 3", "deleted in phase 3",
    "replaced by", "were replaced", "renamed", "breaking rename",
    "intentionally omitted", "pre-pr-", "legacy name", "no longer",
)


def _is_historical(text: str) -> bool:
    """True if the text frames its subject as history rather than as current fact."""
    lowered = text.lower()
    return any(marker in lowered for marker in _HISTORICAL_MARKERS)


def _blocks(text: str):
    """Yield (start_lineno, block_text) for each blank-line-separated block.

    Checks operate on blocks, not lines, because prose wraps: in
    `physical_architecture_standard.md` the phrase "the pre-PR-#18 names" sits on one
    line and the names it is explaining wrap onto the next. Line-based matching split
    the framing from its subject and produced a false positive.
    """
    lines = text.splitlines()
    start, buf = 1, []
    for i, line in enumerate(lines, 1):
        if line.strip():
            if not buf:
                start = i
            buf.append(line)
        elif buf:
            yield start, "\n".join(buf)
            buf = []
    if buf:
        yield start, "\n".join(buf)


def _doc_files():
    """All documentation prose, excluding files whose job is to record history.

    `CHANGELOG.md` is included deliberately. It is the release-notes artifact ADR-022
    requires, which makes it the highest-traffic place in the repo for behavioural
    promises to users — and it sat outside every guard here until 2026-08-02 simply
    because it lives at the repo root rather than under `documentation/`. It is not
    blanket-allowlisted.

    Its retroactive 0.4.0 record necessarily names deleted symbols, which is handled per
    block by `_is_historical` — but note that only the checks which opt into `_blocks()`
    apply that filter. `TestDocumentedErrorMessagesExist` deliberately does not: a fence
    showing an error message claims the code produces it regardless of what the
    surrounding paragraph discusses.
    """
    files = [README, CHANGELOG]
    for path in DOCS.rglob("*.md"):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if any(rel.startswith(prefix) for prefix in _HISTORICAL_ALLOWLIST):
            continue
        files.append(path)
    return files


class TestDocumentationMatchesCode:

    def test_no_doc_lists_a_deleted_module_path(self):
        """Guards the README-structure-tree-listing-evaluation_manager.py drift.

        Block-aware since 2026-08-02, matching the deleted-class check below. It was
        whole-file, which meant no document could ever *say* a path had been removed —
        `CHANGELOG.md`'s "Removed" section names `adapters/pandas.py` because naming
        what was removed is precisely a changelog's job. The claim this guards against
        is presenting a deleted path as **current**, and a block framed as history does
        not do that.
        """
        offenders = []
        for path in _doc_files():
            for lineno, block in _blocks(path.read_text()):
                if _is_historical(block):
                    continue
                for name in _DELETED_MODULE_PATHS:
                    if name in block:
                        offenders.append(
                            f"{path.relative_to(REPO_ROOT)}:{lineno} lists '{name}'"
                        )
        assert not offenders, (
            "Documentation lists module paths deleted in Phase 3:\n  "
            + "\n  ".join(offenders)
        )

    def test_readme_makes_no_legacy_config_key_support_claim(self):
        """The exact C-29 regression guard.

        Published 0.4.0's README told users legacy keys still worked and warned. The
        shim had already been deleted, so those keys silently produced empty reports.

        Matched on the present-tense support claim: prose *explaining* that the old
        documentation was wrong necessarily quotes it, and must not trip this.
        """
        text = README.read_text()
        claim = re.compile(
            r"[Ll]egacy keys still work\b(?!ed)",   # "still work"/"still works", not "still worked"
        )
        assert not claim.search(text), (
            "README claims legacy config keys still work. They were removed in 0.4.0 "
            "and now raise at NativeEvaluator.__init__ (register C-29)."
        )

    def test_readme_states_legacy_keys_were_removed(self):
        """The positive half: silence is not enough, the README must say what happened."""
        text = README.read_text()
        assert re.search(r"[Ll]egacy keys were REMOVED|removed in 0\.4\.0", text), (
            "README must state explicitly that legacy config keys were removed in "
            "0.4.0, so a migrating user is not left guessing."
        )

    def test_legacy_compatibility_default_matches_the_code(self):
        """CIC §3 contradicted §11 and the code on this exact flag."""
        from views_evaluation.evaluation.native_evaluator import NativeEvaluator

        actual = inspect.signature(NativeEvaluator.evaluate).parameters[
            "legacy_compatibility"
        ].default
        cic = (CICS / "NativeEvaluator.md").read_text()

        documented = re.search(
            r"legacy_compatibility`? flag \(\*\*?default `(True|False)`",
            cic,
        )
        assert documented, (
            "NativeEvaluator.md no longer states the legacy_compatibility default; "
            "it must, because the contract previously contradicted the code on it."
        )
        assert (documented.group(1) == "True") is (actual is True), (
            f"CIC documents legacy_compatibility default as {documented.group(1)} "
            f"but the code default is {actual}"
        )

    def test_documented_required_identifiers_match_the_code(self):
        """CIC §4 omitted 'unit', which _validate has always required."""
        from views_evaluation.evaluation import evaluation_frame

        src = inspect.getsource(evaluation_frame)
        match = re.search(r"required_keys\s*=\s*\{([^}]*)\}", src)
        assert match, "could not locate required_keys in EvaluationFrame._validate"
        enforced = set(re.findall(r"'(\w+)'", match.group(1)))

        cic = (CICS / "NativeEvaluator.md").read_text()
        identifier_line = next(
            (ln for ln in cic.splitlines() if "Identifier Presence" in ln), ""
        )
        for key in enforced:
            assert f"`{key}`" in identifier_line, (
                f"EvaluationFrame enforces identifier '{key}' but "
                f"NativeEvaluator.md §4 does not list it"
            )

    def test_no_doc_or_docstring_names_a_deleted_class_as_current(self):
        """Python docstrings were unguarded, and that is how C-02's record went stale.

        `config_schema.py` named `EvaluationManager._validate_config` as the runtime
        validator for four months after that class was deleted — through a Phase 3
        purge, a strategic review, and an epic that rewrote the surrounding module.
        The .md-only checks above could not see it.
        """
        import views_evaluation
        pkg_root = Path(views_evaluation.__file__).parent
        offenders = []
        for py in pkg_root.rglob("*.py"):
            for lineno, block in _blocks(py.read_text()):
                if _is_historical(block):
                    continue
                for name in _DELETED_CLASS_NAMES:
                    if name in block:
                        offenders.append(
                            f"{py.relative_to(pkg_root.parent)}:{lineno} names '{name}'"
                        )
        assert not offenders, (
            "Source docstrings/comments present a deleted class as current:\n  "
            + "\n  ".join(offenders)
        )

    def test_documentation_uses_current_metric_names(self):
        """Metric tokens in prose must exist in METRIC_CATALOG.

        `physical_architecture_standard.md` listed `Brier_point`/`Brier_sample` for four
        months after PR #18 renamed them — despite a reviewer on that PR explicitly
        asking for the reference to be updated.
        """
        from views_evaluation.evaluation.metric_catalog import METRIC_CATALOG

        # Derive the families to police from the catalog rather than hardcoding them, so
        # a future underscore-family metric is covered without editing this test. Only
        # capitalised prefixes qualify: `y_hat_bar` would contribute the prefix "y",
        # which would match `y_true`/`y_pred` throughout the prose.
        prefixes = sorted(
            {m.split("_")[0] for m in METRIC_CATALOG if "_" in m and m[0].isupper()}
        )
        pattern = r"\b(?:" + "|".join(prefixes) + r")_\w+"

        # Prose that explains a rename necessarily quotes the old name; _is_historical
        # allows that framing and flags a token presented as current.
        offenders = []
        for path in _doc_files():
            for lineno, block in _blocks(path.read_text()):
                if _is_historical(block):
                    continue
                for token in set(re.findall(pattern, block)):
                    if token not in METRIC_CATALOG:
                        offenders.append(
                            f"{path.relative_to(REPO_ROOT)}:{lineno} names '{token}'"
                        )
        assert not offenders, (
            "Documentation presents metric tokens absent from METRIC_CATALOG as current:\n  "
            + "\n  ".join(sorted(offenders))
            + f"\nValid: {sorted(METRIC_CATALOG)}"
        )

    def test_every_genome_param_is_consumed_by_its_function(self):
        """MetricCatalog's 'genome honesty' invariant.

        Its absence let Ignorance declare low_bin/high_bin — parameters NumPy silently
        ignores whenever bins is a sequence, so they were required configuration that
        changed nothing (register C-28b).
        """
        from views_evaluation.evaluation.metric_catalog import METRIC_CATALOG

        offenders = []
        for name, spec in METRIC_CATALOG.items():
            if not spec.genome:
                continue
            params = inspect.signature(spec.function).parameters
            if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
                declared = {
                    p for p, v in params.items()
                    if v.kind is inspect.Parameter.KEYWORD_ONLY
                }
            else:
                declared = set(params)
            missing = set(spec.genome) - declared
            if missing:
                offenders.append(f"{name}: declares {sorted(missing)}, not a parameter")
        assert not offenders, (
            "Genome parameters not consumed by their metric function:\n  "
            + "\n  ".join(offenders)
        )


class TestIntentContractRegistry:
    """ADR-021: the Active Contracts list and the CIC files must agree, both ways."""

    def _listed(self):
        readme = (CICS / "README.md").read_text()
        return set(re.findall(r"^- `([A-Z]\w*\.md)`", readme, re.MULTILINE))

    def _on_disk(self):
        return {
            p.name for p in CICS.glob("*.md")
            if p.name not in {"README.md", "cic_template.md"}
        }

    def test_every_listed_contract_exists(self):
        missing = self._listed() - self._on_disk()
        assert not missing, f"CICs/README.md lists non-existent contracts: {sorted(missing)}"

    def test_every_contract_is_listed(self):
        unlisted = self._on_disk() - self._listed()
        assert not unlisted, (
            f"Intent Contracts exist but are absent from the Active Contracts list "
            f"in CICs/README.md: {sorted(unlisted)}"
        )


class TestAdrRegistry:

    def test_every_adr_file_is_indexed(self):
        """A new ADR that nobody can find from the index is effectively invisible."""
        adr_dir = DOCS / "ADRs"
        readme = (adr_dir / "README.md").read_text()
        on_disk = {
            p.name for p in adr_dir.glob("*.md")
            if p.name not in {"README.md", "adr_template.md"}
        }
        missing = {name for name in on_disk if name not in readme}
        assert not missing, (
            f"ADRs exist but are not indexed in ADRs/README.md: {sorted(missing)}"
        )

    @pytest.mark.parametrize("adr", ["015_degenerate_and_empty_results.md",
                                     "022_evolution_and_stability.md"])
    def test_governing_adrs_are_accepted(self, adr):
        """These two govern behaviour asserted elsewhere in the suite."""
        text = (DOCS / "ADRs" / adr).read_text()
        assert re.search(r"^\*\*Status:\*\*\s*Accepted", text, re.MULTILINE), (
            f"{adr} is not in Accepted status, but code depends on its rulings"
        )


class TestLoggingScopeContract:
    """The Level-0/Level-1 split from logging standard §5.1, asserted against code."""

    _LEVEL_0 = [
        "evaluation_frame", "native_evaluator", "evaluation_report",
        "metric_catalog", "native_metric_calculators",
    ]

    @pytest.mark.parametrize("module_name", _LEVEL_0)
    def test_level_zero_modules_declare_no_module_logger(self, module_name):
        import importlib
        module = importlib.import_module(f"views_evaluation.evaluation.{module_name}")
        src = inspect.getsource(module)
        # A module-level `logger = logging.getLogger(__name__)` is the violation.
        assert not re.search(r"^logger\s*=\s*logging\.getLogger", src, re.MULTILINE), (
            f"{module_name} is Level 0 and must not maintain a module logger "
            f"(logging standard §5.1)"
        )

    def test_standard_names_both_levels_explicitly(self):
        """The old scope note named three classes and predated metric_frame.py,
        leaving Level 1 uncovered by omission. It must now be explicit."""
        text = (DOCS / "standards" / "logging_and_observability_standard.md").read_text()
        assert "metric_frame.py" in text, (
            "logging standard §5.1 must name metric_frame.py explicitly, so the "
            "Level-1 gap cannot silently reopen"
        )
        assert "Level 1" in text and "Level 0" in text


class TestDocumentedErrorMessagesExist:
    """Every exception message a doc shows in a code fence must exist in the package.

    Added 2026-08-02 after the PR #45 review found the README advertising

        ValueError: Unknown evaluation config key(s): ['targets'].

    a message no code in this package produces. `targets` is a *legacy* key, so the
    real raise names it as legacy and gives the rename. The README had invented an
    error and shown it to users as a promise.

    This is deliberately a **generalising** check rather than another hand-picked
    assertion. Register C-34 records why: C-29 was closed by adding a targeted test to
    this very file guarding this very README, and that test did not catch the new false
    promise added to that same README in the same epic. Targeted tests cover the claim
    someone thought to encode. This one covers every claim of its shape, including the
    ones nobody has written yet.

    **What this does NOT check — read before trusting a pass.** It compares the *literal*
    text of a message against the package source. Interpolated values are deleted before
    comparison, because they are runtime values that do not appear in the source at all.
    So a documented message can have the right literal skeleton and wrong values and pass.
    That is not hypothetical: the corrected README initially rendered the legacy-key error
    as `'targets' -> 'regression_targets'` when the code produces
    `'targets' -> 'regression_targets' or 'classification_targets'`, and this check passed,
    because the differing half sits inside the quoted span it blanks. Verifying values
    requires executing the documented scenario and comparing the raised string. Tracked as
    the residual half of C-37.
    """

    # A literal run must be this long to be worth asserting on. Measured across every
    # `raise` literal in the package, NOT guessed — two earlier justifications of this
    # constant were written from assumption and were both measurably false. The shortest
    # longest-run of any real message is 14 ("Metric '{m}' failed for ({task}, ...)"),
    # and eight messages sit under 20, including "Input contains NaN" (18) and
    # "Target {target} not found in config" (19). A threshold of 20 put all eight out of
    # scope, so 12 is chosen to keep every message in this package assertable.
    #
    # The asymmetry that makes a low threshold safe: a run too generic to be distinctive
    # can only produce a false PASS (it coincidentally matches something), never a false
    # FAILURE. Raising this number trades detection for nothing.
    _MIN_SEGMENT = 12

    @staticmethod
    def _package_raises():
        """[(exception_type, literal_message)] for every `raise X("...")` in the package.

        Keyed by exception type, not a flat text corpus, for two reasons found by
        adversarially probing the flat-corpus version:

        * A doc could show a real message under the **wrong exception type** and pass,
          because the type was captured and printed but never compared to anything.
        * A fabricated message could borrow a real message's boilerplate tail and pass,
          because matching was `any(segment in corpus)` — one shared segment exonerated
          the whole message. Appending "See the README migration table." to the exact
          C-34 message this class was written to catch was enough to defeat it.

        Messages are implicitly-concatenated f-string fragments wrapped for line length,
        so fragments are joined before extraction — the source's wrap points have nothing
        to do with the doc's.
        """
        out = []
        pat = re.compile(
            r"raise\s+(\w+)\s*\(\s*((?:f?\"(?:[^\"\\]|\\.)*\"\s*)+)", re.S
        )
        for path in sorted((REPO_ROOT / "views_evaluation").rglob("*.py")):
            src = re.sub(r'"\s*\n\s*f?"', "", path.read_text())
            for m in pat.finditer(src):
                literals = re.findall(r'f?"((?:[^"\\]|\\.)*)"', m.group(2))
                out.append((m.group(1), re.sub(r"\s+", " ", "".join(literals))))
        return out

    @staticmethod
    def _documented_messages():
        """(file, exception, message) for every `SomeError: ...` shown in a fence.

        `Warning` and `Exception` are matched as well as `Error`. ADR-022 is largely
        about when a `DeprecationWarning` is owed, and C-29 was a README promising a
        `DeprecationWarning` no code emitted — exactly the shape this check exists to
        catch, and exactly the shape an `Error`-only pattern would miss.
        """
        fence = re.compile(r"^\s*(?:> )?```.*?^\s*(?:> )?```", re.M | re.S)
        found = []
        for path in _doc_files():
            text = path.read_text()
            # Deliberately NOT filtered by `_is_historical`. That helper answers
            # "is this prose explaining a removal", which is a per-block question and
            # is irrelevant here: showing an error message in a fence claims the code
            # produces it, whatever the surrounding paragraph is discussing. Files that
            # legitimately quote historical messages (ADRs, reports) are already
            # excluded at file level by `_HISTORICAL_ALLOWLIST` in `_doc_files()`.
            #
            # This check was briefly written with a whole-file `_is_historical` guard
            # and was vacuous as a result: README contains "were removed", so every
            # fence in the file it was written to guard got skipped.
            for block in fence.finditer(text):
                body = "\n".join(
                    re.sub(r"^\s*> ?", "", line) for line in block.group().splitlines()
                )
                # finditer, not search: a fence may show several messages, and checking
                # only the first let a fabricated second line through.
                for m in re.finditer(
                    r"^(\w*(?:Error|Warning|Exception)): (.+?)"
                    r"(?=^\w*(?:Error|Warning|Exception): |^```|\Z)",
                    body, re.M | re.S,
                ):
                    found.append((path, m.group(1), re.sub(r"\s+", " ", m.group(2)).strip()))
        return found

    def test_every_documented_error_message_exists_in_the_package(self):
        raises = self._package_raises()
        documented = self._documented_messages()
        # Non-vacuity floor. Every other assertion here iterates `documented`, so if
        # extraction ever returns nothing — a changed fence style, a path that stops
        # matching, a regex edit — the whole check passes while testing nothing. That
        # has already happened twice (C-37). Pin the count so it cannot happen silently.
        assert len(documented) >= 2, (
            f"expected at least the two README error fences, found {len(documented)} — "
            f"the extractor has stopped seeing documented messages, so this guard is "
            f"no longer guarding anything"
        )
        failures, unverifiable = [], []
        for path, exc, message in documented:
            # Delete interpolated values, THEN split — only the literal message text
            # is evidence. Splitting on the delimiters alone also yields the values
            # between them, and a value can be a real identifier: the fabricated
            # message this check was written to catch quoted
            # 'classification_point_metrics', a genuine config key present in the
            # source, and that one segment made the whole invented message pass.
            # A doc may truncate a long message with a trailing ellipsis. That is honest
            # abbreviation, not a claim, so drop it before segmenting — otherwise the
            # final segment carries a "..." that no source message contains.
            body = re.sub(r"\s*\.\.\.\s*$", "", message)
            literal = re.sub(r"\[[^\]]*\]|\{[^}]*\}|'[^']*'|\"[^\"]*\"", "\x00", body)
            segments = [
                re.sub(r"\s+", " ", s).strip() for s in literal.split("\x00")
            ]
            segments = [s for s in segments if len(s) >= self._MIN_SEGMENT]
            if not segments:
                # Do NOT `continue` silently. A message with no assertable literal run
                # is a message this check cannot verify, and passing quietly is
                # indistinguishable from verifying it — the C-37 vacuity mode. Fail and
                # say so, so the reader either shortens _MIN_SEGMENT deliberately or
                # verifies that message by hand.
                unverifiable.append(
                    f"{path.relative_to(REPO_ROOT).as_posix()} documents "
                    f"`{exc}: {message[:70]}` whose longest literal run is under "
                    f"{self._MIN_SEGMENT} chars — this check cannot verify it"
                )
                continue
            # Require ONE source message, raised as THIS exception type, containing
            # EVERY literal run. `any(segment)` across a flat corpus was the hole:
            # boilerplate shared with a real message exonerated a fabricated one.
            same_type = [msg for exc_type, msg in raises if exc_type == exc]
            if not any(all(s in cand for s in segments) for cand in same_type):
                rel = path.relative_to(REPO_ROOT).as_posix()
                near = "; ".join(f'"{s[:45]}"' for s in segments if
                                 not any(s in cand for cand in same_type))
                failures.append(
                    f"{rel} documents `{exc}: {message[:70]}...` but no single "
                    f"`raise {exc}(...)` in views_evaluation/ produces it"
                    + (f". Absent literal run(s): {near}" if near else
                       f". Every run appears, but never together in one {exc}")
                )
        assert not failures + unverifiable, (
            "Documentation shows exception messages the code does not produce:\n  "
            + "\n  ".join(failures + unverifiable)
        )
