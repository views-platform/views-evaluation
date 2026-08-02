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
DOCS = REPO_ROOT / "documentation"
CICS = DOCS / "CICs"

# File paths deleted in Phase 3. Documentation must not present them as if they exist.
#
# Deliberately matched as *paths*, not class names: the drift this guards against is a
# project-structure tree or file listing citing a module that is gone. Prose that
# explains why `EvaluationManager` was removed is legitimate and must not trip this.
_DELETED_MODULE_PATHS = [
    "evaluation_manager.py",
    "adapters/pandas.py",
    "adapters.pandas",
]

# Files where a historical mention is legitimate: change logs, decision records and
# the risk register all necessarily discuss things that no longer exist.
_HISTORICAL_ALLOWLIST = {
    "documentation/ADRs",             # ADRs record why things were removed
    "documentation/CICs/EvaluationReport.md",   # Evolution Notes
    "documentation/CICs/NativeEvaluator.md",    # Evolution Notes
    "documentation/contributor_protocols",
    "reports",                        # register + post-mortems are history by nature
}


def _doc_files():
    """All documentation prose, excluding files whose job is to record history."""
    files = [README]
    for path in DOCS.rglob("*.md"):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if any(rel.startswith(prefix) for prefix in _HISTORICAL_ALLOWLIST):
            continue
        files.append(path)
    return files


class TestDocumentationMatchesCode:

    def test_no_doc_lists_a_deleted_module_path(self):
        """Guards the README-structure-tree-listing-evaluation_manager.py drift."""
        offenders = []
        for path in _doc_files():
            text = path.read_text()
            for name in _DELETED_MODULE_PATHS:
                if name in text:
                    offenders.append(f"{path.relative_to(REPO_ROOT)} lists '{name}'")
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
