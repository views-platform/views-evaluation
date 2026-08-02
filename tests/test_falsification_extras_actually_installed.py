"""FAILING TEST — falsification audit round 3, 2026-08-02.

Claim under audit: "it is safe to shut down this session." **FALSIFIED.**

The fix for the CI coverage gap is verified one level away from the thing that matters.

`tests/test_falsification_ci_coverage_gap.py` asserts that the string `--all-extras`
appears in `run_pytest.yml`. That is not the same as asserting the extras were
**installed**, and it is certainly not the same as asserting the 66 optional-dependency
tests **ran**.

The failure mode it leaves open is the original one, unchanged:

    poetry install --all-extras        # partially fails, or resolves without views-frames
    poetry run pytest tests/           # test_metric_frame.py + test_evaluation_report.py
                                       #   silently skip via module-level importorskip
    -> CI reports "295 passed" and goes green
    -> test_falsification_ci_coverage_gap.py still PASSES, because the yml text is intact

`set -e` does not help: it guards the `pytest` step, and a `poetry install` that resolves
successfully without an optional package is not a non-zero exit. The whole defect this
session spent a round finding — a green CI run that never touched the `MetricFrame`
cross-repo contract — can therefore recur with the guard in place and green.

This is register C-37's shape at one further remove: the guard on the guard is vacuous.
A check that reads a config file can only prove intent. Proving effect requires asserting
against the run.

Fix: add an explicit post-install verification step to `run_pytest.yml` that imports every
extra-provided package and exits non-zero if any is missing, e.g.

    - name: Verify optional extras are installed
      run: poetry run python -c "import views_frames, pandas"

so a failed or incomplete extras install stops CI at that step instead of quietly
shrinking the suite. This test is expected to FAIL until such a step exists. Do not "fix"
it by relaxing the assertion.
"""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "run_pytest.yml"
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _extras_table(text):
    """The extras mapping, from either pyproject layout.

    Poetry warns that `[tool.poetry.extras]` is deprecated in favour of PEP 621's
    `[project.optional-dependencies]`. Both are read, so a migration does not turn this
    guard into a false alarm — which would be its own small version of the failure it
    exists to catch.

    Sections are split explicitly rather than matched with a regex. The regex form
    `(?:[^\[].*\n|\n)*` looks like it stops at the next `[section]` and does not:
    `[^\[]` happily matches a newline, so the pattern steps over a blank line onto the
    header and keeps going — it swallowed `[build-system]` and reported `poetry-core`
    as an extra.
    """
    sections, current = {}, None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped[1:-1]
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    for name in ("tool.poetry.extras", "project.optional-dependencies"):
        body = "\n".join(sections.get(name, []))
        if re.search(r"^\w[\w-]*\s*=\s*\[", body, re.M):
            return body
    return None


def _extra_provided_imports():
    """Import names supplied by extras, derived from pyproject rather than hardcoded."""
    table = _extras_table(PYPROJECT.read_text())
    assert table, (
        "no extras table found under either [tool.poetry.extras] or "
        "[project.optional-dependencies]; pyproject layout changed"
    )
    names = set()
    for m in re.finditer(r'^\w[\w-]*\s*=\s*\[([^\]]*)\]', table, re.M):
        for pkg in m.group(1).split(","):
            # PEP 621 entries carry version specifiers ("views-frames>=1.10.2,<2");
            # keep only the distribution name, then map it to its import name.
            pkg = re.split(r"[<>=!~\[; ]", pkg.strip().strip("\"'"))[0].strip()
            if pkg:
                names.add(pkg.replace("-", "_"))
    assert names, "no extra-provided packages parsed"
    return names


class TestCiProvesExtrasWereInstalled:

    def test_workflow_verifies_extras_after_installing_them(self):
        """CI must fail loudly when an extra is absent, not shrink the suite quietly.

        Asserting `--all-extras` is present in the workflow proves intent. It does not
        prove the packages arrived, and every guard downstream of them is a
        module-level `importorskip` that turns absence into silence.
        """
        workflow = WORKFLOW.read_text()
        required = _extra_provided_imports()

        # A verification step is any run-line that imports the extra packages outside
        # the test suite itself, so a missing extra is an error rather than a skip.
        verifies = set()
        for line in workflow.splitlines():
            stripped = line.strip()
            if "pytest" in stripped or stripped.startswith("#"):
                continue
            if "import" not in stripped:
                continue
            for pkg in required:
                if re.search(rf"\b{re.escape(pkg)}\b", stripped):
                    verifies.add(pkg)

        missing = sorted(required - verifies)
        assert not missing, (
            f"run_pytest.yml installs extras but never verifies they arrived. "
            f"No step imports {missing} outside the test suite, so a `poetry install "
            f"--all-extras` that resolves without them leaves "
            f"tests/test_metric_frame.py (44 tests) and "
            f"tests/test_evaluation_report.py (22 tests) silently skipped via "
            f"module-level importorskip — the exact defect this session already fixed "
            f"once, able to recur while test_falsification_ci_coverage_gap.py stays "
            f"green. Add: `poetry run python -c \"import "
            f"{', '.join(sorted(required))}\"` after the install step."
        )
