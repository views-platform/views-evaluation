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
    r"""The extras mapping, from either pyproject layout.

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
    # PEP 621 FIRST. Poetry >= 2.0 ignores `[tool.poetry.extras]` whenever a `[project]`
    # table is present, so PEP 621 is authoritative when both exist. Checking the legacy
    # table first meant a half-finished migration — a stale `[tool.poetry.extras]` left
    # beside the real one — would shadow the authoritative table and make this guard
    # pass while the true extras went unverified.
    for name in ("project.optional-dependencies", "tool.poetry.extras"):
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


def _workflow_steps(text):
    r"""[{'name':..., 'run':...}] for each `- name:` step, in file order.

    Hand-rolled rather than `yaml.safe_load`: PyYAML is neither declared in
    `pyproject.toml` nor installed in the test environment, and depending on a package
    that merely happens to be present transitively is the mistake register C-19 records
    (scipy was an undeclared runtime import satisfied via scikit-learn). Adding a
    dependency to parse one file is the wrong trade; the subset of YAML a workflow step
    uses is small and stable.

    Handles both `run: cmd` and the `run: |` block scalar. Comment lines are dropped so
    that prose in a comment cannot be mistaken for a command — the previous version of
    this guard could be satisfied by text that never executed.
    """
    steps, current, run_lines, in_block, block_indent = [], None, [], False, None

    def _flush():
        if current is not None:
            current["run"] = "\n".join(run_lines).strip()
            steps.append(current)

    for raw in text.splitlines():
        stripped = raw.strip()
        m_step = re.match(r"^(\s*)-\s+name:\s*(.*)$", raw)
        if m_step:
            _flush()
            current, run_lines, in_block = {"name": m_step.group(2).strip()}, [], False
            continue
        if current is None:
            continue
        m_run = re.match(r"^(\s*)run:\s*(\|-?|>-?)?\s*(.*)$", raw)
        if m_run:
            in_block = bool(m_run.group(2))
            block_indent = None
            if m_run.group(3):
                run_lines.append(m_run.group(3))
            continue
        if in_block and stripped:
            indent = len(raw) - len(raw.lstrip())
            if block_indent is None:
                block_indent = indent
            if indent >= block_indent:
                if not stripped.startswith("#"):
                    run_lines.append(stripped)
                continue
            in_block = False
    _flush()
    return steps


class TestCiProvesExtrasWereInstalled:

    def test_workflow_verifies_extras_after_installing_them(self):
        """CI must fail loudly when an extra is absent, not shrink the suite quietly.

        Asserting `--all-extras` is present in the workflow proves intent. It does not
        prove the packages arrived, and every guard downstream of them is a
        module-level `importorskip` that turns absence into silence.
        """
        required = _extra_provided_imports()
        steps = _workflow_steps(WORKFLOW.read_text())
        assert steps, "no steps parsed from run_pytest.yml; its layout has changed"

        def _index(pred):
            return next((i for i, s in enumerate(steps) if pred(s)), None)

        install_at = _index(lambda s: "poetry install" in s["run"])
        tests_at = _index(lambda s: "pytest" in s["run"])
        assert install_at is not None, "no `poetry install` step found"
        assert tests_at is not None, "no pytest step found"

        # A verifying step must EXECUTE the imports. Matching any line containing the
        # package name accepted four inert impostors, each confirmed to slip through
        # the first version of this guard: `echo "import views_frames, pandas"`; the
        # words appearing only in a step's `name:`; `... || true`; and the step placed
        # after the tests it is supposed to gate. `|| true` is the realistic one — it is
        # the standard "unbreak the flaky step" edit, and it leaves the guard green
        # while disabling the check entirely. That is register C-37's vacuity shape
        # reproduced inside the test written to close it.
        def _verified_by(step):
            run = step["run"]
            if "pytest" in run or not re.search(r"python\b.*-c", run):
                return set()
            if re.search(r"\|\|\s*(true|:)\b", run) or re.search(r"^\s*set \+e", run, re.M):
                return set()  # exit status suppressed — proves nothing
            return {p for p in required if re.search(rf"\bimport\b[^\n]*\b{re.escape(p)}\b", run)}

        verified, verified_at = set(), None
        for i, step in enumerate(steps):
            got = _verified_by(step)
            if got:
                verified |= got
                if verified_at is None:
                    verified_at = i

        missing = sorted(required - verified)
        assert not missing, (
            f"run_pytest.yml installs extras but never proves they arrived. No step "
            f"EXECUTES an import of {missing} (an `echo`, a step `name:`, or a command "
            f"neutered with `|| true` does not count), so a `poetry install "
            f"--all-extras` that resolves without them leaves "
            f"tests/test_metric_frame.py (44 tests) and "
            f"tests/test_evaluation_report.py (22 tests) silently skipped via "
            f"module-level importorskip. Add: `poetry run python -c \"import "
            f"{', '.join(sorted(required))}\"` between the install and test steps."
        )
        assert install_at < verified_at < tests_at, (
            f"the extras verification runs at step {verified_at}, outside the window it "
            f"has to guard (install={install_at}, tests={tests_at}). Verifying after the "
            f"tests have already run proves nothing about the run that just happened."
        )
