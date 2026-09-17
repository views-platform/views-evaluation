"""Guard: CI must PROVE the optional extras arrived, not merely request them.

Written as a failing test by falsification round 3 (2026-08-02), kept as a permanent
guard once `run_pytest.yml` gained an explicit verification step.

The defect it caught: `test_falsification_ci_coverage_gap.py` asserts that
`--all-extras` appears in the workflow. That proves intent, not effect. `poetry install`
exits 0 even when it resolves without an optional package, and every downstream guard is
a module-level `pytest.importorskip` that turns absence into silence — so the 67
extras-gated tests could go dark again while the guard against that stayed green. A check
that reads a config file can only prove intent; proving effect requires asserting against
the run.

**This guard has itself been defeated twice**, which is why it is as strict as it is.
Round one accepted any line containing the package names — an `echo`, a step `name:`, a
trailing comment, `|| true`. Round two still accepted `continue-on-error: true` (the
GitHub-native `|| true`), `|| echo skip`, `set +o errexit`, `if ! ...; then true; fi`, a
backgrounded command, a second verifying step placed *after* pytest, and a verify step in
a different job entirely. It now requires a single step, in the same job as install and
pytest and ordered between them, that actually executes a `python -c` import of every
extra-provided package with its exit status intact.

Do not "fix" a failure here by relaxing the assertion: it means real coverage has gone
dark, or the proof that it has not is no longer a proof.
"""
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "run_pytest.yml"
PR_WORKFLOW = WORKFLOW
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
    table = _extras_table(PYPROJECT.read_text(encoding="utf-8"))
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


def _strip_comment(s):
    """Drop a trailing YAML/shell comment, ignoring `#` inside quotes.

    `run: python -c "print(1)"  # import views_frames, pandas` is a comment in YAML, but
    an earlier version of this guard read the whole line and counted the comment as the
    command.
    """
    out, quote = [], None
    for i, ch in enumerate(s):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "#" and (i == 0 or s[i - 1].isspace()):
            break
        out.append(ch)
    return "".join(out).rstrip()


def _commands(run):
    """Executable command lines from a run block, with inert ones dropped.

    `echo 'python -c "import views_frames, pandas"'` prints a command; it does not run
    one. Matching the text of a command anywhere in the step counted that as proof.
    """
    cmds = []
    for line in run.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for part in re.split(r"&&|;", line):
            part = part.strip()
            if part and not re.match(r"^(echo|printf|:)\b", part):
                cmds.append(part)
    return cmds


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
    job = None

    def _flush():
        if current is not None:
            current["run"] = "\n".join(run_lines).strip()
            steps.append(current)

    for raw in text.splitlines():
        stripped = raw.strip()
        m_job = re.match(r"^  (\w[\w-]*):\s*$", raw)
        if m_job and m_job.group(1) not in ("steps",):
            job = m_job.group(1)
        if current is not None and re.match(r"^\s*(continue-on-error|if):", raw):
            # Any value: `${{ true }}`, `True`, a trailing comment, or the value on the
            # next line all reach GitHub as "this step cannot fail / may be skipped".
            current["continue_on_error"] = True
        m_step = re.match(r"^(\s*)-\s+name:\s*(.*)$", raw)
        if m_step:
            _flush()
            current, run_lines, in_block = (
                {"name": m_step.group(2).strip(), "job": job, "continue_on_error": False}, [], False)
            continue
        if current is None:
            continue
        m_run = re.match(r"^(\s*)run:\s*(\|-?|>-?)?\s*(.*)$", raw)
        if m_run:
            in_block = bool(m_run.group(2))
            block_indent = None
            if m_run.group(3):
                run_lines.append(_strip_comment(m_run.group(3)))
            continue
        if in_block and stripped:
            indent = len(raw) - len(raw.lstrip())
            if block_indent is None:
                block_indent = indent
            if indent >= block_indent:
                if not stripped.startswith("#"):
                    # Strip trailing comments too: `python -c "print(1)"  # import pandas`
                    # passed the first version of this guard.
                    run_lines.append(_strip_comment(stripped))
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
        steps = _workflow_steps(WORKFLOW.read_text(encoding="utf-8"))
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
        # Every way an executed command's failure can be prevented from failing the job.
        # Each was confirmed to defeat an earlier version of this guard.
        SUPPRESSED = (
            r"\|\|",                 # `|| true`, `|| :`, `|| echo skip` — any fallback
            r"^\s*set \+(e|o\s+errexit)",   # errexit disabled
            r"^\s*if\s|;\s*then\b",  # wrapped in a conditional that swallows status
            r"&\s*$",                 # backgrounded
            r";\s*exit\s+0",          # status discarded
        )

        def _verified_by(step):
            run = step["run"]
            if step.get("continue_on_error"):
                return set()  # GitHub-native `|| true`: the step cannot fail the job
            if "pytest" in run:
                return set()
            if any(re.search(pat, run, re.M) for pat in SUPPRESSED):
                return set()
            found = set()
            for cmd in _commands(run):
                if not re.search(r"python[0-9.]*\b.*\s-c\b", cmd):
                    continue
                found |= {p for p in required
                          if re.search(rf"\bimport\b[^\n]*\b{re.escape(p)}\b", cmd)}
            return found

        # ONE step must cover ALL required packages, in the same job as install and
        # tests. Accumulating across steps let a second import step sit after pytest and
        # still count; allowing a different job made the ordering window meaningless.
        verified, verified_at = set(), None
        for i, step in enumerate(steps):
            got = _verified_by(step)
            if got >= required and step.get("job") == steps[install_at].get("job") == steps[tests_at].get("job"):
                verified, verified_at = got, i
                break

        missing = sorted(required - verified)
        assert not missing, (
            f"run_pytest.yml installs extras but never proves they arrived. No step "
            f"EXECUTES an import of {missing} (an `echo`, a step `name:`, or a command "
            f"neutered with `|| true` does not count), so a `poetry install "
            f"--all-extras` that resolves without them leaves every test module that "
            f"import-skips on those extras silently skipped whole. Add: `poetry run "
            f"python -c \"import {', '.join(sorted(required))}\"` between the install "
            f"and test steps."
        )
        assert install_at < verified_at < tests_at, (
            f"the extras verification runs at step {verified_at}, outside the window it "
            f"has to guard (install={install_at}, tests={tests_at}). Verifying after the "
            f"tests have already run proves nothing about the run that just happened."
        )


PUBLISH_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish_package.yml"
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"


def _significant_lines(text):
    """The workflow with comments and blank lines removed — what GitHub executes."""
    return [line.rstrip() for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")]


class TestPublishGateIsReal:
    """Register C-36, mechanical half (2026-09-17). `publish_package.yml` used to go from
    the GitHub release event straight to `poetry publish`. It now `needs:` a job that
    runs `run_pytest.yml` by reference — one recipe, nothing copied that could drift.

    The guard is an allowlist, not a parser: each of the two workflow files that together
    form the release gate must be, line for line, the text below. Two review passes and a
    guard audit showed that any denylist over a hand-rolled YAML reader loses to the next
    feature it does not model (`env:`, `shell:`, folded scalars, `if: always()` on the
    publish step, `continue-on-error` on the pytest step or the job, a nameless `- run:`
    step, a re-indented file, `workflow_call` moved under `jobs:`). Holding both files to
    a known-good text catches all of those at once; the cost — updating the copy here when
    a workflow is edited on purpose — is the point: that edit is the review moment for
    anything that can let a red suite publish. Whitespace, blank lines and comments are
    not significant.

    A third guard closes the other way round the gate: no workflow file other than the
    gated one may publish at all.

    These guards run inside the suite they gate, so they protect the pull-request path. A
    release cut from a branch no PR ever ran on cannot be caught by a test; that residual
    is recorded in C-36 with its GitHub-native fix.
    """

    EXPECTED = {
        "publish_package.yml": """\
name: Publish Package
on:
  release:
    types:
        - published
  workflow_dispatch: # enables manual triggering
jobs:
  test:
    uses: ./.github/workflows/run_pytest.yml
  publish:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - name: Checkout repository
      uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"
    - name: Install Poetry
      run: |
          curl -sSL https://install.python-poetry.org | python3 -
    - name: Install Dependencies
      run: |
          python -m pip install --upgrade pip
          python -m pip install packaging
    - name: Validate Version
      run: |
        latest_version=$(curl -s https://pypi.org/pypi/views-evaluation/json | jq -r .info.version)
        new_version=$(poetry version -s)
        python -c "from packaging.version import parse; assert parse('$new_version') > parse('$latest_version'), 'Version must be higher than $latest_version'"
    - name: Publish to PyPI
      run: poetry publish --build --username __token__ --password ${{ secrets.PYPI_TOKEN }}
""",
        "run_pytest.yml": """\
name: Run Pytest
on:
  push:
    branches:
      - main
      - development
  pull_request:
    branches:
      - main
      - development
  workflow_dispatch:
  workflow_call:
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout repository
      uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"
    - name: Install Poetry
      run: |
          curl -sSL https://install.python-poetry.org | python3 -
    - name: Install dependencies
      run: |
        poetry install --all-extras
    - name: Verify optional extras are installed
      run: poetry run python -c "import views_frames, pandas"
    - name: Run tests
      run: |
        set -e
        poetry run pytest tests/
    - name: Validate documentation consistency
      run: bash documentation/validate_docs.sh
""",
    }

    @pytest.mark.parametrize("name", sorted(EXPECTED))
    def test_release_gate_workflow_is_the_known_good_text(self, name):
        import difflib

        actual = _significant_lines((WORKFLOWS_DIR / name).read_text(encoding="utf-8"))
        expected = _significant_lines(self.EXPECTED[name])
        diff = "\n".join(difflib.unified_diff(expected, actual, "expected", name, lineterm=""))
        assert actual == expected, (
            f"{name} differs from the known-good release gate. If the change is deliberate, "
            f"update EXPECTED[{name!r}] in TestPublishGateIsReal in the same PR — that is "
            f"the review point for anything that can let a red suite publish:\n" + diff
        )

    def test_no_other_workflow_publishes(self):
        """A second workflow that uploads to PyPI without `needs:` would bypass the gate
        while the known-good text of the gated one stays intact."""
        publishers = sorted(
            p.name for p in WORKFLOWS_DIR.glob("*.y*ml")
            if re.search(r"poetry publish|pypi-publish|twine upload",
                         "\n".join(_significant_lines(p.read_text(encoding="utf-8"))))
        )
        assert publishers == ["publish_package.yml"], (
            f"workflows that publish: {publishers}; only publish_package.yml may, and it is "
            f"gated on the suite by `needs: test`"
        )
