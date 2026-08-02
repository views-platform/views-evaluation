"""Guard: CI must install every extra the test suite import-skips on.

Written as a failing test by a falsification audit on 2026-08-02 (round 2 of
"we are ready to shut this session down" — FALSIFIED), then kept as a permanent guard
once CI was fixed to `poetry install --all-extras`.

The defect it caught: **the `MetricFrame` contract was not tested in CI.**

`views-frames` is an **optional** extra (declared `optional = true` in `pyproject.toml`
and exposed as the `frames` extra; the version constraint has since moved and is not
restated here, because restating it is how this docstring went stale before). CI runs a bare
`poetry install`, which does not install optional dependencies. `tests/test_metric_frame.py`
guards with a **module-level** `pytest.importorskip("views_frames")` at line 19, so the
entire file is skipped rather than failing.

Measured, not inferred:

    local (frames installed) : 359 passed   [measured at the v0.5.0 tag]
    CI   (frames absent)     : 293 passed, 4 skipped
    delta                    : 66 tests that never run in CI
    of which test_metric_frame.py = 44 and test_evaluation_report.py = 22

    (These read 361/68 until 2026-08-02, against a docstring claiming they were
    "measured, not inferred". They were not: 361 was a later local count that included
    tests added after the tag, and 68 followed from it. 44 + 22 = 66 is forced by the
    two per-file counts on the line above.)

This matters more than the raw count. ADR-022 §1 designates the `MetricFrame` on-disk
format and axis vocabulary a **cross-repo contract**, "treated as public API regardless
of `__all__`". It is the artifact `views-reporting` and `views-pipeline-core` consume.
Its 44 guards, plus the emit path in `test_evaluation_report.py` and the end-to-end probe
in `test_falsification_legacy_compatibility.py`, are exactly the tests that do not run on
any pull request.

The skip is silent by construction: a green CI run reporting "293 passed" is
indistinguishable from one where the contract is verified. That is register C-37's
vacuity mode, hoisted from a single guard to a third of the suite.

Fixed by installing the extras in CI (`poetry install --all-extras`). This guard derives
the required extras from `pyproject.toml` rather than hardcoding them, so adding a new
optional dependency with a new import-skip cannot quietly reopen the gap. Do not "fix" a
failure here by relaxing the assertion — the failure means real coverage has gone dark.
"""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "run_pytest.yml"
PYPROJECT = REPO_ROOT / "pyproject.toml"


class TestCiRunsTheContractTests:

    def test_ci_installs_every_extra_the_test_suite_depends_on(self):
        """Any extra a test module import-skips on must be installed in CI.

        Derived, not hardcoded: the required extras are read from `pyproject.toml`'s
        `[tool.poetry.extras]` and matched against the packages the suite skips on, so
        adding a new optional dependency with a new import-skip cannot quietly reopen
        this gap.
        """
        pyproject = PYPROJECT.read_text()
        # Both layouts: poetry warns `[tool.poetry.extras]` is deprecated in favour of
        # PEP 621's `[project.optional-dependencies]`. Reading only the deprecated one
        # would make a routine migration fail this guard as a false alarm.
        # Sections split explicitly, not by regex: `(?:[^\[].*\n|\n)*` looks like it
        # stops at the next `[section]` but `[^\[]` matches a newline, so it steps over
        # the header and keeps going.
        sections, current = {}, None
        for line in pyproject.splitlines():
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                current = stripped[1:-1]
                sections[current] = []
            elif current is not None:
                sections[current].append(line)
        # PEP 621 FIRST: poetry >= 2.0 ignores [tool.poetry.extras] when a [project]
        # table exists, so a stale legacy table left by a half-finished migration must
        # not shadow the authoritative one.
        extras_block = None
        for header in ("project.optional-dependencies", "tool.poetry.extras"):
            body = "\n".join(sections.get(header, []))
            if re.search(r"^\w[\w-]*\s*=\s*\[", body, re.M):
                extras_block = body
                break
        assert extras_block, (
            "no extras table found under [tool.poetry.extras] or "
            "[project.optional-dependencies] in pyproject.toml"
        )
        # extra name -> the distributions it pulls, e.g. frames -> views-frames
        extras = {
            m.group(1): m.group(2)
            for m in re.finditer(r'^(\w[\w-]*)\s*=\s*\[([^\]]*)\]', extras_block, re.M)
        }
        assert extras, "no extras parsed; the pyproject layout has changed"

        # Which optional packages does the suite import-skip on?
        skipped_on = set()
        for test in (REPO_ROOT / "tests").glob("*.py"):
            for m in re.finditer(
                r'importorskip\(\s*[\'"]([\w.]+)[\'"]', test.read_text()
            ):
                skipped_on.add(m.group(1).split(".")[0])

        # Map those back to the extras that provide them (views_frames -> views-frames).
        def _import_names(pkgs):
            # PEP 621 entries carry version specifiers ("views-frames>=1.10.2,<2");
            # keep only the distribution name before mapping it to an import name.
            for p in pkgs.split(","):
                name = re.split(r"[<>=!~\[; ]", p.strip().strip("\"'"))[0].strip()
                if name:
                    yield name.replace("-", "_")

        required = {
            name for name, pkgs in extras.items()
            if any(n in skipped_on for n in _import_names(pkgs))
        }
        assert required, (
            "no test import-skips on a declared extra; if that is genuinely true this "
            "check is obsolete, but verify before deleting it"
        )

        workflow = WORKFLOW.read_text()
        install = re.search(r"poetry install[^\n]*", workflow)
        assert install, "no `poetry install` step found in run_pytest.yml"
        install_cmd = install.group()
        installs_all = "--all-extras" in install_cmd
        named = set(re.findall(r"--extras[= ]([\w]+)", install_cmd))

        missing = sorted(required - named) if not installs_all else []
        assert not missing, (
            f"CI runs `{install_cmd.strip()}`, which does not install the "
            f"extra(s) {missing} that the test suite import-skips on. Those tests are "
            f"silently skipped on every pull request: tests/test_metric_frame.py (44) and "
            f"tests/test_evaluation_report.py (22) are skipped whole. MetricFrame "
            f"is designated a cross-repo contract by ADR-022 §1. "
            f"Fix: `poetry install --all-extras`."
        )
