"""Guard: CI must install every extra the test suite import-skips on.

Written as a failing test by a falsification audit on 2026-08-02 (round 2 of
"we are ready to shut this session down" — FALSIFIED), then kept as a permanent guard
once CI was fixed to `poetry install --all-extras`.

The defect it caught: **the `MetricFrame` contract was not tested in CI.**

`views-frames` is an **optional** extra (`pyproject.toml`: `views-frames = {version =
"^1.4", optional = true}`, exposed as the `frames` extra). CI runs a bare
`poetry install`, which does not install optional dependencies. `tests/test_metric_frame.py`
guards with a **module-level** `pytest.importorskip("views_frames")` at line 19, so the
entire file is skipped rather than failing.

Measured, not inferred:

    local (frames installed) : 361 passed
    CI   (frames absent)     : 293 passed, 4 skipped
    delta                    : 68 tests that never run in CI
    of which test_metric_frame.py alone accounts for 44

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
        extras_block = re.search(
            r"\[tool\.poetry\.extras\]\n((?:.*\n)*?)\n", pyproject
        )
        assert extras_block, "[tool.poetry.extras] not found in pyproject.toml"
        # extra name -> the distributions it pulls, e.g. frames -> views-frames
        extras = {
            m.group(1): m.group(2)
            for m in re.finditer(r'^(\w+)\s*=\s*\[([^\]]*)\]', extras_block.group(1), re.M)
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
        required = {
            name for name, pkgs in extras.items()
            if any(p.strip().strip('"\'').replace("-", "_") in skipped_on
                   for p in pkgs.split(","))
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
            f"silently skipped on every pull request — locally 361 tests pass, in CI "
            f"293 do. tests/test_metric_frame.py alone contributes 44, and MetricFrame "
            f"is designated a cross-repo contract by ADR-022 §1. "
            f"Fix: `poetry install --all-extras`."
        )
