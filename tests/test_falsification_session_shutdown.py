"""Guards against release notes outliving the facts they assert.

Written as failing tests by a falsification audit on 2026-08-02 ("we are ready to shut
this session down" — FALSIFIED), then kept as permanent guards once the findings were
fixed. Both encode a mechanism distinct from ordinary documentation drift: **fixing a
documented limitation falsifies its documentation at the moment of the fix**, in the
part of a changeset that is least likely to be re-read.

F1 — `CHANGELOG.md` "Known limitations" stated that `views-pipeline-core` pins
    `views-evaluation = "^0.4.0"` and "will reject this release on install until that
    pin is widened". The pin had been widened to `^0.5.0` (pipeline-core PR #379,
    merged 2026-08-02T04:13Z) *hours before* that text published — so the claim was
    already false in the repo, in the GitHub release notes, and on PyPI.

F2 — the migration paragraph said "For the two key-name rows" while the table it
    describes had three. The third appeared when the single "A legacy key" row was
    split into `metrics`/`*_uncertainty_metrics` and `targets` during the third review
    iteration; the sentence counting those rows was three rows away and was missed.

A third finding — the persistent project memory recording version 0.4.0, branch
`feature/samples_for_fao`, and 241 tests against a reality of 0.5.0, `development`, and
359 — was fixed outside this repo. It has no test here on purpose: that file is neither
a repo artifact nor present in CI, so a check for it would do nothing but skip.
"""
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
PIPELINE_CORE = REPO_ROOT.parent / "views-pipeline-core" / "pyproject.toml"


def _config_table_rows():
    """The `Input | Was, in 0.4.0 | Now` table's first column, one entry per row."""
    text = CHANGELOG.read_text()
    table = re.search(
        r"\| Input \| Was, in 0\.4\.0 \| Now \|\n\|[-| ]+\|\n((?:\|.*\n)+)", text
    )
    assert table, "the 0.4.0-comparison table has moved or been reshaped"
    return [r.split("|")[1].strip() for r in table.group(1).strip().splitlines()]


class TestReleaseNotesDoNotOutliveTheirFacts:

    def test_changelog_does_not_claim_a_pin_the_consumer_no_longer_has(self):
        """F1. The release notes assert a downstream pin that has since changed.

        This is the C-34 defect class in the artifact created to satisfy ADR-022:
        a published document making a checkable claim about the world that is false.
        The claim is load-bearing — it tells consumers this release is uninstallable
        for them, which is now the opposite of true.
        """
        if not PIPELINE_CORE.exists():
            pytest.skip("views-pipeline-core not checked out alongside this repo")
        actual_pin = re.search(
            r'^views-evaluation\s*=\s*"([^"]+)"', PIPELINE_CORE.read_text(), re.M
        )
        assert actual_pin, "could not find the views-evaluation pin in pipeline-core"
        claimed_stale = 'pins `views-evaluation = "^0.4.0"`' in CHANGELOG.read_text()
        assert not (claimed_stale and actual_pin.group(1) != "^0.4.0"), (
            f"CHANGELOG.md says views-pipeline-core pins ^0.4.0 and 'will reject this "
            f"release on install', but the pin is now {actual_pin.group(1)!r}. "
            f"The same text ships in the published v0.5.0 release notes."
        )

    def test_migration_paragraph_row_counts_match_the_table(self):
        """F2. The prose counts rows of a table that has since gained one.

        Counted, not asserted: the check derives both numbers from the file, so it
        cannot go stale the way the sentence it guards did.
        """
        text = CHANGELOG.read_text()
        rows = _config_table_rows()
        key_rows = [r for r in rows if "steps" not in r
                    and ("Legacy" in r or "key" in r.lower())]
        steps_rows = [r for r in rows if "steps" in r]
        words = {"two": 2, "three": 3, "four": 4, "five": 5}

        claimed_key = re.search(r"For the (\w+) key-name rows?", text)
        claimed_steps = re.search(r"For the (\w+) `steps` rows?", text)
        assert claimed_key and claimed_steps, "migration paragraph has been reworded"

        mismatches = []
        if words.get(claimed_key.group(1)) != len(key_rows):
            mismatches.append(
                f"prose says {claimed_key.group(1)} key-name rows, table has "
                f"{len(key_rows)}: {key_rows}"
            )
        if words.get(claimed_steps.group(1)) != len(steps_rows):
            mismatches.append(
                f"prose says {claimed_steps.group(1)} `steps` rows, table has "
                f"{len(steps_rows)}"
            )
        assert not mismatches, "CHANGELOG migration prose miscounts its own table:\n  " \
            + "\n  ".join(mismatches)
