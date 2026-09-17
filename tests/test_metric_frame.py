"""
Direct unit tests for MetricFrame and EvaluationReport.to_metric_frame().

Structured per ADR-020 (Red/Beige/Green):
  GREEN — emit for all 4 (task, pred_type) cells; envelope conformance; save/load round-trip;
          vocabulary mapping; provenance split; mean aggregate rows.
  BEIGE — empty schema, NaN metric values, run_id=None at emit, scoring_code_version default.
  RED   — fail-loud construction (non-float32, wrong ndim, length mismatch, missing axis).

The whole module requires the optional 'views-frames' dependency; it is skipped otherwise.
"""
import json
import logging
import tempfile

import numpy as np
import pytest
from pathlib import Path

pytest.importorskip("views_frames")

from views_frames import FrameMetadata  # noqa: E402
from views_frames.conformance import assert_frame_envelope  # noqa: E402

from views_evaluation.evaluation.evaluation_report import EvaluationReport  # noqa: E402
from views_evaluation.evaluation.metric_catalog import METRIC_MEMBERSHIP  # noqa: E402
from views_evaluation.evaluation.metric_frame import (  # noqa: E402
    AXES,
    MEAN_GROUP_ID,
    SCHEMA_TO_EVAL_TYPE,
    SCHEMA_VERSION,
    MetricFrame,
    MetricFrameMetadata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _regression_point_results():
    return {
        'month':       {'month100': {'MSE': 1.0}, 'month101': {'MSE': 3.0}},
        'time_series': {'ts00': {'MSE': 1.5}},
        'step':        {'step01': {'MSE': 1.0}, 'step02': {'MSE': 2.0}},
    }

def _regression_sample_results():
    return {
        'month':       {'month100': {'CRPS': 0.5}, 'month101': {'CRPS': 0.7}},
        'time_series': {'ts00': {'CRPS': 0.4}},
        'step':        {'step01': {'CRPS': 0.6}},
    }

def _classification_point_results():
    return {
        'month':       {'month100': {'AP': 0.8}},
        'time_series': {'ts00': {'AP': 0.75}},
        'step':        {'step01': {'AP': 0.85}},
    }

def _classification_sample_results():
    return {
        'month':       {'month100': {'CRPS': 0.3}},
        'time_series': {'ts00': {'CRPS': 0.25}},
        'step':        {'step01': {'CRPS': 0.35}},
    }

# Canonical metric set views-reporting renders (config/_reporting.py:_CANONICAL_REPORT_METRICS).
_CANONICAL_REPORT_METRICS = {
    ("regression", "point"): {"MSLE", "MSE", "MCR_point", "y_hat_bar"},
    ("regression", "sample"): {"CRPS", "MIS", "Ignorance", "MCR_sample", "y_hat_bar"},
    ("classification", "point"): {"AP", "Brier_cls_point"},
    ("classification", "sample"): {"Brier_cls_sample", "CRPS"},
}


def _rows_as_dicts(mf):
    """Materialise a MetricFrame as a list of per-row dicts for easy assertions."""
    n = mf.n_rows
    return [
        {**{axis: str(mf.identifiers[axis][i]) for axis in AXES}, "value": float(mf.values[i, 0])}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# GREEN: emit for all 4 cells
# ---------------------------------------------------------------------------

class TestToMetricFrameGreen:

    @pytest.mark.parametrize("task,pred_type,results", [
        ("regression", "point", _regression_point_results()),
        ("regression", "sample", _regression_sample_results()),
        ("classification", "point", _classification_point_results()),
        ("classification", "sample", _classification_sample_results()),
    ])
    def test_emit_all_cells_produces_metric_frame(self, task, pred_type, results):
        report = EvaluationReport('lr_ged_sb', task, pred_type, results)
        mf = report.to_metric_frame(model_id='m', partition='calibration', level='pgm')
        assert isinstance(mf, MetricFrame)
        assert mf.values.dtype == np.float32
        assert mf.values.ndim == 2 and mf.values.shape[1] == 1
        assert mf.values.shape[0] == mf.n_rows
        assert set(mf.identifiers.keys()) == set(AXES)

    def test_envelope_conformance(self):
        """The views-frames published checker (incl. save/load round-trip) passes."""
        report = EvaluationReport('t', 'regression', 'point', _regression_point_results())
        mf = report.to_metric_frame(model_id='m', partition='calibration', level='pgm')
        assert_frame_envelope(mf)  # raises AssertionError on any violation

    def test_eval_type_vocabulary_mapped(self):
        report = EvaluationReport('t', 'regression', 'point', _regression_point_results())
        mf = report.to_metric_frame()
        eval_types = set(mf.identifiers['eval_type'].tolist())
        assert eval_types == {"month-wise", "time-series-wise", "step-wise"}
        assert SCHEMA_TO_EVAL_TYPE == {
            "month": "month-wise", "time_series": "time-series-wise", "step": "step-wise",
        }

    def test_mean_aggregate_row_is_mean_over_groups(self):
        # month has MSE 1.0 and 3.0 across two groups -> mean row = 2.0
        report = EvaluationReport('t', 'regression', 'point', _regression_point_results())
        mf = report.to_metric_frame()
        rows = _rows_as_dicts(mf)
        mean_rows = [r for r in rows
                     if r['group_id'] == MEAN_GROUP_ID
                     and r['eval_type'] == 'month-wise' and r['metric'] == 'MSE']
        assert len(mean_rows) == 1
        assert mean_rows[0]['value'] == pytest.approx(2.0)

    def test_target_and_axes_populated_per_row(self):
        report = EvaluationReport('lr_ged_sb', 'regression', 'point', _regression_point_results())
        mf = report.to_metric_frame(partition='calibration', level='pgm')
        rows = _rows_as_dicts(mf)
        assert all(r['target'] == 'lr_ged_sb' for r in rows)
        assert all(r['partition'] == 'calibration' for r in rows)
        assert all(r['level'] == 'pgm' for r in rows)

    def test_provenance_split_generic_vs_eval_specific(self):
        report = EvaluationReport('t', 'regression', 'point', _regression_point_results())
        mf = report.to_metric_frame(
            model_id='purple_alien', run_id='abc123', data_version='v7',
            run_type='calibration', timestamp=42, seed=7,
            scoring_code_version='9.9.9', evaluation_timestamp='2026-06-25T10:00:00',
        )
        # Generic provenance lives in the reused FrameMetadata.
        assert isinstance(mf.metadata.provenance, FrameMetadata)
        assert mf.metadata.provenance.model == 'purple_alien'
        assert mf.metadata.provenance.run_id == 'abc123'
        assert mf.metadata.provenance.data_version == 'v7'
        # Eval-specific provenance stays out of the generic header (register C-47).
        generic = mf.metadata.provenance.to_dict()
        assert 'scoring_code_version' not in generic
        assert 'evaluation_timestamp' not in generic
        assert mf.metadata.scoring_code_version == '9.9.9'
        assert mf.metadata.evaluation_timestamp == '2026-06-25T10:00:00'
        assert mf.metadata.schema_version == SCHEMA_VERSION

    def test_numpy_typed_timestamp_seed_round_trip(self):
        """Injected numpy scalar timestamp/seed must not break json serialization (save)."""
        report = EvaluationReport('t', 'regression', 'point', _regression_point_results())
        mf = report.to_metric_frame(timestamp=np.int64(1700000000), seed=np.int32(42))
        with tempfile.TemporaryDirectory() as d:
            mf.save(d)  # would TypeError without numpy-aware json default
            loaded = MetricFrame.load(d)
        assert loaded.metadata.provenance.timestamp == 1700000000
        assert loaded.metadata.provenance.seed == 42

    def test_save_load_round_trip_including_metadata(self):
        report = EvaluationReport('t', 'regression', 'sample', _regression_sample_results())
        mf = report.to_metric_frame(model_id='m', data_version='v1',
                                     evaluation_timestamp='2026-06-25T00:00:00')
        with tempfile.TemporaryDirectory() as d:
            mf.save(d)
            loaded = MetricFrame.load(d)
        assert np.array_equal(loaded.values, mf.values, equal_nan=True)
        for axis in AXES:
            assert np.array_equal(loaded.identifiers[axis], mf.identifiers[axis])
        assert loaded.metadata.to_dict() == mf.metadata.to_dict()

    def test_backward_compatible_to_dict_unaffected(self):
        report = EvaluationReport('t', 'regression', 'point', _regression_point_results())
        before = report.to_dict()
        report.to_metric_frame()
        assert report.to_dict() == before


# ---------------------------------------------------------------------------
# GREEN: drift guard — kills views-reporting C-41 at this repo's CI
# ---------------------------------------------------------------------------

class TestCanonicalTokenDriftGuard:

    @pytest.mark.parametrize("cell", list(_CANONICAL_REPORT_METRICS.keys()))
    def test_catalog_membership_superset_of_reporting_canonical(self, cell):
        """Every metric views-reporting renders must remain a valid token in this cell."""
        canonical = _CANONICAL_REPORT_METRICS[cell]
        membership = METRIC_MEMBERSHIP[cell]
        missing = canonical - membership
        assert not missing, (
            f"Canonical reporting metrics {sorted(missing)} are no longer valid for {cell} "
            f"in METRIC_MEMBERSHIP — this would render as 'not calculated' downstream (C-41)."
        )

    def test_emitted_tokens_match_canonical_when_present(self):
        """A frame built from the canonical regression-sample metrics emits those exact tokens."""
        canonical = _CANONICAL_REPORT_METRICS[("regression", "sample")]
        results = {
            'month': {'month100': {m: 0.5 for m in canonical}},
            'time_series': {},
            'step': {},
        }
        report = EvaluationReport('t', 'regression', 'sample', results)
        mf = report.to_metric_frame()
        emitted = set(mf.identifiers['metric'].tolist())
        assert canonical <= emitted


# ---------------------------------------------------------------------------
# BEIGE: realistic edge cases
# ---------------------------------------------------------------------------

class TestToMetricFrameBeige:

    def test_empty_schema_emits_no_rows_for_that_schema(self):
        results = {'month': {'month100': {'MSE': 1.0}}, 'time_series': {}, 'step': {}}
        report = EvaluationReport('t', 'regression', 'point', results)
        mf = report.to_metric_frame()
        eval_types = set(mf.identifiers['eval_type'].tolist())
        assert eval_types == {"month-wise"}  # empty schemas contribute nothing
        # one per-group row + one mean row
        assert mf.n_rows == 2

    def test_nan_metric_value_round_trips(self):
        results = {'month': {'month100': {'Pearson': float('nan')}},
                   'time_series': {}, 'step': {}}
        report = EvaluationReport('t', 'regression', 'point', results)
        mf = report.to_metric_frame()
        assert np.isnan(mf.values).any()
        assert_frame_envelope(mf)  # round-trip is NaN-tolerant

    def test_all_nan_group_mean_is_nan_without_error(self):
        results = {'month': {'month100': {'MSE': float('nan')},
                             'month101': {'MSE': float('nan')}},
                   'time_series': {}, 'step': {}}
        report = EvaluationReport('t', 'regression', 'point', results)
        mf = report.to_metric_frame()
        rows = _rows_as_dicts(mf)
        mean_row = [r for r in rows if r['group_id'] == MEAN_GROUP_ID][0]
        assert np.isnan(mean_row['value'])

    def test_run_id_none_at_emit_is_omitted_from_metadata(self):
        report = EvaluationReport('t', 'regression', 'point', _regression_point_results())
        mf = report.to_metric_frame(model_id='m', run_id=None)
        assert mf.metadata.provenance.run_id is None
        assert 'run_id' not in mf.metadata.provenance.to_dict()

    def test_scoring_code_version_defaults_to_installed_version(self):
        report = EvaluationReport('t', 'regression', 'point', _regression_point_results())
        mf = report.to_metric_frame()  # no scoring_code_version passed
        # Installed package version is a non-empty string in the test env.
        assert isinstance(mf.metadata.scoring_code_version, str)
        assert mf.metadata.scoring_code_version

    def test_scoring_code_version_identifies_the_code_that_ran(self):
        """C-25: a bare version cannot distinguish the tree from the last install.

        `importlib.metadata` reports the *installed distribution*, not the executing
        code. Under an editable install they drift as soon as the source moves ahead of
        the last `pip install` — measured on 2026-08-02 with the tree at 1.0.0 and the
        dist-info at 0.5.0, emitting frames stamped `0.5.0` from 1.0.0 code.

        The previous version of this test asserted only that the stamp was a non-empty
        string, so it was satisfied by exactly the wrong answer. It then decided
        skip-vs-assert from where *this test file* lives; it now decides from where the
        *imported module* lives, because an installed copy imported while this checkout
        is on disk must stamp a bare version (C-39), not this checkout's SHA.
        """
        import subprocess
        from pathlib import Path
        from views_evaluation.evaluation import metric_frame as mf_mod

        module_root = Path(mf_mod.__file__).resolve().parents[2]
        repo_root = Path(__file__).resolve().parents[1]
        stamp = mf_mod.default_scoring_code_version()
        if module_root != repo_root:
            assert hasattr(mf_mod, "_read_head_sha"), (
                f"the imported module at {module_root} predates the C-39 fix — a stale "
                f"installed copy is being tested instead of this checkout"
            )
            assert stamp and "+g" not in stamp, (
                f"the imported module is an installed copy at {module_root}, which must "
                f"stamp a bare version (C-39), got {stamp!r}"
            )
            return

        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=module_root,
        ).stdout.strip()
        if not head:
            pytest.skip("not running from a git checkout")
        assert stamp and stamp.endswith(f"+g{head[:7]}"), (
            f"stamp {stamp!r} does not identify the running code (HEAD {head[:7]}). A bare "
            f"version here is the C-25 defect: it is indistinguishable from a stale one."
        )

    def test_scoring_code_version_is_bare_when_there_is_no_worktree(self, monkeypatch):
        """When no SHA can be attributed to this checkout, the stamp is the bare version.

        This stubs the SHA reader; the layouts that make it return None (a wheel, wherever
        it lives; a vendored copy) are exercised in `TestSourceGitShaBoundary*`.

        That is a property of how the package was installed, not a failure — ADR-015's
        fault-versus-data-property test — so it must not raise and must not invent a SHA.
        """
        from views_evaluation.evaluation import metric_frame as mf_mod

        monkeypatch.setattr(mf_mod, "_source_git_sha", lambda: None)
        stamp = mf_mod.default_scoring_code_version()
        assert stamp and "+g" not in stamp, f"expected a bare version, got {stamp!r}"

    def test_partition_level_default_to_empty_string(self):
        report = EvaluationReport('t', 'regression', 'point', _regression_point_results())
        mf = report.to_metric_frame()  # no partition/level
        assert set(mf.identifiers['partition'].tolist()) == {""}
        assert set(mf.identifiers['level'].tolist()) == {""}

    def test_fully_empty_report_raises_rather_than_emitting_zero_rows(self):
        """Superseded behaviour: this used to assert a valid zero-row frame was emitted.

        ADR-015 ruling 6 forbids it — such a frame passes every envelope check and
        persists as a legitimate-looking audit artifact recording nothing (C-30).
        The *container* may still hold zero rows, so that load() can read back
        whatever save() wrote; that half is asserted in
        ``TestVacuousEmitRed.test_metricframe_itself_still_accepts_zero_rows``
        and in ``test_zero_row_container_still_conforms`` below.
        """
        report = EvaluationReport('t', 'regression', 'point',
                                  {'month': {}, 'time_series': {}, 'step': {}})
        with pytest.raises(ValueError, match="produced no rows"):
            report.to_metric_frame()

    def test_zero_row_container_still_conforms(self):
        """A directly-constructed empty frame still satisfies the views-frames envelope."""
        ids = {axis: np.asarray([], dtype=str) for axis in AXES}
        mf = MetricFrame(np.zeros((0, 1), dtype=np.float32), ids)
        assert mf.n_rows == 0
        assert mf.values.shape == (0, 1)
        assert mf.values.dtype == np.float32
        assert_frame_envelope(mf)  # empty container still conforms + round-trips


# ---------------------------------------------------------------------------
# RED: fail-loud construction
# ---------------------------------------------------------------------------

class TestMetricFrameConstructionRed:

    def _valid_ids(self, n):
        return {axis: np.asarray(["x"] * n, dtype=str) for axis in AXES}

    def test_non_float32_values_raise(self):
        with pytest.raises(ValueError, match="float32"):
            MetricFrame(np.zeros((2, 1), dtype=np.float64), self._valid_ids(2))

    def test_one_dimensional_values_raise(self):
        with pytest.raises(ValueError, match="2D"):
            MetricFrame(np.zeros(2, dtype=np.float32), self._valid_ids(2))

    def test_missing_axis_raises(self):
        ids = self._valid_ids(2)
        del ids["partition"]
        with pytest.raises(ValueError, match="missing required axes"):
            MetricFrame(np.zeros((2, 1), dtype=np.float32), ids)

    def test_identifier_length_mismatch_raises(self):
        ids = self._valid_ids(2)
        ids["metric"] = np.asarray(["only_one"], dtype=str)
        with pytest.raises(ValueError, match="mismatch"):
            MetricFrame(np.zeros((2, 1), dtype=np.float32), ids)

    def test_non_array_values_raise(self):
        with pytest.raises(ValueError, match="numpy array"):
            MetricFrame([[0.0], [1.0]], self._valid_ids(2))

    def test_two_dimensional_identifier_raises(self):
        ids = self._valid_ids(2)
        ids["group_id"] = np.asarray([["a"], ["b"]], dtype=str)  # (2, 1), len==2 but 2D
        with pytest.raises(ValueError, match="1D"):
            MetricFrame(np.zeros((2, 1), dtype=np.float32), ids)


# ---------------------------------------------------------------------------
# RED: Level-1 log-and-raise (ADR-013; logging standard §4, §5.1, §8)
#
# MetricFrame is Level 1 — it persists the evaluation-of-record — so its raises
# must ALSO log at ERROR. Level-0 modules stay exempt and are asserted separately
# in test_level_zero_modules_have_no_logger.
# ---------------------------------------------------------------------------

class TestMetricFrameLogAndRaiseRed:

    def _valid_ids(self, n):
        return {axis: np.asarray(["x"] * n, dtype=str) for axis in AXES}

    def test_non_float32_logs_error_before_raising(self, caplog):
        with caplog.at_level(logging.ERROR, logger="views_evaluation.evaluation.metric_frame"):
            with pytest.raises(ValueError, match="float32"):
                MetricFrame(np.zeros((2, 1), dtype=np.float64), self._valid_ids(2))
        errors = [r for r in caplog.records if r.levelname == "ERROR"]
        assert errors, "structural failure raised without logging (ADR-013 requires both)"
        assert "float32" in errors[0].getMessage()

    def test_missing_axis_logs_error_before_raising(self, caplog):
        ids = self._valid_ids(2)
        del ids["partition"]
        with caplog.at_level(logging.ERROR, logger="views_evaluation.evaluation.metric_frame"):
            with pytest.raises(ValueError, match="missing required axes"):
                MetricFrame(np.zeros((2, 1), dtype=np.float32), ids)
        errors = [r for r in caplog.records if r.levelname == "ERROR"]
        assert errors
        assert "missing required axes" in errors[0].getMessage()

    def test_logged_message_is_identical_to_exception_message(self, caplog):
        """Standard §4: log and raise carry the same message — not a paraphrase."""
        ids = self._valid_ids(2)
        ids["metric"] = np.asarray(["only_one"], dtype=str)
        with caplog.at_level(logging.ERROR, logger="views_evaluation.evaluation.metric_frame"):
            with pytest.raises(ValueError) as excinfo:
                MetricFrame(np.zeros((2, 1), dtype=np.float32), ids)
        errors = [r for r in caplog.records if r.levelname == "ERROR"]
        assert errors
        assert errors[0].getMessage() == str(excinfo.value)

    def test_level_zero_modules_have_no_logger(self):
        """Logging standard §5.1: Level-0 pure-math modules must NOT acquire loggers.

        Guards the exemption in the other direction — a well-meaning contributor
        adding a logger to the numeric core is a violation, not an improvement.
        """
        import inspect
        from views_evaluation.evaluation import (
            evaluation_frame,
            metric_catalog,
            native_evaluator,
            native_metric_calculators,
        )
        for module in (evaluation_frame, native_evaluator, metric_catalog,
                       native_metric_calculators):
            src = inspect.getsource(module)
            assert "getLogger" not in src, (
                f"{module.__name__} is Level 0 and must not maintain a logger "
                f"(logging standard §5.1)"
            )

    def test_library_never_configures_root_logger(self):
        """A library must not call basicConfig or attach handlers."""
        import inspect
        import views_evaluation.evaluation.metric_frame as mf
        src = inspect.getsource(mf)
        assert "basicConfig" not in src
        assert "addHandler" not in src


# ---------------------------------------------------------------------------
# MetricFrameMetadata direct tests
# ---------------------------------------------------------------------------

class TestMetricFrameMetadata:

    def test_to_dict_from_dict_round_trip(self):
        meta = MetricFrameMetadata(
            provenance=FrameMetadata(model='m', run_id='r', data_version='v'),
            scoring_code_version='1.2.3',
            evaluation_timestamp='2026-06-25T00:00:00',
        )
        d = meta.to_dict()
        assert json.loads(json.dumps(d)) == d  # JSON-serialisable
        back = MetricFrameMetadata.from_dict(d)
        assert back == meta

    def test_schema_version_always_present(self):
        meta = MetricFrameMetadata()
        assert meta.to_dict()["schema_version"] == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# RED: vacuous emit (ADR-015 ruling 6; risk register C-30)
#
# A zero-row MetricFrame passes _validate AND assert_frame_envelope — dtype,
# dimensionality and axis alignment are all satisfied. The guard therefore has to
# live at the emit site, which is also the only place with enough context to say
# why the report was empty.
# ---------------------------------------------------------------------------

class TestVacuousEmitRed:

    def test_report_with_no_metric_values_raises(self):
        report = EvaluationReport('t', 'regression', 'sample',
                                  {'month': {'month100': {}}, 'time_series': {}, 'step': {}})
        with pytest.raises(ValueError, match="produced no rows"):
            report.to_metric_frame(model_id='m')

    def test_completely_empty_report_raises(self):
        report = EvaluationReport('t', 'regression', 'sample',
                                  {'month': {}, 'time_series': {}, 'step': {}})
        with pytest.raises(ValueError, match="produced no rows"):
            report.to_metric_frame(model_id='m')

    def test_error_message_names_target_and_schemas(self):
        report = EvaluationReport('ged_sb_best', 'regression', 'sample',
                                  {'month': {'month100': {}}, 'time_series': {}, 'step': {}})
        with pytest.raises(ValueError) as excinfo:
            report.to_metric_frame(model_id='m')
        msg = str(excinfo.value)
        assert 'ged_sb_best' in msg
        assert 'month' in msg

    def test_vacuous_emit_logs_before_raising(self, caplog):
        """Level-1 emit path logs at ERROR even though the raise sits in a Level-0 file."""
        report = EvaluationReport('t', 'regression', 'sample',
                                  {'month': {'month100': {}}, 'time_series': {}, 'step': {}})
        with caplog.at_level(logging.ERROR, logger="views_evaluation.evaluation.metric_frame"):
            with pytest.raises(ValueError):
                report.to_metric_frame(model_id='m')
        assert [r for r in caplog.records if r.levelname == "ERROR"]

    def test_partial_report_still_emits(self):
        """A single metric value in a single schema is enough — partial is not vacuous."""
        report = EvaluationReport('t', 'regression', 'sample',
                                  {'month': {'month100': {'CRPS': 0.5}},
                                   'time_series': {}, 'step': {}})
        mf = report.to_metric_frame(model_id='m')
        assert mf.n_rows == 2      # the group row + the "mean" aggregate row

    def test_metricframe_itself_still_accepts_zero_rows(self):
        """The container must stay able to represent whatever load() reads back.

        The guard belongs at emit, not in _validate — tightening the container
        would break the save/load round-trip and change a cross-repo envelope.
        """
        ids = {axis: np.asarray([], dtype=str) for axis in AXES}
        mf = MetricFrame(np.zeros((0, 1), dtype=np.float32), ids)
        assert mf.n_rows == 0


# ---------------------------------------------------------------------------
# C-39: the provenance SHA must come from THIS repository's checkout or not at all
# ---------------------------------------------------------------------------

_OWN_PYPROJECT = '[tool.poetry]\nname = "views_evaluation"\n'
# A consumer's pyproject that *mentions* this package in unrelated tables — a dependency
# line and an import-linter contract named after it — but does not declare it.
_CONSUMER_PYPROJECT = (
    '[project]\nname = "views-reporting"\n'
    'dependencies = ["views_evaluation>=1.0.0"]\n'
    '[[tool.importlinter.contracts]]\nname = "views_evaluation"\n'
)


def _fake_package_file(root):
    """The path `metric_frame.py` would have if this package lived under `root`.

    Anchored to the module's real path relative to the package's parent, so moving the
    module (or adopting a `src/` layout) changes the fixture and turns the Green test red
    instead of leaving the `parents[2]` assumption in `_source_git_sha` silently wrong.
    """
    import views_evaluation
    from views_evaluation.evaluation import metric_frame as mf_mod

    rel = Path(mf_mod.__file__).resolve().relative_to(
        Path(views_evaluation.__file__).resolve().parents[1]
    )
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _git_dir(root, head, refs=None, packed=None):
    """Build a `.git` directory: HEAD content, loose refs {ref: sha}, optional packed-refs."""
    (root / ".git").mkdir(parents=True, exist_ok=True)
    (root / ".git" / "HEAD").write_text(head + "\n")
    for ref, sha in (refs or {}).items():
        p = root / ".git" / ref
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(sha + "\n")
    if packed is not None:
        (root / ".git" / "packed-refs").write_text(packed)
    return root / ".git"


class TestSourceGitShaBoundaryRed:
    """Register C-39. The bound has two halves — no parent walk, and a name gate — and a
    Red test for each, so a mutant that restores the walk or loosens the gate fails."""

    def test_wheel_inside_consumer_checkout_stamps_no_sha(self, tmp_path, monkeypatch):
        """uv's default `.venv` sits below the consumer's `.git` AND its pyproject."""
        from views_evaluation.evaluation import metric_frame as mf_mod

        consumer = tmp_path / "consumer"
        _git_dir(consumer, "e15f298aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        (consumer / "pyproject.toml").write_text(_CONSUMER_PYPROJECT)
        site = consumer / ".venv" / "lib" / "python3.11" / "site-packages"
        monkeypatch.setattr(mf_mod, "__file__", str(_fake_package_file(site)))

        assert mf_mod._source_git_sha() is None, (
            "the walker found the CONSUMER's .git and would stamp e15f298 as this "
            "library's version — register C-39"
        )

    def test_vendored_copy_in_consumer_checkout_stamps_no_sha(self, tmp_path, monkeypatch):
        """The package directory directly under a consumer's root: `.git` and
        `pyproject.toml` are both at parents[2], and only the name gate says no."""
        from views_evaluation.evaluation import metric_frame as mf_mod

        consumer = tmp_path / "consumer"
        _git_dir(consumer, "e15f298aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        (consumer / "pyproject.toml").write_text(_CONSUMER_PYPROJECT)
        monkeypatch.setattr(mf_mod, "__file__", str(_fake_package_file(consumer)))

        assert mf_mod._source_git_sha() is None

    def test_unreadable_git_state_logs_and_stamps_bare(self, tmp_path, monkeypatch, caplog):
        """A `.git` that is ours but yields no commit is a fault, not a data property: the
        stamp is bare AND a WARNING is left (ADR-013, logging standard §5.1)."""
        from views_evaluation.evaluation import metric_frame as mf_mod

        repo = tmp_path / "repo"
        _git_dir(repo, "ref: refs/heads/alias", refs={"refs/heads/alias": "ref: refs/heads/main"})
        (repo / "pyproject.toml").write_text(_OWN_PYPROJECT)
        monkeypatch.setattr(mf_mod, "__file__", str(_fake_package_file(repo)))

        with caplog.at_level(logging.WARNING, logger="views_evaluation.evaluation.metric_frame"):
            assert mf_mod._source_git_sha() is None
        assert any("no commit could be read" in r.message for r in caplog.records), (
            "a symbolic-ref chain must not be truncated into a garbage stamp, and must be logged"
        )

    def test_missing_head_logs_and_stamps_bare(self, tmp_path, monkeypatch, caplog):
        from views_evaluation.evaluation import metric_frame as mf_mod

        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)  # no HEAD at all
        (repo / "pyproject.toml").write_text(_OWN_PYPROJECT)
        monkeypatch.setattr(mf_mod, "__file__", str(_fake_package_file(repo)))

        with caplog.at_level(logging.WARNING, logger="views_evaluation.evaluation.metric_frame"):
            assert mf_mod._source_git_sha() is None
        assert any(
            r.levelno == logging.WARNING and "no commit could be read" in r.message
            for r in caplog.records
        ), "a missing HEAD in our own .git is a fault and must be logged, not classified as a wheel"


    def test_wheel_inside_own_checkout_venv_stamps_no_sha(self, tmp_path, monkeypatch):
        """A wheel copy under THIS repository's own `.venv` is still a wheel. A parent
        walk that keeps the name gate would climb to the checkout, find a pyproject
        that declares us, and stamp the checkout's SHA onto an installed copy."""
        from views_evaluation.evaluation import metric_frame as mf_mod

        repo = tmp_path / "repo"
        _git_dir(repo, "ref: refs/heads/main", refs={"refs/heads/main": "cafebabe000000000000000000000000000000000"})
        (repo / "pyproject.toml").write_text(_OWN_PYPROJECT)
        site = repo / ".venv" / "lib" / "python3.11" / "site-packages"
        monkeypatch.setattr(mf_mod, "__file__", str(_fake_package_file(site)))

        assert mf_mod._source_git_sha() is None

    def test_git_without_pyproject_is_silent(self, tmp_path, monkeypatch, caplog):
        """`.git` present but no `pyproject.toml` beside it is not our checkout: None,
        and — being a contracted data property, not a fault — no log record."""
        from views_evaluation.evaluation import metric_frame as mf_mod

        repo = tmp_path / "repo"
        _git_dir(repo, "ref: refs/heads/main", refs={"refs/heads/main": "cafebabe000000000000000000000000000000000"})
        monkeypatch.setattr(mf_mod, "__file__", str(_fake_package_file(repo)))

        with caplog.at_level(logging.DEBUG, logger="views_evaluation.evaluation.metric_frame"):
            assert mf_mod._source_git_sha() is None
        assert not caplog.records


class TestSourceGitShaBoundaryGreen:

    def test_loose_ref_wins_over_a_stale_packed_ref(self, tmp_path, monkeypatch):
        """git semantics: after `git pack-refs` and a later commit, the loose file is
        current and the packed line is stale. Loose must be consulted first."""
        from views_evaluation.evaluation import metric_frame as mf_mod

        repo = tmp_path / "repo"
        _git_dir(repo, "ref: refs/heads/main",
                 refs={"refs/heads/main": "bbbbbbb0000000000000000000000000000000000"},
                 packed="# pack-refs with: peeled fully-peeled sorted\n"
                        "aaaaaaa0000000000000000000000000000000000 refs/heads/main\n")
        (repo / "pyproject.toml").write_text(_OWN_PYPROJECT)
        monkeypatch.setattr(mf_mod, "__file__", str(_fake_package_file(repo)))

        assert mf_mod._source_git_sha() == "bbbbbbb"

    def test_packed_ref_is_matched_by_name_not_position(self, tmp_path, monkeypatch):
        from views_evaluation.evaluation import metric_frame as mf_mod

        repo = tmp_path / "repo"
        _git_dir(repo, "ref: refs/heads/zeta",
                 packed="# pack-refs with: peeled fully-peeled sorted\n"
                        "1111111000000000000000000000000000000000 refs/heads/main\n"
                        "2222222000000000000000000000000000000000 refs/heads/zeta\n")
        (repo / "pyproject.toml").write_text(_OWN_PYPROJECT)
        monkeypatch.setattr(mf_mod, "__file__", str(_fake_package_file(repo)))

        assert mf_mod._source_git_sha() == "2222222"

    def test_mixed_case_name_is_this_distribution(self, tmp_path, monkeypatch):
        """PEP 503 normalisation is case-insensitive, not only separator-insensitive."""
        from views_evaluation.evaluation import metric_frame as mf_mod

        repo = tmp_path / "repo"
        _git_dir(repo, "0123456abcdef0123456abcdef0123456abcdef01")
        (repo / "pyproject.toml").write_text('[project]\nname = "Views-Evaluation"\n')
        monkeypatch.setattr(mf_mod, "__file__", str(_fake_package_file(repo)))

        assert mf_mod._source_git_sha() == "0123456"

    def test_every_read_is_utf8(self):
        """TOML is UTF-8 by spec and git writes refs as bytes; a locale-dependent
        `read_text()` would lose the SHA under a C or East-Asian code page. Asserted on
        the source, since the process locale cannot be switched safely mid-suite."""
        import inspect
        import re as _re
        from views_evaluation.evaluation import metric_frame as mf_mod

        for fn in (mf_mod._normalised_project_name, mf_mod._read_head_sha):
            src = inspect.getsource(fn)
            calls = _re.findall(r"\.read_text\(([^)]*)\)", src)
            assert calls, f"{fn.__name__} has no read_text call to check"
            for args in calls:
                assert 'encoding="utf-8"' in args, (
                    f"{fn.__name__}: read_text({args}) is locale-dependent; pass encoding=\"utf-8\""
                )


    def test_own_checkout_loose_ref_stamps_head(self, tmp_path, monkeypatch):
        from views_evaluation.evaluation import metric_frame as mf_mod

        repo = tmp_path / "repo"
        _git_dir(repo, "ref: refs/heads/main", refs={"refs/heads/main": "abcdef1234567890"})
        (repo / "pyproject.toml").write_text(_OWN_PYPROJECT)
        monkeypatch.setattr(mf_mod, "__file__", str(_fake_package_file(repo)))

        assert mf_mod._source_git_sha() == "abcdef1"

    def test_own_checkout_packed_ref_stamps_head(self, tmp_path, monkeypatch):
        """After `git pack-refs`/`git gc` the branch has no loose file."""
        from views_evaluation.evaluation import metric_frame as mf_mod

        repo = tmp_path / "repo"
        _git_dir(repo, "ref: refs/heads/main",
                 packed="# pack-refs with: peeled fully-peeled sorted\n"
                        "abc1234567890abcdef0000000000000000000000 refs/heads/main\n")
        (repo / "pyproject.toml").write_text(_OWN_PYPROJECT)
        monkeypatch.setattr(mf_mod, "__file__", str(_fake_package_file(repo)))

        assert mf_mod._source_git_sha() == "abc1234"

    def test_pep621_hyphenated_name_is_this_distribution(self, tmp_path, monkeypatch):
        """`views-evaluation` and `views_evaluation` are the same distribution (PEP 503)."""
        from views_evaluation.evaluation import metric_frame as mf_mod

        repo = tmp_path / "repo"
        _git_dir(repo, "0123456abcdef0123456abcdef0123456abcdef01")
        (repo / "pyproject.toml").write_text('[project]\nname = "views-evaluation"\n')
        monkeypatch.setattr(mf_mod, "__file__", str(_fake_package_file(repo)))

        assert mf_mod._source_git_sha() == "0123456"

    def test_linked_worktree_on_a_branch_stamps_head(self, tmp_path, monkeypatch):
        """`git worktree add ../wt -b feature`: `.git` is a file with an absolute gitdir,
        HEAD is symbolic in the worktree's own dir, and the ref lives in the main
        repository's dir named by `commondir`."""
        from views_evaluation.evaluation import metric_frame as mf_mod

        main = tmp_path / "main"
        _git_dir(main, "ref: refs/heads/main",
                 refs={"refs/heads/main": "1111111111111111111111111111111111111111",
                       "refs/heads/feature": "feedface1234567890feedface1234567890feed"})
        wt_gitdir = main / ".git" / "worktrees" / "wt"
        wt_gitdir.mkdir(parents=True)
        (wt_gitdir / "HEAD").write_text("ref: refs/heads/feature\n")
        (wt_gitdir / "commondir").write_text("../..\n")
        wt = tmp_path / "wt"
        wt.mkdir()
        (wt / ".git").write_text(f"gitdir: {wt_gitdir}\n")
        (wt / "pyproject.toml").write_text(_OWN_PYPROJECT)
        monkeypatch.setattr(mf_mod, "__file__", str(_fake_package_file(wt)))

        assert mf_mod._source_git_sha() == "feedfac"


class TestSourceGitShaBoundaryBeige:

    def test_submodule_relative_gitdir_resolves_against_the_git_file(self, tmp_path, monkeypatch):
        """A `git submodule update` clone has a `.git` *file* with a relative `gitdir:`
        and a detached (raw-SHA) HEAD; the path is relative to that file, not to
        wherever the process happens to be running."""
        from views_evaluation.evaluation import metric_frame as mf_mod

        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".git").write_text("gitdir: ../real.git\n")
        (repo / "pyproject.toml").write_text(_OWN_PYPROJECT)
        (tmp_path / "real.git").mkdir()
        (tmp_path / "real.git" / "HEAD").write_text("1234567deadbeef\n")
        # Two levels down, so a cwd-relative "../real.git" does NOT exist by accident.
        elsewhere = tmp_path / "elsewhere" / "deeper"
        elsewhere.mkdir(parents=True)
        monkeypatch.chdir(elsewhere)
        monkeypatch.setattr(mf_mod, "__file__", str(_fake_package_file(repo)))

        assert mf_mod._source_git_sha() == "1234567"
