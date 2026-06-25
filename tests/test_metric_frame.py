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
import tempfile

import numpy as np
import pytest

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

    def test_partition_level_default_to_empty_string(self):
        report = EvaluationReport('t', 'regression', 'point', _regression_point_results())
        mf = report.to_metric_frame()  # no partition/level
        assert set(mf.identifiers['partition'].tolist()) == {""}
        assert set(mf.identifiers['level'].tolist()) == {""}

    def test_fully_empty_report_yields_zero_row_frame(self):
        report = EvaluationReport('t', 'regression', 'point',
                                  {'month': {}, 'time_series': {}, 'step': {}})
        mf = report.to_metric_frame()
        assert mf.n_rows == 0
        assert mf.values.shape == (0, 1)
        assert mf.values.dtype == np.float32
        assert_frame_envelope(mf)  # empty frame still conforms + round-trips


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
