"""
Direct unit tests for EvaluationReport.

Structured per ADR-020 (Red/Beige/Green):
  GREEN — construction, to_dict, get_schema_results, to_dataframe for all four task/type combos
  BEIGE — empty schema, multiple metrics per group, raw schema passthrough
  RED   — unknown schema key, invalid task/pred_type combination
"""
import pandas as pd
import pytest

from views_evaluation.evaluation.evaluation_report import EvaluationReport
from views_evaluation.evaluation.metrics import (
    RegressionPointEvaluationMetrics,
    RegressionSampleEvaluationMetrics,
    ClassificationPointEvaluationMetrics,
    ClassificationSampleEvaluationMetrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _regression_point_results():
    return {
        'month':       {'month100': {'MSE': 1.0}, 'month101': {'MSE': 2.0}},
        'time_series': {'ts00': {'MSE': 1.5}},
        'step':        {'step01': {'MSE': 1.0}, 'step02': {'MSE': 2.0}},
    }

def _regression_sample_results():
    return {
        'month':       {'month100': {'CRPS': 0.5}},
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


# ---------------------------------------------------------------------------
# GREEN: Construction and core API
# ---------------------------------------------------------------------------

class TestEvaluationReportGreen:

    def test_construction_stores_all_fields(self):
        report = EvaluationReport(
            target='ged_sb_best',
            task='regression',
            pred_type='point',
            results=_regression_point_results(),
        )
        assert report.target == 'ged_sb_best'
        assert report.task == 'regression'
        assert report.pred_type == 'point'

    def test_repr_contains_target_and_task(self):
        report = EvaluationReport('ged_sb_best', 'regression', 'point', _regression_point_results())
        r = repr(report)
        assert 'ged_sb_best' in r
        assert 'regression' in r

    def test_to_dict_top_level_keys(self):
        report = EvaluationReport('t', 'regression', 'point', _regression_point_results())
        d = report.to_dict()
        assert set(d.keys()) == {'target', 'task', 'pred_type', 'schemas'}

    def test_to_dict_preserves_metric_values(self):
        report = EvaluationReport('t', 'regression', 'point', _regression_point_results())
        d = report.to_dict()
        assert d['schemas']['month']['month100']['MSE'] == 1.0
        assert d['schemas']['month']['month101']['MSE'] == 2.0

    def test_get_schema_results_regression_point_returns_correct_cls(self):
        report = EvaluationReport('t', 'regression', 'point', _regression_point_results())
        schema = report.get_schema_results('month')
        assert isinstance(schema['month100'], RegressionPointEvaluationMetrics)
        assert schema['month100'].MSE == 1.0
        assert schema['month101'].MSE == 2.0

    def test_get_schema_results_regression_sample_returns_correct_cls(self):
        report = EvaluationReport('t', 'regression', 'sample', _regression_sample_results())
        schema = report.get_schema_results('month')
        assert isinstance(schema['month100'], RegressionSampleEvaluationMetrics)
        assert schema['month100'].CRPS == 0.5

    def test_get_schema_results_classification_point_returns_correct_cls(self):
        report = EvaluationReport('by_sb', 'classification', 'point', _classification_point_results())
        schema = report.get_schema_results('month')
        assert isinstance(schema['month100'], ClassificationPointEvaluationMetrics)
        assert schema['month100'].AP == 0.8

    def test_get_schema_results_classification_sample_returns_correct_cls(self):
        report = EvaluationReport('by_sb', 'classification', 'sample', _classification_sample_results())
        schema = report.get_schema_results('month')
        assert isinstance(schema['month100'], ClassificationSampleEvaluationMetrics)
        assert schema['month100'].CRPS == 0.3

    def test_to_dataframe_returns_dataframe_with_correct_shape(self):
        report = EvaluationReport('t', 'regression', 'point', _regression_point_results())
        df = report.to_dataframe('month')
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2  # month100 and month101

    def test_to_dataframe_index_matches_group_keys(self):
        report = EvaluationReport('t', 'regression', 'point', _regression_point_results())
        df = report.to_dataframe('month')
        assert set(df.index.tolist()) == {'month100', 'month101'}

    def test_to_dataframe_time_series_schema(self):
        report = EvaluationReport('t', 'regression', 'point', _regression_point_results())
        df = report.to_dataframe('time_series')
        assert isinstance(df, pd.DataFrame)
        assert 'ts00' in df.index

    def test_to_dataframe_step_schema(self):
        report = EvaluationReport('t', 'regression', 'point', _regression_point_results())
        df = report.to_dataframe('step')
        assert isinstance(df, pd.DataFrame)
        assert 'step01' in df.index


# ---------------------------------------------------------------------------
# BEIGE: Edge cases
# ---------------------------------------------------------------------------

class TestEvaluationReportBeige:

    def test_to_dataframe_empty_schema_returns_empty_dataframe(self):
        results = {'month': {}, 'time_series': {}, 'step': {}}
        report = EvaluationReport('t', 'regression', 'point', results)
        df = report.to_dataframe('month')
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_get_schema_results_empty_schema_returns_empty_dict(self):
        results = {'month': {}, 'time_series': {}, 'step': {}}
        report = EvaluationReport('t', 'regression', 'point', results)
        assert report.get_schema_results('month') == {}

    def test_multiple_metrics_per_group_accessible(self):
        results = {
            'month': {'month100': {'MSE': 1.0, 'RMSLE': 0.5}},
            'time_series': {},
            'step': {},
        }
        report = EvaluationReport('t', 'regression', 'point', results)
        schema = report.get_schema_results('month')
        assert schema['month100'].MSE == 1.0
        assert schema['month100'].RMSLE == 0.5

    def test_to_dataframe_raw_returns_internal_results_dict_with_deprecation(self):
        """schema='raw' is a deprecated passthrough to the raw results dict."""
        import warnings
        results = {'month': {'month100': {'MSE': 42.0}}, 'time_series': {}, 'step': {}}
        report = EvaluationReport('t', 'regression', 'point', results)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            raw = report.to_dataframe('raw')
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "to_dict()" in str(w[0].message)
        assert 'month' in raw
        assert raw['month']['month100']['MSE'] == 42.0

    def test_all_four_task_pred_type_combinations_resolve_correctly(self):
        """get_schema_results must not raise for any valid (task, pred_type) pair."""
        combos = [
            ('regression', 'point',      _regression_point_results()),
            ('regression', 'sample',     _regression_sample_results()),
            ('classification', 'point',  _classification_point_results()),
            ('classification', 'sample', _classification_sample_results()),
        ]
        for task, pred_type, results in combos:
            report = EvaluationReport('t', task, pred_type, results)
            # Should not raise
            result = report.get_schema_results('month')
            assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# RED: Failure modes
# ---------------------------------------------------------------------------

class TestEvaluationReportRed:

    def test_non_dict_schema_value_fails_at_access(self):
        """Malformed result dict: schema value is a string, not a dict.

        Construction succeeds (no deep validation), but get_schema_results
        fails when it tries to iterate the non-dict value.
        """
        results = {'month': 'not_a_dict', 'time_series': {}, 'step': {}}
        report = EvaluationReport('t', 'regression', 'point', results)
        with pytest.raises(AttributeError):
            report.get_schema_results('month')

    def test_get_schema_results_unknown_schema_raises_key_error(self):
        report = EvaluationReport('t', 'regression', 'point', {})
        with pytest.raises(KeyError, match="nonexistent"):
            report.get_schema_results('nonexistent')

    def test_to_dataframe_unknown_schema_raises_key_error(self):
        report = EvaluationReport('t', 'regression', 'point', _regression_point_results())
        with pytest.raises(KeyError):
            report.to_dataframe('nonexistent_schema')

    def test_invalid_task_type_raises_on_get_schema_results(self):
        """An unrecognised (task, pred_type) pair must fail at lookup time."""
        results = {'month': {'m': {'MSE': 1.0}}, 'time_series': {}, 'step': {}}
        report = EvaluationReport('t', 'INVALID_TASK', 'point', results)
        with pytest.raises(KeyError):
            report.get_schema_results('month')

    def test_invalid_pred_type_raises_on_get_schema_results(self):
        results = {'month': {'m': {'MSE': 1.0}}, 'time_series': {}, 'step': {}}
        report = EvaluationReport('t', 'regression', 'INVALID_PRED_TYPE', results)
        with pytest.raises(KeyError):
            report.get_schema_results('month')
