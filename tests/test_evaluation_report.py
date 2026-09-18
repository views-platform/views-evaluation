"""
Direct unit tests for EvaluationReport.

Structured per ADR-020 (Red/Beige/Green):
  GREEN — construction, to_dict, get_schema_results for all four task/type combos
  BEIGE — empty schema, multiple metrics per group
  RED   — unknown schema key, invalid task/pred_type combination, the removed
          to_dataframe() surface staying removed

Until 2.0.0 this module import-skipped on pandas for the to_dataframe() tests, so
without the `dataframe` extra every test here was silently skipped. It now runs in
every environment; the report has no pandas surface left.
"""
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

class TestEvaluationReportBeige:

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

    def test_removed_dataframe_surface_stays_removed(self):
        """2.0.0 removed to_dataframe() and the evaluation_dict_to_dataframe helper after
        1.1.0's DeprecationWarning (ADR-022 §2; register C-40), and the three
        make_*_evaluation_dict factories as private cleanup (unused since 0.4.0). This
        pins those five NAMES, on the class and on an instance (a `__getattr__` shim
        serves instances only). A helper under another name is not caught here; the
        import-purity guard catches any pandas import whatever the name."""
        report = EvaluationReport('t', 'regression', 'point', _regression_point_results())
        assert not hasattr(report, "to_dataframe")
        assert not hasattr(type(report), "to_dataframe")
        for cls in (RegressionPointEvaluationMetrics, RegressionSampleEvaluationMetrics,
                    ClassificationPointEvaluationMetrics, ClassificationSampleEvaluationMetrics):
            for name in ("evaluation_dict_to_dataframe", "make_time_series_wise_evaluation_dict",
                         "make_step_wise_evaluation_dict", "make_month_wise_evaluation_dict"):
                assert not hasattr(cls, name), f"{cls.__name__}.{name} is back"
                assert not hasattr(cls(), name), f"{cls.__name__}().{name} is served by a shim"

    def test_metric_with_no_dataclass_field_raises(self):
        """FM1 guard: a computed metric that has no field on the typed dataclass raises
        instead of being dropped. The CIC has claimed a Red test for this since before
        2.0.0; there was none, and the guard could be replaced by `continue` unnoticed."""
        report = EvaluationReport('t', 'regression', 'point', {
            'month': {'month100': {'MSE': 1.0, 'BOGUS': 9.9}}, 'time_series': {}, 'step': {}})
        with pytest.raises(ValueError, match="no field exists"):
            report.get_schema_results('month')

    def test_get_schema_results_unknown_schema_raises_key_error(self):
        report = EvaluationReport('t', 'regression', 'point', {})
        with pytest.raises(KeyError, match="nonexistent"):
            report.get_schema_results('nonexistent')

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
