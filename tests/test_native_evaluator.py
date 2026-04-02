"""
Direct unit tests for NativeEvaluator.

Structured per ADR-020 (Red/Beige/Green):
  GREEN — schema grouping correctness, key formats, metric values, step filtering
  BEIGE — single-origin, classification target, sample predictions
  RED   — target not in config, invalid/unimplemented metric, schema key errors
"""
import numpy as np
import pytest

from views_evaluation.evaluation.evaluation_frame import EvaluationFrame
from views_evaluation.evaluation.native_evaluator import NativeEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_parallelogram_ef(n_origins=3, n_steps=4, n_units=2, start_time=100):
    """
    Build an EvaluationFrame representing a predictive parallelogram.
      - n_origins sequences (origins 0 .. n_origins-1)
      - n_steps lead times   (steps   1 .. n_steps)
      - n_units  locations   (units   1 .. n_units)
      - y_true == y_pred == 0  (so MSE == 0.0 everywhere)

    Row layout: for each (origin o, step s, unit u):
        time   = start_time + o + (s - 1)
        unit   = u
        origin = o
        step   = s
    """
    times, units, origins, steps = [], [], [], []
    for o in range(n_origins):
        for s in range(1, n_steps + 1):
            for u in range(1, n_units + 1):
                times.append(start_time + o + (s - 1))
                units.append(u)
                origins.append(o)
                steps.append(s)
    n = len(times)
    return EvaluationFrame(
        y_true=np.zeros(n),
        y_pred=np.zeros((n, 1)),
        identifiers={
            'time':   np.array(times),
            'unit':   np.array(units),
            'origin': np.array(origins),
            'step':   np.array(steps),
        },
        metadata={'target': 'test_target'},
    )


def _regression_point_config(steps=None, metrics=None):
    return {
        'steps': steps if steps is not None else [1, 2, 3, 4],
        'regression_targets': ['test_target'],
        'regression_point_metrics': metrics if metrics is not None else ['MSE'],
    }


# ---------------------------------------------------------------------------
# GREEN: Schema grouping and correctness
# ---------------------------------------------------------------------------

class TestNativeEvaluatorGreen:

    def test_report_has_three_schemas(self):
        ef = _make_parallelogram_ef()
        report = NativeEvaluator(_regression_point_config()).evaluate(ef)
        assert set(report.to_dict()['schemas'].keys()) == {'month', 'time_series', 'step'}

    def test_report_target_task_pred_type_fields(self):
        ef = _make_parallelogram_ef()
        report = NativeEvaluator(_regression_point_config()).evaluate(ef)
        assert report.target == 'test_target'
        assert report.task == 'regression'
        assert report.pred_type == 'point'

    def test_month_wise_group_count(self):
        """3 origins × 4 steps → 6 unique calendar months (100-105)."""
        ef = _make_parallelogram_ef(n_origins=3, n_steps=4, start_time=100)
        report = NativeEvaluator(_regression_point_config()).evaluate(ef)
        month_results = report.to_dict()['schemas']['month']
        assert len(month_results) == 6

    def test_month_wise_group_keys(self):
        ef = _make_parallelogram_ef(n_origins=3, n_steps=4, start_time=100)
        report = NativeEvaluator(_regression_point_config()).evaluate(ef)
        keys = set(report.to_dict()['schemas']['month'].keys())
        assert keys == {f"month{m}" for m in range(100, 106)}

    def test_time_series_wise_group_count(self):
        """One group per origin."""
        ef = _make_parallelogram_ef(n_origins=3)
        report = NativeEvaluator(_regression_point_config()).evaluate(ef)
        ts_results = report.to_dict()['schemas']['time_series']
        assert len(ts_results) == 3

    def test_time_series_wise_group_keys(self):
        ef = _make_parallelogram_ef(n_origins=3)
        report = NativeEvaluator(_regression_point_config()).evaluate(ef)
        keys = set(report.to_dict()['schemas']['time_series'].keys())
        assert keys == {'ts00', 'ts01', 'ts02'}

    def test_step_wise_configured_steps_are_populated(self):
        ef = _make_parallelogram_ef(n_steps=4)
        report = NativeEvaluator(_regression_point_config(steps=[1, 2, 3, 4])).evaluate(ef)
        step_results = report.to_dict()['schemas']['step']
        populated = {k: v for k, v in step_results.items() if v}
        assert len(populated) == 4

    def test_mse_is_zero_when_predictions_match_actuals(self):
        """y_true == y_pred == 0 ⟹ MSE == 0 in every group."""
        ef = _make_parallelogram_ef()
        report = NativeEvaluator(_regression_point_config()).evaluate(ef)
        for schema_name, schema_groups in report.to_dict()['schemas'].items():
            for group_key, metrics in schema_groups.items():
                if metrics:
                    assert metrics['MSE'] == pytest.approx(0.0), \
                        f"MSE != 0 in {schema_name}/{group_key}"

    def test_step_filtering_only_configured_steps_initialised(self):
        """Steps beyond max(config['steps']) must not appear in the result."""
        ef = _make_parallelogram_ef(n_steps=4)
        config = _regression_point_config(steps=[1, 2])  # cap at step 2
        report = NativeEvaluator(config).evaluate(ef)
        step_keys = set(report.to_dict()['schemas']['step'].keys())
        assert step_keys == {'step01', 'step02'}

    def test_sparse_step_config_only_declared_steps_returned(self):
        """steps=[1,3,6] must produce exactly step01, step03, step06 — not step02, step04, step05."""
        ef = _make_parallelogram_ef(n_steps=6, n_origins=2)
        config = _regression_point_config(steps=[1, 3, 6])
        report = NativeEvaluator(config).evaluate(ef)
        step_keys = set(report.to_dict()['schemas']['step'].keys())
        assert step_keys == {'step01', 'step03', 'step06'}
        assert 'step02' not in step_keys
        assert 'step04' not in step_keys
        assert 'step05' not in step_keys

    def test_legacy_compatibility_true_truncates_to_shortest_sequence(self):
        """
        If origin 0 has 4 steps and origin 1 has 2 steps, legacy_compatibility=True
        should truncate step results to max_step=2.
        """
        rows = []
        for s in range(1, 5):   # origin 0: 4 steps
            rows.append((100 + s - 1, 1, 0, s))
        for s in range(1, 3):   # origin 1: 2 steps
            rows.append((101 + s - 1, 1, 1, s))
        n = len(rows)
        ef = EvaluationFrame(
            y_true=np.zeros(n),
            y_pred=np.zeros((n, 1)),
            identifiers={
                'time':   np.array([r[0] for r in rows]),
                'unit':   np.array([r[1] for r in rows]),
                'origin': np.array([r[2] for r in rows]),
                'step':   np.array([r[3] for r in rows]),
            },
            metadata={'target': 'test_target'},
        )
        config = _regression_point_config(steps=[1, 2, 3, 4])
        report = NativeEvaluator(config).evaluate(ef, legacy_compatibility=True)
        step_results = report.to_dict()['schemas']['step']
        # Steps 1 and 2 should be populated; steps 3 and 4 should be empty (truncated)
        assert bool(step_results.get('step01'))  # non-empty dict
        assert bool(step_results.get('step02'))  # non-empty dict
        assert not step_results.get('step03')    # empty dict (truncated)
        assert not step_results.get('step04')    # empty dict (truncated)

    def test_legacy_compatibility_false_includes_all_steps_with_data(self):
        """With legacy_compatibility=False, all steps that have data are populated."""
        rows = []
        for s in range(1, 5):   # origin 0: 4 steps
            rows.append((100 + s - 1, 1, 0, s))
        for s in range(1, 3):   # origin 1: 2 steps
            rows.append((101 + s - 1, 1, 1, s))
        n = len(rows)
        ef = EvaluationFrame(
            y_true=np.zeros(n),
            y_pred=np.zeros((n, 1)),
            identifiers={
                'time':   np.array([r[0] for r in rows]),
                'unit':   np.array([r[1] for r in rows]),
                'origin': np.array([r[2] for r in rows]),
                'step':   np.array([r[3] for r in rows]),
            },
            metadata={'target': 'test_target'},
        )
        config = _regression_point_config(steps=[1, 2, 3, 4])
        report = NativeEvaluator(config).evaluate(ef, legacy_compatibility=False)
        step_results = report.to_dict()['schemas']['step']
        assert bool(step_results.get('step03'))  # has data from origin 0
        assert bool(step_results.get('step04'))  # has data from origin 0


# ---------------------------------------------------------------------------
# BEIGE: Edge cases
# ---------------------------------------------------------------------------

class TestNativeEvaluatorBeige:

    def test_single_origin_single_step(self):
        ef = EvaluationFrame(
            y_true=np.array([1.0, 2.0]),
            y_pred=np.array([[1.0], [2.0]]),
            identifiers={
                'time':   np.array([100, 100]),
                'unit':   np.array([1, 2]),
                'origin': np.array([0, 0]),
                'step':   np.array([1, 1]),
            },
            metadata={'target': 'test_target'},
        )
        config = _regression_point_config(steps=[1])
        report = NativeEvaluator(config).evaluate(ef)
        d = report.to_dict()['schemas']
        assert 'month100' in d['month']
        assert 'ts00' in d['time_series']
        assert 'step01' in d['step']

    def test_multi_target_regression_and_classification(self):
        """Config with both target types; each evaluated separately via EvaluationFrame metadata."""
        n = 4
        config = {
            'steps': [1, 2],
            'regression_targets': ['ged_sb'],
            'classification_targets': ['by_sb'],
            'regression_point_metrics': ['MSE'],
            'classification_point_metrics': ['AP'],
        }
        # Evaluate regression target
        ef_reg = EvaluationFrame(
            y_true=np.array([1.0, 2.0, 3.0, 4.0]),
            y_pred=np.array([[1.1], [2.1], [3.1], [4.1]]),
            identifiers={
                'time':   np.array([100, 100, 101, 101]),
                'unit':   np.array([1, 2, 1, 2]),
                'origin': np.zeros(n, dtype=int),
                'step':   np.array([1, 1, 2, 2]),
            },
            metadata={'target': 'ged_sb'},
        )
        report_reg = NativeEvaluator(config).evaluate(ef_reg)
        assert report_reg.task == 'regression'
        assert 'MSE' in report_reg.to_dict()['schemas']['month']['month100']

        # Evaluate classification target
        ef_cls = EvaluationFrame(
            y_true=np.array([0.0, 1.0, 0.0, 1.0]),
            y_pred=np.array([[0.2], [0.8], [0.3], [0.7]]),
            identifiers={
                'time':   np.array([100, 100, 101, 101]),
                'unit':   np.array([1, 2, 1, 2]),
                'origin': np.zeros(n, dtype=int),
                'step':   np.array([1, 1, 2, 2]),
            },
            metadata={'target': 'by_sb'},
        )
        report_cls = NativeEvaluator(config).evaluate(ef_cls)
        assert report_cls.task == 'classification'
        assert 'AP' in report_cls.to_dict()['schemas']['month']['month100']

    def test_classification_target(self):
        n = 6
        ef = EvaluationFrame(
            y_true=np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
            y_pred=np.array([[0.2], [0.8], [0.3], [0.7], [0.4], [0.6]]),
            identifiers={
                'time':   np.array([100, 100, 101, 101, 102, 102]),
                'unit':   np.array([1, 2, 1, 2, 1, 2]),
                'origin': np.zeros(n, dtype=int),
                'step':   np.array([1, 1, 2, 2, 3, 3]),
            },
            metadata={'target': 'by_sb_best'},
        )
        config = {
            'steps': [1, 2, 3],
            'classification_targets': ['by_sb_best'],
            'classification_point_metrics': ['AP'],
        }
        report = NativeEvaluator(config).evaluate(ef)
        assert report.task == 'classification'
        assert report.pred_type == 'point'
        assert 'month100' in report.to_dict()['schemas']['month']

    def test_classification_sample_brier(self):
        """Brier_sample and CRPS work for classification sample predictions."""
        n = 6
        ef = EvaluationFrame(
            y_true=np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
            y_pred=np.random.default_rng(42).uniform(0, 2, size=(n, 20)),
            identifiers={
                'time':   np.array([100, 100, 101, 101, 102, 102]),
                'unit':   np.array([1, 2, 1, 2, 1, 2]),
                'origin': np.zeros(n, dtype=int),
                'step':   np.array([1, 1, 2, 2, 3, 3]),
            },
            metadata={'target': 'by_sb_best'},
        )
        config = {
            'steps': [1, 2, 3],
            'classification_targets': ['by_sb_best'],
            'classification_sample_metrics': ['Brier_sample', 'CRPS'],
        }
        report = NativeEvaluator(config).evaluate(ef)
        assert report.task == 'classification'
        assert report.pred_type == 'sample'
        d = report.to_dict()['schemas']
        assert 'Brier_sample' in d['month']['month100']
        assert 'CRPS' in d['month']['month100']

    def test_classification_point_brier(self):
        """AP and Brier_point work together for classification point predictions."""
        n = 6
        ef = EvaluationFrame(
            y_true=np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
            y_pred=np.array([[0.2], [0.8], [0.3], [0.7], [0.4], [0.6]]),
            identifiers={
                'time':   np.array([100, 100, 101, 101, 102, 102]),
                'unit':   np.array([1, 2, 1, 2, 1, 2]),
                'origin': np.zeros(n, dtype=int),
                'step':   np.array([1, 1, 2, 2, 3, 3]),
            },
            metadata={'target': 'by_sb_best'},
        )
        config = {
            'steps': [1, 2, 3],
            'classification_targets': ['by_sb_best'],
            'classification_point_metrics': ['AP', 'Brier_point'],
        }
        report = NativeEvaluator(config).evaluate(ef)
        d = report.to_dict()['schemas']
        assert 'AP' in d['step']['step01']
        assert 'Brier_point' in d['step']['step01']

    def test_evaluate_twice_produces_identical_results(self):
        """NativeEvaluator is stateless — same input yields same output."""
        ef = _make_parallelogram_ef(n_origins=2, n_steps=3, n_units=2)
        config = _regression_point_config(steps=[1, 2, 3])
        evaluator = NativeEvaluator(config)
        report1 = evaluator.evaluate(ef)
        report2 = evaluator.evaluate(ef)
        assert report1.to_dict() == report2.to_dict()

    def test_sample_predictions_produce_point_pred_type_false(self):
        n = 4
        ef = EvaluationFrame(
            y_true=np.zeros(n),
            y_pred=np.random.rand(n, 50),
            identifiers={
                'time':   np.array([100, 100, 101, 101]),
                'unit':   np.array([1, 2, 1, 2]),
                'origin': np.zeros(n, dtype=int),
                'step':   np.array([1, 1, 2, 2]),
            },
            metadata={'target': 'test_target'},
        )
        config = {
            'steps': [1, 2],
            'regression_targets': ['test_target'],
            'regression_sample_metrics': ['CRPS'],
        }
        report = NativeEvaluator(config).evaluate(ef)
        assert report.pred_type == 'sample'
        assert report.task == 'regression'


# ---------------------------------------------------------------------------
# RED: Failure modes
# ---------------------------------------------------------------------------

class TestNativeEvaluatorRed:

    def test_target_not_in_config_raises_value_error(self):
        n = 4
        ef = EvaluationFrame(
            y_true=np.zeros(n), y_pred=np.zeros((n, 1)),
            identifiers={
                'time': np.array([100, 100, 101, 101]),
                'unit': np.array([1, 2, 1, 2]),
                'origin': np.zeros(n, dtype=int),
                'step': np.array([1, 1, 2, 2]),
            },
            metadata={'target': 'undeclared_target'},
        )
        config = {
            'steps': [1, 2],
            'regression_targets': ['some_other_target'],
            'regression_point_metrics': ['MSE'],
        }
        with pytest.raises(ValueError, match="undeclared_target"):
            NativeEvaluator(config).evaluate(ef)

    def test_unimplemented_metric_raises_value_error_with_clear_message(self):
        ef = _make_parallelogram_ef(n_origins=1, n_steps=2, n_units=2)
        config = _regression_point_config(steps=[1, 2], metrics=['SD'])
        with pytest.raises(ValueError, match="not yet implemented"):
            NativeEvaluator(config).evaluate(ef)

    def test_invalid_metric_name_raises_value_error(self):
        ef = _make_parallelogram_ef(n_origins=1, n_steps=2, n_units=2)
        config = _regression_point_config(steps=[1, 2], metrics=['NOSUCHMETRIC'])
        with pytest.raises(ValueError, match="not valid"):
            NativeEvaluator(config).evaluate(ef)

    def test_empty_config_accepted_at_init_fails_at_evaluate(self):
        """Empty config is accepted at init (C-02 known gap) but fails at evaluate().

        NativeEvaluator.__init__ only validates profile name (defaults to 'base').
        Structural config errors surface at evaluate() time, not construction.
        """
        ef = _make_parallelogram_ef(n_origins=1, n_steps=2, n_units=2)
        evaluator = NativeEvaluator({})  # does NOT raise — C-02
        with pytest.raises((ValueError, KeyError)):
            evaluator.evaluate(ef)
    def test_classification_metric_on_regression_target_raises(self):
        """AP is only valid for classification; using it with regression_targets must fail."""
        ef = _make_parallelogram_ef(n_origins=1, n_steps=2, n_units=2)
        config = {
            'steps': [1, 2],
            'regression_targets': ['test_target'],
            'regression_point_metrics': ['AP'],  # AP is not in REGRESSION_POINT_NATIVE
        }
        with pytest.raises(ValueError, match="not valid"):
            NativeEvaluator(config).evaluate(ef)
