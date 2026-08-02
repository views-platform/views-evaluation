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
        # ADR-015 ruling 7: this used to return step03/step04 as EMPTY DICTS —
        # requested by the caller, present in the report, looking evaluated, scoring
        # nothing and emitting no MetricFrame rows. Silently not fulfilling an explicit
        # request is now a loud failure (C-20, truncation half).
        config = _regression_point_config(steps=[1, 2, 3, 4])
        with pytest.raises(ValueError, match="legacy_compatibility=True truncates"):
            NativeEvaluator(config).evaluate(ef, legacy_compatibility=True)

    def test_legacy_compatibility_true_succeeds_when_only_available_steps_requested(self):
        """Truncation itself is still supported — ask only for steps that exist.

        The flag's legacy-parity purpose is preserved: the caller declares the steps
        the shortest sequence can supply and gets exactly those, fully scored.
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
        config = _regression_point_config(steps=[1, 2])   # only what origin 1 can supply
        report = NativeEvaluator(config).evaluate(ef, legacy_compatibility=True)
        step_results = report.to_dict()['schemas']['step']
        assert bool(step_results.get('step01'))
        assert bool(step_results.get('step02'))
        assert set(step_results) == {'step01', 'step02'}, \
            "no empty placeholder keys should survive"

    def test_legacy_compatibility_true_is_a_noop_for_equal_length_origins(self):
        """Equal-length sequences truncate nothing, so the flag must not interfere."""
        rows = []
        for o in (0, 1):
            for s in range(1, 4):
                rows.append((100 + o + s - 1, 1, o, s))
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
        config = _regression_point_config(steps=[1, 2, 3])
        report = NativeEvaluator(config).evaluate(ef, legacy_compatibility=True)
        step_results = report.to_dict()['schemas']['step']
        assert all(bool(step_results.get(f'step0{s}')) for s in (1, 2, 3))

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
        """Brier_cls_sample and CRPS work for classification sample predictions."""
        n = 6
        ef = EvaluationFrame(
            y_true=np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
            y_pred=np.random.default_rng(42).uniform(0, 1, size=(n, 20)),  # probability samples in [0,1]
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
            'classification_sample_metrics': ['Brier_cls_sample', 'CRPS'],
        }
        report = NativeEvaluator(config).evaluate(ef)
        assert report.task == 'classification'
        assert report.pred_type == 'sample'
        d = report.to_dict()['schemas']
        assert 'Brier_cls_sample' in d['month']['month100']
        assert 'CRPS' in d['month']['month100']

    def test_classification_point_brier(self):
        """AP and Brier_cls_point work together for classification point predictions."""
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
            'classification_point_metrics': ['AP', 'Brier_cls_point'],
        }
        report = NativeEvaluator(config).evaluate(ef)
        d = report.to_dict()['schemas']
        assert 'AP' in d['step']['step01']
        assert 'Brier_cls_point' in d['step']['step01']

    def test_evaluate_twice_produces_identical_results(self):
        """NativeEvaluator is stateless — same input yields same output."""
        ef = _make_parallelogram_ef(n_origins=2, n_steps=3, n_units=2)
        config = _regression_point_config(steps=[1, 2, 3])
        evaluator = NativeEvaluator(config)
        report1 = evaluator.evaluate(ef)
        report2 = evaluator.evaluate(ef)
        assert report1.to_dict() == report2.to_dict()

    def test_step_values_above_999_not_silently_dropped(self):
        """Steps >= 1000 must not be silently dropped by a hardcoded sentinel (C-17)."""
        n = 4
        ef = EvaluationFrame(
            y_true=np.zeros(n),
            y_pred=np.zeros((n, 1)),
            identifiers={
                'time':   np.array([2000, 2000, 2001, 2001]),
                'unit':   np.array([1, 2, 1, 2]),
                'origin': np.zeros(n, dtype=int),
                'step':   np.array([1000, 1000, 1001, 1001]),
            },
            metadata={'target': 'test_target'},
        )
        config = _regression_point_config(steps=[1000, 1001])
        report = NativeEvaluator(config).evaluate(ef, legacy_compatibility=False)
        step_results = report.to_dict()['schemas']['step']
        # Metrics must be computed (non-empty dict), not just pre-initialized
        assert 'MSE' in step_results.get('step1000', {}), \
            "Step 1000 metrics were silently dropped by sentinel"
        assert 'MSE' in step_results.get('step1001', {}), \
            "Step 1001 metrics were silently dropped by sentinel"

    def test_nan_metric_result_is_finite_checkable(self):
        """Metric results that are NaN (e.g., Pearson on constant data) must be
        detectable via np.isfinite. This documents that NaN can appear in results
        when data is degenerate, and callers should check.

        ADR-015 ruling 2 (as revised): a constant series is a property of the data,
        not a fault, so this is a documented sentinel rather than a raise.
        """
        n = 4
        ef = EvaluationFrame(
            y_true=np.array([1.0, 1.0, 1.0, 1.0]),  # constant → Pearson = NaN
            y_pred=np.array([[1.0], [1.0], [1.0], [1.0]]),
            identifiers={
                'time':   np.array([100, 100, 101, 101]),
                'unit':   np.array([1, 2, 1, 2]),
                'origin': np.zeros(n, dtype=int),
                'step':   np.array([1, 1, 2, 2]),
            },
            metadata={'target': 'test_target'},
        )
        config = _regression_point_config(steps=[1, 2], metrics=['Pearson'])
        report = NativeEvaluator(config).evaluate(ef)
        month_results = report.to_dict()['schemas']['month']
        pearson_val = month_results['month100']['Pearson']
        assert np.isnan(pearson_val), "Pearson on constant data should be NaN"
        # Callers can detect this with np.isfinite
        assert not np.isfinite(pearson_val)

    def test_constant_baseline_model_evaluates_without_crashing(self):
        """Regression guard for the ADR-015 ruling-2 reversal (2026-08-02).

        A "predict zero everywhere" baseline has a constant prediction series. ADR-041
        states reports exist to compare "ensemble models against constituent models and
        baselines", so this is a routine workflow.

        For one day, Pearson raised on constant input — which aborted the ENTIRE
        evaluation (every metric, every schema) whenever a baseline was scored. This
        test pins that it must not happen again: the baseline evaluates, MSE is real,
        and Pearson records NaN for the undefined groups.
        """
        n = 6
        ef = EvaluationFrame(
            y_true=np.array([0.0, 5.0, 0.0, 12.0, 3.0, 0.0]),   # varied, real data
            y_pred=np.zeros((n, 1)),                             # the baseline
            identifiers={
                'time':   np.array([100, 100, 100, 101, 101, 101]),
                'unit':   np.array([1, 2, 3, 1, 2, 3]),
                'origin': np.zeros(n, dtype=int),
                'step':   np.array([1, 1, 1, 2, 2, 2]),
            },
            metadata={'target': 'test_target'},
        )
        config = _regression_point_config(steps=[1, 2], metrics=['MSE', 'Pearson'])
        report = NativeEvaluator(config).evaluate(ef)          # must not raise
        month = report.to_dict()['schemas']['month']
        assert np.isfinite(month['month100']['MSE']), "MSE must still be computed"
        assert np.isnan(month['month100']['Pearson']), "Pearson undefined → NaN, not a crash"

    def test_cross_schema_consistency_mse_values(self):
        """MSE computed via month-wise on a single-month window must equal
        step-wise MSE for the same data slice."""
        # Single origin, single step, single month → all schemas see same data
        n = 4
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([[1.5], [2.5], [3.5], [4.5]])
        ef = EvaluationFrame(
            y_true=y_true,
            y_pred=y_pred,
            identifiers={
                'time':   np.array([100, 100, 100, 100]),
                'unit':   np.array([1, 2, 3, 4]),
                'origin': np.zeros(n, dtype=int),
                'step':   np.ones(n, dtype=int),
            },
            metadata={'target': 'test_target'},
        )
        config = _regression_point_config(steps=[1], metrics=['MSE'])
        report = NativeEvaluator(config).evaluate(ef)
        schemas = report.to_dict()['schemas']
        mse_month = schemas['month']['month100']['MSE']
        mse_step = schemas['step']['step01']['MSE']
        mse_ts = schemas['time_series']['ts00']['MSE']
        # All three schemas see the same 4 observations → same MSE
        assert mse_month == pytest.approx(mse_step, abs=1e-12)
        assert mse_month == pytest.approx(mse_ts, abs=1e-12)
        # And the value is correct: mean((0.5)^2) = 0.25
        assert mse_month == pytest.approx(0.25, abs=1e-12)

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
        """Unknown metric names now fail at construction, not at evaluate() (C-02)."""
        config = _regression_point_config(steps=[1, 2], metrics=['NOSUCHMETRIC'])
        with pytest.raises(ValueError, match="Unknown metric"):
            NativeEvaluator(config)

    def test_empty_config_rejected_at_init(self):
        """An empty config fails loudly at construction (C-02, ADR-015 rulings 4/5).

        Supersedes ``test_empty_config_accepted_at_init_fails_at_evaluate``, which
        asserted the pre-ADR-015 behaviour: that ``NativeEvaluator({})`` constructed
        successfully and only failed later. That leniency was the Tier-1 defect.
        """
        with pytest.raises(ValueError):
            NativeEvaluator({})

    # ── C-02 / ADR-015 rulings 4 & 5: config must fail loud at construction ──────
    #
    # Each of these previously produced an empty-but-successful-looking report.

    def test_misspelled_metric_list_key_rejected_at_init(self):
        """The exact C-02 reproduction: a missing 's' silently emptied the report."""
        config = {
            'steps': [1],
            'regression_targets': ['target'],
            'regression_sample_metric': ['CRPS'],   # typo — should be ..._metrics
        }
        with pytest.raises(ValueError, match="closely resembles"):
            NativeEvaluator(config)

    def test_typo_is_reported_but_never_substituted(self):
        """A suspected typo must be named, and then NOT acted on.

        Inferring what the caller meant and proceeding would be exactly the silent
        repair ADR-015 forbids. The error names the resemblance; the caller fixes it.
        """
        config = {'steps': [1], 'regression_targets': ['target'],
                  'regression_sample_metric': ['CRPS']}
        with pytest.raises(ValueError) as excinfo:
            NativeEvaluator(config)
        msg = str(excinfo.value)
        assert "regression_sample_metrics" in msg, "must name what it resembles"
        assert "NOT been interpreted" in msg, "must state that nothing was assumed"

    def test_short_key_typo_rejected_at_init(self):
        """'step' vs 'steps' — one character, and previously silent."""
        config = {'step': [1], 'steps': [1], 'regression_targets': ['target'],
                  'regression_sample_metrics': ['CRPS']}
        with pytest.raises(ValueError, match="closely resembles"):
            NativeEvaluator(config)

    def test_legacy_config_keys_rejected_at_init(self):
        """Legacy keys removed in 0.4.0 fail loudly and are NOT translated (C-29).

        Detected by exact name from an enumerated set — not inferred, and never
        silently mapped to their replacement.
        """
        for legacy in ('targets', 'metrics',
                       'regression_uncertainty_metrics', 'classification_uncertainty_metrics'):
            config = {'steps': [1], 'regression_targets': ['target'],
                      'regression_point_metrics': ['MSE'], legacy: ['x']}
            with pytest.raises(ValueError, match="Legacy evaluation config key"):
                NativeEvaluator(config)

    def test_missing_steps_rejected_at_init(self):
        """Without 'steps' the entire step-wise schema was silently absent."""
        config = {'regression_targets': ['target'], 'regression_sample_metrics': ['CRPS']}
        with pytest.raises(ValueError, match="non-empty 'steps'"):
            NativeEvaluator(config)

    def test_empty_steps_rejected_at_init(self):
        config = {'steps': [], 'regression_targets': ['target'],
                  'regression_sample_metrics': ['CRPS']}
        with pytest.raises(ValueError, match="non-empty 'steps'"):
            NativeEvaluator(config)

    def test_non_positive_step_rejected_at_init(self):
        """'steps' is 1-indexed; step 0 would silently match no group."""
        config = {'steps': [0, 1], 'regression_targets': ['target'],
                  'regression_sample_metrics': ['CRPS']}
        with pytest.raises(ValueError, match="1-indexed positive integers"):
            NativeEvaluator(config)

    def test_no_targets_rejected_at_init(self):
        config = {'steps': [1], 'regression_sample_metrics': ['CRPS']}
        with pytest.raises(ValueError, match="declares no targets"):
            NativeEvaluator(config)

    def test_targets_without_metrics_rejected_at_init(self):
        """Declaring a task with no metrics for it evaluated every group to {}."""
        config = {'steps': [1], 'regression_targets': ['target']}
        with pytest.raises(ValueError, match="provides no metrics"):
            NativeEvaluator(config)

    def test_metric_invalid_for_its_cell_rejected_at_init(self):
        """CRPS is a sample metric; declaring it under point metrics is a config error."""
        config = {'steps': [1], 'regression_targets': ['target'],
                  'regression_point_metrics': ['CRPS']}
        with pytest.raises(ValueError, match="not valid for"):
            NativeEvaluator(config)

    def test_non_dict_config_rejected_at_init(self):
        with pytest.raises(ValueError, match="must be a dict"):
            NativeEvaluator(['steps', 1])

    def test_sample_frame_without_sample_metrics_fails_at_evaluate(self):
        """pred_type depends on the frame, so this can only be caught at evaluate().

        A point-only config evaluating a sample frame previously produced empty
        per-group dicts with no error.
        """
        # Built inline: the shared helper is point-only (y_pred is (n, 1)).
        ef = EvaluationFrame(
            y_true=np.zeros(4),
            y_pred=np.zeros((4, 3)),          # 3 samples per row -> pred_type == 'sample'
            identifiers={
                'time':   np.array([100, 100, 101, 101]),
                'unit':   np.array([1, 2, 1, 2]),
                'origin': np.array([0, 0, 0, 0]),
                'step':   np.array([1, 1, 2, 2]),
            },
            metadata={'target': 'test_target'},
        )
        config = {'steps': [1, 2], 'regression_targets': ['test_target'],
                  'regression_point_metrics': ['MSE']}
        with pytest.raises(ValueError, match="No metrics configured for"):
            NativeEvaluator(config).evaluate(ef)

    def test_metric_function_error_includes_metric_name(self):
        """When a metric function raises, the error message must name the metric (C-16)."""
        import dataclasses
        from unittest.mock import patch, MagicMock
        from views_evaluation.evaluation.metric_catalog import METRIC_CATALOG

        ef = _make_parallelogram_ef(n_origins=1, n_steps=2, n_units=2)
        config = _regression_point_config(steps=[1, 2], metrics=['MSE'])

        # Inject a failure into MSE's function
        original_spec = METRIC_CATALOG['MSE']
        broken_fn = MagicMock(side_effect=RuntimeError("sklearn internal error"))
        broken_spec = dataclasses.replace(original_spec, function=broken_fn)
        with patch.dict(METRIC_CATALOG, {'MSE': broken_spec}):
            with pytest.raises(ValueError, match="MSE"):
                NativeEvaluator(config).evaluate(ef)

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


# ---------------------------------------------------------------------------
# BEIGE: the combined-config contract with views-pipeline-core
#
# pipeline-core calls `NativeEvaluator(context.configs)` where `context.configs` is
# `_config_manager.get_combined_config()` — the WHOLE merged model config (meta +
# hyperparameters + deployment + partitions + queryset + sweep). It therefore carries
# dozens of keys this library knows nothing about, by design.
#
# A blanket "reject unrecognised keys" check was written on 2026-08-02 and would have
# broken every model in the platform: 17 rejected keys on a single real config, at
# `NativeEvaluator(...)`, before any evaluation ran. It was caught by reading
# pipeline-core before merging, not by the test suite — hence this file.
#
# Register C-33 tracks the underlying config-separation debt.
# ---------------------------------------------------------------------------

class TestCombinedConfigTolerance:

    def _real_combined_config(self):
        """A faithful combined config, keys taken verbatim from a real model.

        Source: views-models/models/ravaging_thief/configs/{config_meta,
        config_hyperparameters}.py — an NHiTS cm-level model.
        """
        return {
            # config_meta.py
            "name": "ravaging_thief", "algorithm": "NHiTSModel", "level": "cm",
            "creator": "Dylan", "queryset": "escwa001_cflong",
            "regression_targets": ["lr_ged_os"],
            "regression_point_baselines": ["average_cmbaseline", "zero_cmbaseline",
                                           "locf_cmbaseline"],
            "regression_point_metrics": ["MSLE", "MSE", "MCR_point", "y_hat_bar"],
            "regression_sample_metrics": ["CRPS", "y_hat_bar"],
            "rolling_origin_stride": 1, "prediction_format": "dataframe",
            # config_hyperparameters.py
            "steps": list(range(1, 37)), "input_chunk_length": 36,
            "output_chunk_length": 36, "output_chunk_shift": 0, "random_state": 67,
            "time_steps": 36, "num_samples": 1, "mc_dropout": False, "n_jobs": -1,
            "batch_size": 128, "n_epochs": 300,
        }

    def test_real_combined_model_config_is_accepted(self):
        """The regression guard. If this fails, every model in the platform is broken."""
        NativeEvaluator(self._real_combined_config())      # must not raise

    def test_foreign_keys_are_ignored_not_rejected(self):
        """Keys that plainly belong to another concern are none of this library's business."""
        config = {'steps': [1], 'regression_targets': ['t'],
                  'regression_sample_metrics': ['CRPS'],
                  'batch_size': 128, 'n_epochs': 300, 'algorithm': 'NHiTSModel',
                  'wandb_project': 'views', 'sweep_id': 'abc123'}
        NativeEvaluator(config)                             # must not raise

    def test_baseline_key_is_not_mistaken_for_a_metric_key(self):
        """`regression_point_baselines` is a REAL pipeline-core key.

        It shares a 17-character prefix with `regression_point_metrics`, so a loose
        fuzzy matcher would flag it as a typo and break every model that declares
        baselines. It must be ignored.
        """
        config = {'steps': [1], 'regression_targets': ['t'],
                  'regression_point_metrics': ['MSE'],
                  'regression_point_baselines': ['zero_cmbaseline'],
                  'regression_sample_baselines': ['red_ranger']}
        NativeEvaluator(config)                             # must not raise

    def test_typo_still_caught_inside_a_combined_config(self):
        """Tolerating foreign keys must not cost us the typo protection."""
        config = self._real_combined_config()
        del config['regression_sample_metrics']
        config['regression_sample_metric'] = ['CRPS']       # the typo
        with pytest.raises(ValueError, match="closely resembles"):
            NativeEvaluator(config)
