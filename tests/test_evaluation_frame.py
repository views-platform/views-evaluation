"""
Direct unit tests for EvaluationFrame.

Structured per ADR-020 (Red/Beige/Green):
  GREEN — happy-path construction, properties, grouping, subsetting
  BEIGE — edge cases and "boring but dangerous" usage
  RED   — invalid inputs, structural violations, fail-loud behaviour
"""
import numpy as np
import pytest

from views_evaluation.evaluation.evaluation_frame import EvaluationFrame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_identifiers(n: int, origin: int = 0) -> dict:
    """Minimal valid identifiers dict for a single-origin frame of n rows."""
    return {
        'time':   np.arange(100, 100 + n),
        'unit':   np.ones(n, dtype=int),
        'origin': np.full(n, origin, dtype=int),
        'step':   np.arange(1, n + 1),
    }


# ---------------------------------------------------------------------------
# GREEN: Construction and properties
# ---------------------------------------------------------------------------

class TestEvaluationFrameGreen:

    def test_construction_point_forecast(self):
        n = 6
        ef = EvaluationFrame(
            y_true=np.ones(n),
            y_pred=np.ones((n, 1)),
            identifiers=_make_identifiers(n),
        )
        assert ef.n_rows == n
        assert ef.n_samples == 1
        assert ef.is_sample is False

    def test_construction_sample_forecast(self):
        n, s = 10, 50
        ef = EvaluationFrame(
            y_true=np.zeros(n),
            y_pred=np.zeros((n, s)),
            identifiers=_make_identifiers(n),
        )
        assert ef.n_rows == n
        assert ef.n_samples == s
        assert ef.is_sample is True

    def test_metadata_defaults_to_empty_dict(self):
        n = 3
        ef = EvaluationFrame(np.zeros(n), np.zeros((n, 1)), _make_identifiers(n))
        assert ef.metadata == {}

    def test_metadata_stored_when_provided(self):
        n = 3
        meta = {'target': 'ged_sb_best', 'model': 'test_model'}
        ef = EvaluationFrame(np.zeros(n), np.zeros((n, 1)), _make_identifiers(n), metadata=meta)
        assert ef.metadata['target'] == 'ged_sb_best'
        assert ef.metadata['model'] == 'test_model'

    def test_repr_contains_expected_fields(self):
        n = 4
        ef = EvaluationFrame(np.zeros(n), np.zeros((n, 1)), _make_identifiers(n))
        r = repr(ef)
        assert 'EvaluationFrame' in r
        assert 'n_rows=4' in r
        assert 'n_samples=1' in r

    def test_get_group_indices_single_group(self):
        n = 4
        ids = _make_identifiers(n)
        ids['origin'] = np.zeros(n, dtype=int)
        ef = EvaluationFrame(np.zeros(n), np.zeros((n, 1)), ids)
        groups = ef.get_group_indices('origin')
        assert len(groups) == 1
        assert 0 in groups
        assert len(groups[0]) == n

    def test_get_group_indices_multiple_groups(self):
        # 6 rows: origins 0,0,1,1,2,2
        n = 6
        ids = {
            'time':   np.array([100, 101, 101, 102, 102, 103]),
            'unit':   np.ones(n, dtype=int),
            'origin': np.array([0, 0, 1, 1, 2, 2]),
            'step':   np.array([1, 2, 1, 2, 1, 2]),
        }
        ef = EvaluationFrame(np.zeros(n), np.zeros((n, 1)), ids)
        groups = ef.get_group_indices('origin')
        assert set(groups.keys()) == {0, 1, 2}
        assert len(groups[0]) == 2
        assert len(groups[1]) == 2
        assert len(groups[2]) == 2

    def test_get_group_indices_maps_to_correct_row_indices(self):
        # origin 0 → rows 0,1; origin 1 → rows 2,3
        n = 4
        ids = {
            'time':   np.array([100, 101, 100, 101]),
            'unit':   np.ones(n, dtype=int),
            'origin': np.array([0, 0, 1, 1]),
            'step':   np.array([1, 2, 1, 2]),
        }
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        ef = EvaluationFrame(y_true, np.zeros((n, 1)), ids)
        groups = ef.get_group_indices('origin')
        assert set(ef.y_true[groups[0]].tolist()) == {10.0, 20.0}
        assert set(ef.y_true[groups[1]].tolist()) == {30.0, 40.0}

    def test_select_indices_returns_correct_subset(self):
        n = 6
        ef = EvaluationFrame(
            y_true=np.arange(6, dtype=float),
            y_pred=np.arange(6, dtype=float).reshape(-1, 1),
            identifiers=_make_identifiers(n),
        )
        sub = ef.select_indices(np.array([0, 2, 4]))
        assert sub.n_rows == 3
        np.testing.assert_array_equal(sub.y_true, [0.0, 2.0, 4.0])

    def test_select_indices_does_not_mutate_original(self):
        n = 4
        ef = EvaluationFrame(np.ones(n), np.ones((n, 1)), _make_identifiers(n))
        _ = ef.select_indices(np.array([0, 1]))
        assert ef.n_rows == n

    def test_select_indices_propagates_all_identifiers(self):
        n = 4
        ids = {
            'time':   np.array([100, 101, 102, 103]),
            'unit':   np.ones(n, dtype=int),
            'origin': np.zeros(n, dtype=int),
            'step':   np.array([1, 2, 3, 4]),
        }
        ef = EvaluationFrame(np.zeros(n), np.zeros((n, 1)), ids)
        sub = ef.select_indices(np.array([1, 3]))
        np.testing.assert_array_equal(sub.identifiers['time'], [101, 103])
        np.testing.assert_array_equal(sub.identifiers['step'], [2, 4])

    def test_select_indices_propagates_metadata(self):
        n = 3
        ef = EvaluationFrame(
            np.zeros(n), np.zeros((n, 1)), _make_identifiers(n),
            metadata={'target': 'my_target'}
        )
        sub = ef.select_indices(np.array([0, 1]))
        assert sub.metadata['target'] == 'my_target'


# ---------------------------------------------------------------------------
# BEIGE: Edge cases
# ---------------------------------------------------------------------------

class TestEvaluationFrameBeige:

    def test_single_row_frame(self):
        ef = EvaluationFrame(
            y_true=np.array([42.0]),
            y_pred=np.array([[42.0]]),
            identifiers={
                'time':   np.array([500]),
                'unit':   np.array([1]),
                'origin': np.array([0]),
                'step':   np.array([1]),
            },
        )
        assert ef.n_rows == 1

    def test_large_sample_count(self):
        n, s = 100, 1000
        ef = EvaluationFrame(np.zeros(n), np.zeros((n, s)), _make_identifiers(n))
        assert ef.n_samples == s
        assert ef.is_sample is True

    def test_get_group_indices_by_time_multiple_units_per_timestep(self):
        n = 6
        ids = {
            'time':   np.array([100, 100, 101, 101, 102, 102]),
            'unit':   np.array([1, 2, 1, 2, 1, 2]),
            'origin': np.zeros(n, dtype=int),
            'step':   np.array([1, 1, 2, 2, 3, 3]),
        }
        ef = EvaluationFrame(np.zeros(n), np.zeros((n, 1)), ids)
        groups = ef.get_group_indices('time')
        assert set(groups.keys()) == {100, 101, 102}
        for g in groups.values():
            assert len(g) == 2

    def test_select_indices_with_full_index_range(self):
        """Selecting all indices should produce an identical frame."""
        n = 4
        ef = EvaluationFrame(
            y_true=np.array([1.0, 2.0, 3.0, 4.0]),
            y_pred=np.ones((n, 1)),
            identifiers=_make_identifiers(n),
        )
        all_idx = np.arange(n)
        sub = ef.select_indices(all_idx)
        np.testing.assert_array_equal(sub.y_true, ef.y_true)
        assert sub.n_rows == ef.n_rows


# ---------------------------------------------------------------------------
# RED: Invalid inputs and structural violations
# ---------------------------------------------------------------------------

class TestEvaluationFrameRed:

    def test_y_pred_row_mismatch_raises(self):
        with pytest.raises(ValueError, match="mismatch"):
            EvaluationFrame(np.ones(5), np.ones((4, 1)), _make_identifiers(5))

    def test_y_pred_1d_raises(self):
        """1D y_pred must be rejected — callers must provide (N, S) shape."""
        n = 3
        with pytest.raises(ValueError, match="y_pred must be 2D"):
            EvaluationFrame(np.ones(n), np.ones(n), _make_identifiers(n))

    def test_y_pred_3d_raises(self):
        """3D y_pred must be rejected."""
        n = 2
        with pytest.raises(ValueError, match="y_pred must be 2D"):
            EvaluationFrame(np.ones(n), np.ones((n, 3, 2)), _make_identifiers(n))

    def test_nan_in_y_true_raises(self):
        n = 4
        with pytest.raises(ValueError, match="NaN"):
            EvaluationFrame(
                y_true=np.array([1.0, np.nan, 3.0, 4.0]),
                y_pred=np.ones((n, 1)),
                identifiers=_make_identifiers(n),
            )

    def test_nan_in_y_pred_raises(self):
        n = 4
        with pytest.raises(ValueError, match="NaN"):
            EvaluationFrame(
                y_true=np.ones(n),
                y_pred=np.array([[1.0], [np.nan], [3.0], [4.0]]),
                identifiers=_make_identifiers(n),
            )

    def test_inf_in_y_true_raises(self):
        n = 3
        with pytest.raises(ValueError, match="infinity"):
            EvaluationFrame(
                y_true=np.array([1.0, np.inf, 3.0]),
                y_pred=np.ones((n, 1)),
                identifiers=_make_identifiers(n),
            )

    def test_inf_in_y_pred_raises(self):
        n = 3
        with pytest.raises(ValueError, match="infinity"):
            EvaluationFrame(
                y_true=np.ones(n),
                y_pred=np.array([[1.0], [np.inf], [3.0]]),
                identifiers=_make_identifiers(n),
            )

    def test_nan_in_float_identifier_raises(self):
        n = 3
        ids = {
            'time':   np.array([np.nan, 101.0, 102.0]),
            'unit':   np.ones(n),
            'origin': np.zeros(n),
            'step':   np.array([1.0, 2.0, 3.0]),
        }
        with pytest.raises(ValueError, match="NaN"):
            EvaluationFrame(np.ones(n), np.ones((n, 1)), ids)

    def test_none_in_object_identifier_raises(self):
        n = 3
        ids = {
            'time':   np.array([100, None, 102], dtype=object),
            'unit':   np.array([1, 1, 1], dtype=object),
            'origin': np.array([0, 0, 0], dtype=object),
            'step':   np.array([1, 2, 3], dtype=object),
        }
        with pytest.raises(ValueError, match="None"):
            EvaluationFrame(np.ones(n), np.ones((n, 1)), ids)

    def test_identifier_length_mismatch_raises(self):
        n = 4
        ids = _make_identifiers(n)
        ids['step'] = np.array([1, 2])  # wrong length
        with pytest.raises(ValueError, match="length"):
            EvaluationFrame(np.ones(n), np.ones((n, 1)), ids)

    def test_missing_required_key_raises(self):
        n = 3
        ids = {
            'time': np.array([100, 101, 102]),
            'unit': np.ones(n, dtype=int),
            # 'origin' and 'step' missing
        }
        with pytest.raises(ValueError, match="origin"):
            EvaluationFrame(np.ones(n), np.ones((n, 1)), ids)

    def test_all_required_keys_listed_in_error(self):
        n = 2
        with pytest.raises(ValueError, match="Required"):
            EvaluationFrame(
                np.ones(n), np.ones((n, 1)),
                {'time': np.array([100, 101])},  # missing unit, origin, step
            )
