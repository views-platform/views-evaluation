import numpy as np
import pytest

from views_evaluation.evaluation.evaluation_frame import EvaluationFrame
from views_evaluation.evaluation.native_evaluator import NativeEvaluator


class TestAdversarialNativeInputs:
    """
    Adversarial tests targeting EvaluationFrame + NativeEvaluator directly.

    Every test here asserts ADR-013 (Fail-Loud) behaviour from the
    permanent native-path entry points.
    """

    @staticmethod
    def _simple_ids(n: int) -> dict:
        return {
            'time':   np.arange(100, 100 + n),
            'unit':   np.ones(n, dtype=int),
            'origin': np.zeros(n, dtype=int),
            'step':   np.arange(1, n + 1),
        }

    def test_nan_in_y_true_rejected_at_frame_boundary(self):
        """ADR-013: NaN in actuals must be caught before any evaluation begins."""
        with pytest.raises(ValueError, match="NaN"):
            EvaluationFrame(
                y_true=np.array([np.nan, 1.0]),
                y_pred=np.array([[0.5], [0.8]]),
                identifiers=self._simple_ids(2),
            )

    def test_nan_in_y_pred_rejected_at_frame_boundary(self):
        """ADR-013: NaN in predictions must be caught before any evaluation begins."""
        with pytest.raises(ValueError, match="NaN"):
            EvaluationFrame(
                y_true=np.array([1.0, 2.0]),
                y_pred=np.array([[np.nan], [0.8]]),
                identifiers=self._simple_ids(2),
            )

    def test_inf_in_y_true_rejected_at_frame_boundary(self):
        """ADR-013: Infinity in actuals must be caught before any evaluation begins."""
        with pytest.raises(ValueError, match="infinity"):
            EvaluationFrame(
                y_true=np.array([np.inf, 1.0]),
                y_pred=np.array([[0.5], [0.8]]),
                identifiers=self._simple_ids(2),
            )

    def test_inf_in_y_pred_rejected_at_frame_boundary(self):
        """ADR-013: Infinity in predictions must be caught before any evaluation begins."""
        with pytest.raises(ValueError, match="infinity"):
            EvaluationFrame(
                y_true=np.array([1.0, 2.0]),
                y_pred=np.array([[np.inf], [0.8]]),
                identifiers=self._simple_ids(2),
            )

    def test_nan_in_identifier_rejected_at_frame_boundary(self):
        """ADR-012: NaN in identifier arrays is forbidden."""
        with pytest.raises(ValueError, match="NaN"):
            EvaluationFrame(
                y_true=np.array([1.0, 2.0]),
                y_pred=np.array([[0.5], [0.8]]),
                identifiers={
                    'time':   np.array([np.nan, 101.0]),
                    'unit':   np.array([1.0, 1.0]),
                    'origin': np.array([0.0, 0.0]),
                    'step':   np.array([1.0, 2.0]),
                },
            )

    def test_undeclared_target_fails_loudly_at_evaluator(self):
        """ADR-013: Target missing from config must raise at evaluate() time, not silently."""
        ef = EvaluationFrame(
            y_true=np.array([1.0, 2.0]),
            y_pred=np.array([[0.5], [0.8]]),
            identifiers=self._simple_ids(2),
            metadata={'target': 'undeclared_target'},
        )
        config = {
            'steps': [1, 2],
            'regression_targets': ['other_target'],
            'regression_point_metrics': ['MSE'],
        }
        with pytest.raises(ValueError, match="undeclared_target"):
            NativeEvaluator(config).evaluate(ef)

    def test_unimplemented_metric_raises_clear_value_error(self):
        """Unimplemented-but-declared metrics must raise ValueError, not NotImplementedError."""
        ef = EvaluationFrame(
            y_true=np.array([1.0, 2.0]),
            y_pred=np.array([[0.5], [0.8]]),
            identifiers=self._simple_ids(2),
            metadata={'target': 'test_target'},
        )
        config = {
            'steps': [1, 2],
            'regression_targets': ['test_target'],
            'regression_point_metrics': ['SD'],
        }
        with pytest.raises(ValueError, match="not yet implemented"):
            NativeEvaluator(config).evaluate(ef)

    def test_nan_rejected_before_brier_executes(self):
        """Defense-in-depth: EvaluationFrame rejects NaN so Brier's NaN-swallowing
        comparison semantics can never be triggered through the normal evaluation path."""
        with pytest.raises(ValueError, match="NaN"):
            EvaluationFrame(
                y_true=np.array([np.nan, 1.0]),
                y_pred=np.array([[0.5], [0.8]]),
                identifiers=self._simple_ids(2),
                metadata={'target': 'cls_target'},
            )

    def test_inf_rejected_before_metric_executes(self):
        """Defense-in-depth: EvaluationFrame rejects Inf before any metric function runs."""
        with pytest.raises(ValueError, match="infinity"):
            EvaluationFrame(
                y_true=np.array([np.inf, 1.0]),
                y_pred=np.array([[0.5], [0.8]]),
                identifiers=self._simple_ids(2),
                metadata={'target': 'cls_target'},
            )
