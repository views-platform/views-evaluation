import pandas as pd
import numpy as np
import pytest

from views_evaluation.evaluation.evaluation_manager import EvaluationManager
from views_evaluation.evaluation.evaluation_frame import EvaluationFrame
from views_evaluation.evaluation.native_evaluator import NativeEvaluator

@pytest.fixture
def adversarial_data_factory(mock_data_factory):
    """A fixture that extends the mock_data_factory to create adversarial data."""
    def _generate(
        target_name="lr_ged_sb_best",
        num_sequences=1,
        num_steps=1,
        num_locations=1,
        start_month=500,
        actuals_value=10.0,
        predictions_value=[[10.0]],
    ):
        pred_col_name = f"pred_{target_name}"
        loc_id_name = "country_id"

        # 1. Actuals DataFrame
        actuals_index = pd.MultiIndex.from_product(
            [range(start_month, start_month + num_steps), range(num_locations)],
            names=['month_id', loc_id_name]
        )
        actuals = pd.DataFrame(
            {target_name: actuals_value},
            index=actuals_index
        )

        # 2. Predictions List
        predictions_list = []
        preds_index = pd.MultiIndex.from_product(
            [range(start_month, start_month + num_steps), range(num_locations)],
            names=['month_id', loc_id_name]
        )
        preds = pd.DataFrame(
            {pred_col_name: predictions_value},
            index=preds_index
        )
        predictions_list.append(preds)

        # 3. Config
        config = {
            'steps': list(range(1, num_steps + 1)),
            'regression_targets': [target_name],
            'regression_point_metrics': ['RMSLE'],
        }

        return actuals, predictions_list, target_name, config

    return _generate


class TestAdversarialInputs:
    """
    A test suite for Phase 2: Adversarial and Edge-Case Testing.
    These tests probe for robustness and predictable failure modes.
    """

    def test_corrupted_numerical_data_nan_in_actuals(self, adversarial_data_factory):
        """
        Tests behavior when np.nan is present in the actuals data.
        Expected: A ValueError should be raised by the underlying sklearn metric.
        """
        # Arrange
        actuals, predictions, target, config = adversarial_data_factory(
            actuals_value=np.nan,
            predictions_value=[[10.0]]
        )
        manager = EvaluationManager()

        # Act & Assert
        with pytest.raises(ValueError, match="Input contains NaN"):
            manager.evaluate(
                actual=actuals,
                predictions=predictions,
                target=target,
                config=config
            )

    def test_corrupted_numerical_data_nan_in_predictions(self, adversarial_data_factory):
        """
        Tests behavior when np.nan is present in the predictions data.
        Expected: A ValueError should be raised by the underlying sklearn metric.
        """
        # Arrange
        actuals, predictions, target, config = adversarial_data_factory(
            actuals_value=10.0,
            predictions_value=[[np.nan]]
        )
        manager = EvaluationManager()

        # Act & Assert
        with pytest.raises(ValueError, match="Input contains NaN"):
            manager.evaluate(
                actual=actuals,
                predictions=predictions,
                target=target,
                config=config
            )

    def test_corrupted_numerical_data_inf_in_actuals(self, adversarial_data_factory):
        """
        Tests behavior when np.inf is present in the actuals data.
        Expected: A ValueError should be raised.
        """
        # Arrange
        actuals, predictions, target, config = adversarial_data_factory(
            actuals_value=np.inf,
            predictions_value=[[10.0]]
        )
        manager = EvaluationManager()

        # Act & Assert
        with pytest.raises(ValueError, match="Input contains infinity"):
            manager.evaluate(
                actual=actuals,
                predictions=predictions,
                target=target,
                config=config
            )

    def test_corrupted_numerical_data_inf_in_predictions(self, adversarial_data_factory):
        """
        Tests behavior when np.inf is present in the predictions data.
        Expected: A ValueError should be raised.
        """
        # Arrange
        actuals, predictions, target, config = adversarial_data_factory(
            actuals_value=10.0,
            predictions_value=[[np.inf]]
        )
        manager = EvaluationManager()

        # Act & Assert
        with pytest.raises(ValueError, match="Input contains infinity"):
            manager.evaluate(
                actual=actuals,
                predictions=predictions,
                target=target,
                config=config
            )

    def test_malformed_structural_data_empty_predictions_list(self, adversarial_data_factory):
        """
        Tests behavior when an empty list is passed for predictions.
        Expected: A ValueError should be raised by pandas.concat.
        """
        # Arrange
        actuals, _, target, config = adversarial_data_factory()
        empty_predictions = []
        manager = EvaluationManager()

        # Act & Assert
        with pytest.raises(ValueError, match="No objects to concatenate"):
            manager.evaluate(
                actual=actuals,
                predictions=empty_predictions,
                target=target,
                config=config
            )

    def test_malformed_structural_data_empty_actuals_df(self, adversarial_data_factory):
        """
        Tests behavior when an empty DataFrame is passed for actuals.
        Expected: A KeyError should be raised when trying to access the target column.
        """
        # Arrange
        _, predictions, target, config = adversarial_data_factory()
        empty_actuals = pd.DataFrame()
        manager = EvaluationManager()

        # Act & Assert
        with pytest.raises(KeyError):
            manager.evaluate(
                actual=empty_actuals,
                predictions=predictions,
                target=target,
                config=config
            )

    def test_malformed_structural_data_non_overlapping_indices(self, adversarial_data_factory):
        """
        Tests behavior when actuals and predictions have no overlapping indices.
        Expected: A ValueError should be raised by np.concatenate in the metric calculator.
        """
        # Arrange
        # Create actuals starting at month 500
        actuals, _, target, config = adversarial_data_factory(start_month=500, num_locations=1)
        
        # Create predictions starting at month 600, ensuring no overlap
        pred_col_name = f"pred_{target}"
        # Correctly create a 2-level MultiIndex
        preds_index = pd.MultiIndex.from_product(
            [range(600, 602), [10]], # Non-overlapping range for month_id
            names=['month_id', "country_id"]
        )
        preds = pd.DataFrame({pred_col_name: [[10.0]] * 2}, index=preds_index)
        predictions_non_overlapping = [preds]
        
        manager = EvaluationManager()

        # Act & Assert
        with pytest.raises(ValueError, match="need at least one array to concatenate"):
            manager.evaluate(
                actual=actuals,
                predictions=predictions_non_overlapping,
                target=target,
                config=config
            )


class TestAdversarialNativeInputs:
    """
    Adversarial tests targeting EvaluationFrame + NativeEvaluator directly.

    These tests are the native-path equivalents of TestAdversarialInputs above.
    They must survive Phase 3 of the orchestrator migration (ADR-011), when
    EvaluationManager and PandasAdapter are removed from this repository.

    Every test here asserts ADR-013 (Fail-Loud) behaviour from the entry
    points that will remain after the migration.
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

