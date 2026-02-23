import pandas as pd
import numpy as np
import pytest

from views_evaluation.evaluation.evaluation_manager import EvaluationManager

class TestMetricCorrectness:
    """
    A test suite for Phase 3: Data-Centric & Metric-Specific Validation.
    These tests verify the numerical correctness of the metric calculators
    using 'golden datasets' with pre-calculated, known outcomes.
    """

    def test_rmsle_golden_dataset_perfect_match(self):
        """
        Tests the RMSLE calculation with a perfect match.
        Expected: RMSLE should be 0.0.
        """
        # Arrange
        target_name = "lr_test"
        pred_col_name = f"pred_{target_name}"

        # Create a simple, non-random dataset
        actuals_index = pd.MultiIndex.from_product([[500], [10, 20]], names=['month_id', 'country_id'])
        actuals = pd.DataFrame({target_name: [100, 50]}, index=actuals_index)

        # Predictions are identical to actuals
        predictions_df = pd.DataFrame({pred_col_name: [[100.0], [50.0]]}, index=actuals_index)
        predictions = [predictions_df]

        config = {
            'steps': [1],
            'regression_targets': [target_name],
            'regression_point_metrics': ['RMSLE'],
        }
        manager = EvaluationManager()

        # Act
        results = manager.evaluate(
            actual=actuals,
            predictions=predictions,
            target=target_name,
            config=config
        )

        # Assert
        # Check all evaluation schemas for correctness
        rmsle_step = results['step'][1]['RMSLE'].iloc[0]
        rmsle_ts = results['time_series'][1]['RMSLE'].iloc[0]
        rmsle_month = results['month'][1]['RMSLE'].iloc[0]

        assert rmsle_step == 0.0
        assert rmsle_ts == 0.0
        assert rmsle_month == 0.0

    def test_rmsle_golden_dataset_simple_mismatch(self):
        """
        Tests the RMSLE calculation with a simple, known mismatch.
        actual = e - 1, pred = 0.
        log(actual + 1) = log(e) = 1.
        log(pred + 1) = log(1) = 0.
        RMSLE = sqrt((1-0)^2) = 1.
        Expected: RMSLE should be 1.0.
        """
        # Arrange
        target_name = "lr_test"
        pred_col_name = f"pred_{target_name}"

        actual_val = np.e - 1
        pred_val = 0.0

        actuals_index = pd.MultiIndex.from_product([[500], [10]], names=['month_id', 'country_id'])
        actuals = pd.DataFrame({target_name: [actual_val]}, index=actuals_index)

        predictions_df = pd.DataFrame({pred_col_name: [[pred_val]]}, index=actuals_index)
        predictions = [predictions_df]

        config = {
            'steps': [1],
            'regression_targets': [target_name],
            'regression_point_metrics': ['RMSLE'],
        }
        manager = EvaluationManager()

        # Act
        results = manager.evaluate(
            actual=actuals,
            predictions=predictions,
            target=target_name,
            config=config
        )

        # Assert
        rmsle_step = results['step'][1]['RMSLE'].iloc[0]

        assert rmsle_step == pytest.approx(1.0)

    def test_ap_metric_with_prebinarised_inputs(self):
        """
        Tests the AP (Average Precision) metric with pre-binarised actuals and probability
        scores as predictions.  AP is a classification metric; actuals must already be
        binary (0/1) before reaching evaluate().  No threshold kwarg is accepted.
        """
        # Arrange
        target_name = "cls_binary"
        pred_col_name = f"pred_{target_name}"

        # Pre-binarised actuals and probability scores
        y_true_binary = [0, 1, 1, 0]
        y_scores = [0.1, 0.4, 0.35, 0.8]

        actuals_index = pd.MultiIndex.from_product(
            [[500], [10, 20, 30, 40]], names=['month_id', 'country_id']
        )
        actuals = pd.DataFrame({target_name: y_true_binary}, index=actuals_index)
        predictions_df = pd.DataFrame(
            {pred_col_name: [[s] for s in y_scores]}, index=actuals_index
        )
        predictions = [predictions_df]

        config = {
            'steps': [1],
            'classification_targets': [target_name],
            'classification_point_metrics': ['AP'],
        }
        manager = EvaluationManager()

        # Act
        results = manager.evaluate(
            actual=actuals,
            predictions=predictions,
            target=target_name,
            config=config
        )

        ap_step = results['step'][1]['AP'].iloc[0]

        # Expected AP from sklearn with the raw probability scores as the ranking signal
        from sklearn.metrics import average_precision_score
        expected_ap = average_precision_score(y_true_binary, y_scores)

        assert ap_step == pytest.approx(expected_ap)

    def test_crps_golden_dataset_point_prediction(self):
        """
        Tests the CRPS calculation for point predictions (single-value ensemble).
        Expected: CRPS matches properscoring for a 1-sample ensemble.
        """
        # Arrange
        target_name = "lr_test_crps_point"
        pred_col_name = f"pred_{target_name}"

        # Simple dataset: one actual, one prediction
        actual_val = 5.0
        pred_val = 6.0

        actuals_index = pd.MultiIndex.from_product([[500], [10]], names=['month_id', 'country_id'])
        actuals = pd.DataFrame({target_name: [actual_val]}, index=actuals_index)

        # Single-value prediction → point prediction, use regression_uncertainty_metrics
        # by providing a multi-element ensemble so it's detected as uncertainty type.
        # Use the same scalar as a 3-sample degenerate ensemble for CRPS:
        predictions_df = pd.DataFrame({pred_col_name: [[pred_val, pred_val, pred_val]]}, index=actuals_index)
        predictions = [predictions_df]

        config = {
            'steps': [1],
            'regression_targets': [target_name],
            'regression_point_metrics': ['RMSLE'],       # required by _validate_config
            'regression_uncertainty_metrics': ['CRPS'],  # routed to because predictions are multi-element
        }
        manager = EvaluationManager()

        # Act
        results = manager.evaluate(
            actual=actuals,
            predictions=predictions,
            target=target_name,
            config=config
        )

        # Assert
        crps_step = results['step'][1]['CRPS'].iloc[0]

        # Calculate expected CRPS using properscoring for the degenerate 3-sample ensemble
        import properscoring as ps
        expected_crps = ps.crps_ensemble(actual_val, np.array([pred_val, pred_val, pred_val]))

        assert crps_step == pytest.approx(expected_crps)

    def test_crps_golden_dataset_uncertainty_prediction(self):
        """
        Tests the CRPS calculation for uncertainty predictions (ensemble of multiple values).
        Expected: CRPS for uncertainty predictions matches properscoring.
        """
        # Arrange
        target_name = "lr_test_crps_uncertainty"
        pred_col_name = f"pred_{target_name}"

        # Simple dataset: one actual, one prediction ensemble
        actual_val = 5.0
        prediction_ensemble = [3.0, 4.0, 5.0, 6.0, 7.0]  # A simple ensemble

        actuals_index = pd.MultiIndex.from_product([[500], [10]], names=['month_id', 'country_id'])
        actuals = pd.DataFrame({target_name: [actual_val]}, index=actuals_index)

        # Uncertainty prediction is a list of multiple values
        predictions_df = pd.DataFrame({pred_col_name: [prediction_ensemble]}, index=actuals_index)
        predictions = [predictions_df]

        config = {
            'steps': [1],
            'regression_targets': [target_name],
            'regression_point_metrics': ['RMSLE'],       # required by _validate_config
            'regression_uncertainty_metrics': ['CRPS'],  # routed to because predictions are multi-element
        }
        manager = EvaluationManager()

        # Act
        results = manager.evaluate(
            actual=actuals,
            predictions=predictions,
            target=target_name,
            config=config
        )

        # Assert
        crps_step = results['step'][1]['CRPS'].iloc[0]

        # Calculate expected CRPS using properscoring for the ensemble
        import properscoring as ps
        expected_crps = ps.crps_ensemble(actual_val, np.array(prediction_ensemble))

        assert crps_step == pytest.approx(expected_crps)
