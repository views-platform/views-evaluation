"""
PHASE-3-DELETE: Tests documentation contracts for the legacy EvaluationManager path.
Will be deleted when Phase 3 of the orchestrator migration is complete.
See reports/2026-02-25_evaluation_frame_refactor/10_orchestrator_migration_plan.md
"""
import pandas as pd
import numpy as np
import pytest

from views_evaluation.evaluation.evaluation_manager import EvaluationManager


class TestDocumentationContracts:
    """
    A test suite to verify the claims made in the project's documentation.
    """

    def test_eval_lib_imp_actuals_schema_prefix_requirement_succeeds(self, mock_data_factory):
        """
        Verifies Section 3.1 of eval_lib_imp.md.
        Claim: Evaluation succeeds if the target name has a valid prefix.
        """
        # Arrange
        target_with_prefix = "lr_ged_sb_best"
        actuals, predictions, target, config = mock_data_factory(target_name=target_with_prefix)
        manager = EvaluationManager()

        # Act & Assert
        try:
            manager.evaluate(
                actual=actuals,
                predictions=predictions,
                target=target,
                config=config
            )
        except ValueError as e:
            pytest.fail(f"Evaluation failed unexpectedly with a valid prefix: {e}")

    def test_eval_lib_imp_actuals_schema_prefix_requirement_fails(self, mock_data_factory):
        """
        Verifies updated behaviour from Section 3.1 of eval_lib_imp.md.
        Old claim: Evaluation fails if the target name is missing a valid prefix.
        New behaviour: The new EvaluationManager no longer validates prefixes in evaluate().
        transform_data() issues a warning for unknown prefixes but applies an identity
        transform and continues. Evaluation therefore *succeeds* with an unknown prefix as
        long as the target is declared in the config.
        """
        # Arrange
        target_without_prefix = "ged_sb_best"
        actuals, predictions, target, config = mock_data_factory(target_name=target_without_prefix)
        manager = EvaluationManager()

        # Act & Assert — should now succeed (prefix validation removed from evaluate())
        try:
            manager.evaluate(
                actual=actuals,
                predictions=predictions,
                target=target,
                config=config
            )
        except Exception as e:
            pytest.fail(
                f"evaluate() raised unexpectedly for a target with no recognised prefix: {e}"
            )

    def test_eval_lib_imp_predictions_schema_point_canonical_succeeds(self, mock_data_factory):
        """
        Verifies Section 3.2 of eval_lib_imp.md.
        Claim: Evaluation succeeds if point predictions are canonical (list of single float).
        """
        # Arrange
        actuals, predictions, target, config = mock_data_factory(point_predictions_as_list=True)
        manager = EvaluationManager()

        # Act & Assert
        try:
            manager.evaluate(
                actual=actuals,
                predictions=predictions,
                target=target,
                config=config
            )
        except ValueError as e:
            pytest.fail(f"Evaluation failed unexpectedly with canonical point predictions: {e}")

    def test_eval_lib_imp_predictions_schema_point_non_canonical_succeeds_due_to_implicit_conversion(self, mock_data_factory):
        """
        Verifies Section 3.2 of eval_lib_imp.md by demonstrating a divergence.
        Claim: Documentation states evaluation fails if point predictions are non-canonical (raw float).
        Observed: Evaluation *succeeds* due to implicit conversion in EvaluationManager, making documentation inaccurate.
        """
        # Arrange
        actuals, predictions, target, config = mock_data_factory(point_predictions_as_list=False)
        manager = EvaluationManager()

        # Act & Assert
        try:
            manager.evaluate(
                actual=actuals,
                predictions=predictions,
                target=target,
                config=config
            )
        except ValueError as e:
            pytest.fail(f"Evaluation *should have succeeded* with non-canonical point predictions due to implicit conversion, but failed with: {e}")

    def test_evaluation_manager_implicitly_converts_raw_floats_to_arrays(self, mock_data_factory):
        """
        Explicitly verifies the implicit conversion of raw float predictions to np.ndarray([float])
        by EvaluationManager's internal _process_data method.
        This behavior contradicts eval_lib_imp.md's claim that raw floats should cause an error.
        """
        # Arrange
        actuals, predictions, target, config = mock_data_factory(point_predictions_as_list=False)
        manager = EvaluationManager()

        # Act
        manager.evaluate(
            actual=actuals,
            predictions=predictions,
            target=target,
            config=config
        )

        # Assert
        # After evaluate, internal predictions should be processed
        processed_predictions = manager.predictions
        # Check that the first value in the first DataFrame of processed_predictions is now a np.ndarray
        assert isinstance(processed_predictions[0].iloc[0, 0], np.ndarray)
        # Check that its length is 1 (single element)
        assert len(processed_predictions[0].iloc[0, 0]) == 1

    def test_eval_lib_imp_api_contract_missing_steps_config_fails(self, mock_data_factory):
        """
        Verifies Section 4.2 of eval_lib_imp.md.
        Claim: The `evaluate` method's `config` parameter *must* contain the key 'steps'.
        """
        # Arrange
        actuals, predictions, target, _ = mock_data_factory() # Use _ to ignore the default config
        manager = EvaluationManager()
        invalid_config = {} # Missing 'steps' key

        # Act & Assert
        with pytest.raises(KeyError, match="'steps'"):
            manager.evaluate(
                actual=actuals,
                predictions=predictions,
                target=target,
                config=invalid_config
            )

    def test_eval_lib_imp_data_state_coherency_no_inverse_transform(self, mock_data_factory):
        """
        Verifies Section 3.4 of eval_lib_imp.md.
        Claim: EvaluationManager does NOT perform inverse transformations on prediction data (producer's responsibility).
        """
        # Arrange
        target_name = "lr_some_var" # lr_ prefix means raw, no transform by EM
        pred_col_name = f"pred_{target_name}"
        loc_id_name = "country_id"

        # Create actuals (raw counts)
        actuals_index = pd.MultiIndex.from_product([[500], [10]], names=['month_id', loc_id_name])
        actuals = pd.DataFrame(
            {target_name: [100]}, # Actual value is 100
            index=actuals_index
        )

        # Create predictions that are log-transformed, but named as 'lr_' to indicate raw input
        # So, if EM were to inverse transform, it would be wrong, but it shouldn't inverse transform
        predictions_list = []
        preds_index = pd.MultiIndex.from_product([[500], [10]], names=['month_id', loc_id_name])
        # Prediction of log(100+1) - 1, which is approximately 4.6 (ln(101)-1)
        # If EM doesn't inverse transform, RMSLE will be calculated with 4.6 vs 100
        # If EM incorrectly inverse transformed, it would see 4.6, transform it back, then calculate RMSLE
        pred_values_log_transformed = [[np.log1p(100)]] # Represents log(100+1)
        predictions_df = pd.DataFrame(
            {pred_col_name: pred_values_log_transformed},
            index=preds_index
        )
        predictions_list.append(predictions_df)
        
        manager = EvaluationManager()
        
        # We need a config with steps and the new required keys
        config = {
            'steps': [1],
            'regression_targets': [target_name],
            'regression_point_metrics': ['RMSLE'],
        }

        # Act
        results = manager.evaluate(
            actual=actuals,
            predictions=predictions_list,
            target=target_name,
            config=config
        )

        # Assert
        # Get the RMSLE for the step-wise evaluation
        rmsle = results['step'][1]['RMSLE'][0] # Access the DataFrame, then RMSLE column, then first value
        
        # If EM incorrectly inverse-transformed, RMSLE would be close to 0
        # If EM correctly *doesn't* inverse-transform, RMSLE is calculated with actual=100 and pred=log1p(100)
        # log1p(100) is approx 4.615
        # RMSLE(100, 4.615) is large.
        
        # A simple check: if RMSLE is very small, it means inverse transform *did* happen.
        # We expect it to be large.
        assert rmsle > 1.0 # Arbitrary large threshold to show it's not a small error

    def test_r2darts2_report_point_prediction_format_succeeds(self, mock_data_factory):
        """
        Verifies Section B.1 of the plan (from r2darts2_full_imp_report.md).
        Claim: views-r2darts2 produces point predictions as a list (e.g., [[25.5]]).
        """
        # Arrange
        # Use mock_data_factory with point_predictions_as_list=True to simulate r2darts2 output
        actuals, predictions, target, config = mock_data_factory(point_predictions_as_list=True)
        manager = EvaluationManager()

        # Act & Assert
        try:
            manager.evaluate(
                actual=actuals,
                predictions=predictions,
                target=target,
                config=config
            )
        except ValueError as e:
            pytest.fail(f"Evaluation failed unexpectedly when processing r2darts2-like canonical point predictions: {e}")

    def test_stepshifter_report_point_prediction_format_succeeds_despite_raw_float_output(self, mock_data_factory):
        """
        Verifies Section C.1 of the plan (from stepshifter_full_imp_report.md).
        Claim: views-stepshifter produces point predictions as raw np.float64 values (contradicts eval_lib_imp.md).
        Observed: EvaluationManager implicitly converts and processes successfully.
        """
        # Arrange
        # Use mock_data_factory with point_predictions_as_list=False to simulate stepshifter output
        actuals, predictions, target, config = mock_data_factory(point_predictions_as_list=False)
        manager = EvaluationManager()

        # Act & Assert
        try:
            manager.evaluate(
                actual=actuals,
                predictions=predictions,
                target=target,
                config=config
            )
        except ValueError as e:
            pytest.fail(f"Evaluation *should have succeeded* with stepshifter-like raw float predictions due to implicit conversion, but failed with: {e}")

    def test_stepshifter_report_reconciliation_fix_succeeds(self, mock_data_factory):
        """
        Verifies Section C.2 of the plan (from stepshifter_full_imp_report.md).
        Claim: Applying the reconciliation fix (float -> list) to stepshifter's raw float output
               should allow EvaluationManager to process the data successfully.
        """
        # Arrange
        # Simulate stepshifter output (raw floats)
        actuals, predictions_raw_floats, target, config = mock_data_factory(point_predictions_as_list=False)
        manager = EvaluationManager()

        # Apply the reconciliation logic as described in the report
        # "Wrap every cell value in a list to conform to the canonical standard."
        reconciled_predictions = [df.applymap(lambda x: [x]) for df in predictions_raw_floats]

        # Act & Assert
        try:
            manager.evaluate(
                actual=actuals,
                predictions=reconciled_predictions,
                target=target,
                config=config
            )
        except ValueError as e:
            pytest.fail(f"Evaluation failed unexpectedly after applying stepshifter's reconciliation fix: {e}")



        


