import pandas as pd
import numpy as np
import pytest

# A fixture to generate mock data for tests
@pytest.fixture
def mock_data_factory():
    def _generate(
        target_name="lr_ged_sb_best",
        point_predictions_as_list=True,
        num_sequences=2,
        num_steps=3,
        num_locations=2,
        start_month=500,
    ):
        pred_col_name = f"pred_{target_name}"
        loc_id_name = "country_id"

        # 1. Actuals DataFrame
        actuals_index = pd.MultiIndex.from_product(
            [range(start_month, start_month + num_sequences + num_steps), range(num_locations)],
            names=['month_id', loc_id_name]
        )
        actuals = pd.DataFrame(
            {target_name: np.random.randint(0, 50, size=len(actuals_index))},
            index=actuals_index
        )

        # 2. Predictions List
        predictions_list = []
        for i in range(num_sequences):
            preds_index = pd.MultiIndex.from_product(
                [range(start_month + i, start_month + i + num_steps), range(num_locations)],
                names=['month_id', loc_id_name]
            )
            
            if point_predictions_as_list:
                # Canonical format: list of single floats
                pred_values = [[val] for val in np.random.rand(len(preds_index)) * 50]
            else:
                # Non-canonical format: raw floats
                pred_values = [val for val in np.random.rand(len(preds_index)) * 50]

            preds = pd.DataFrame(
                {pred_col_name: pred_values},
                index=preds_index
            )
            predictions_list.append(preds)

        # 3. Config
        config = {'steps': list(range(1, num_steps + 1))}

        return actuals, predictions_list, target_name, config

    return _generate
