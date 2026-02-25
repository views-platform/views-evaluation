import numpy as np
import pandas as pd
from typing import List, Union
from views_evaluation.evaluation.evaluation_frame import EvaluationFrame

class PandasAdapter:
    """
    Adapter to convert Pandas DataFrames into the native EvaluationFrame.
    
    This class 'knows' about Pandas, allowing the rest of the core
    to remain pure.
    """
    
    @staticmethod
    def from_dataframes(
        actual: pd.DataFrame,
        predictions: List[pd.DataFrame],
        target: str,
    ) -> EvaluationFrame:
        """
        Convert the current List[DataFrame] structure into a single EvaluationFrame.
        
        Args:
            actual: DataFrame with MultiIndex [time, unit]
            predictions: List of DataFrames with MultiIndex [time, unit]
            target: The name of the target column
        """
        
        all_y_true = []
        all_y_pred = []
        all_times = []
        all_units = []
        all_origins = []
        all_steps = []
        
        pred_col = f"pred_{target}"
        
        for i, df in enumerate(predictions):
            # 1. Align/Match Actuals (duplicated logic from EvaluationManager)
            common_idx = actual.index.intersection(df.index)
            matched_pred = df.loc[common_idx]
            matched_actual = actual.loc[common_idx, target]
            
            # 2. Extract Data
            # Note: We assume all cells have the same number of samples
            # This is where we explode the 'list-in-cell'
            samples = np.array(matched_pred[pred_col].tolist())
            if samples.ndim == 1: # Point forecasts
                samples = samples.reshape(-1, 1)
            
            n_rows = len(matched_actual)
            
            all_y_true.append(matched_actual.values)
            all_y_pred.append(samples)
            
            # 3. Extract Identifiers
            all_times.append(matched_pred.index.get_level_values(0).values)
            all_units.append(matched_pred.index.get_level_values(1).values)
            
            # 4. Synthesize Origin and Step
            # Origin is the list index
            all_origins.append(np.full(n_rows, i))
            
            # Step is positional lead-time per unique month in the sequence
            unique_times = matched_pred.index.get_level_values(0).unique()
            time_to_step = {t: step_idx + 1 for step_idx, t in enumerate(unique_times)}
            steps = np.array([time_to_step[t] for t in matched_pred.index.get_level_values(0)])
            all_steps.append(steps)
            
        return EvaluationFrame(
            y_true=np.concatenate(all_y_true),
            y_pred=np.concatenate(all_y_pred),
            identifiers={
                'time': np.concatenate(all_times),
                'unit': np.concatenate(all_units),
                'origin': np.concatenate(all_origins),
                'step': np.concatenate(all_steps),
            },
            metadata={'target': target}
        )
