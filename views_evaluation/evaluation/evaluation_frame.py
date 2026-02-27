import numpy as np
from typing import Dict, Any, Optional

class EvaluationFrame:
    """
    The canonical internal representation for Evaluation.
    
    This class is framework-agnostic and uses only NumPy arrays.
    It enforces shape consistency and provides minimal methods for 
    filtering and grouping required by the evaluation schemas.
    """
    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        identifiers: Dict[str, np.ndarray],
        metadata: Optional[Dict[str, Any]] = None
    ):
        self._validate(y_true, y_pred, identifiers)
        self.y_true = y_true
        self.y_pred = y_pred
        self.identifiers = identifiers
        self.metadata = metadata or {}

    @staticmethod
    def _validate(y_true: np.ndarray, y_pred: np.ndarray, identifiers: Dict[str, np.ndarray]):
        n_rows = len(y_true)
        if y_pred.shape[0] != n_rows:
            raise ValueError(f"y_pred rows ({y_pred.shape[0]}) mismatch y_true ({n_rows})")
        
        # ADR-013: Fail-Loud on corrupted numerical data
        # Align with legacy test expectations for error messages
        def check_corrupted(arr, name):
            if arr.dtype.kind in 'fc': # float or complex
                if np.any(np.isnan(arr)):
                    raise ValueError("Input contains NaN")
                if np.any(np.isinf(arr)):
                    raise ValueError("Input contains infinity")
            elif arr.dtype.kind == 'O': # object
                # Attempt to convert to float for checking if possible
                try:
                    float_arr = arr.astype(float)
                    if np.any(np.isnan(float_arr)):
                        raise ValueError("Input contains NaN")
                    if np.any(np.isinf(float_arr)):
                        raise ValueError("Input contains infinity")
                except (ValueError, TypeError):
                    pass # Not numeric data

        check_corrupted(y_true, "y_true")
        check_corrupted(y_pred, "y_pred")

        # ADR-014: Enforce required identifier keys
        required_keys = {'time', 'unit', 'origin', 'step'}
        missing_keys = required_keys - set(identifiers.keys())
        if missing_keys:
            raise ValueError(
                f"EvaluationFrame identifiers missing required keys: {sorted(missing_keys)}. "
                f"Required: {sorted(required_keys)}"
            )

        for key, arr in identifiers.items():
            if len(arr) != n_rows:
                raise ValueError(f"Identifier '{key}' length ({len(arr)}) mismatch y_true ({n_rows})")
            
            # ADR-012: No NaNs in identifiers
            if arr.dtype.kind in 'f' and np.any(np.isnan(arr)):
                raise ValueError(f"Identifier '{key}' contains NaN values. This is forbidden.")
            if arr.dtype.kind in 'O' and np.any(np.equal(arr, None)):
                raise ValueError(f"Identifier '{key}' contains None values. This is forbidden.")

    @property
    def n_rows(self) -> int:
        return len(self.y_true)

    @property
    def n_samples(self) -> int:
        return self.y_pred.shape[1]

    @property
    def is_sample(self) -> bool:
        return self.n_samples > 1

    def get_group_indices(self, identifier_key: str) -> Dict[Any, np.ndarray]:
        """
        Returns a mapping of unique values in the identifier to the indices 
        where they occur. Equivalent to a groupby.
        Optimized using argsort and split.
        """
        arr = self.identifiers[identifier_key]
        
        # Sort indices by value
        idx_sort = np.argsort(arr, kind='stable')
        sorted_arr = arr[idx_sort]
        
        # Find unique values and their first occurrences in the sorted array
        unique_vals, first_indices = np.unique(sorted_arr, return_index=True)
        
        # Split the sorted indices into groups
        # np.split expects indices of where to split (excluding the first 0)
        groups = np.split(idx_sort, first_indices[1:])
        
        return dict(zip(unique_vals, groups))


    def select_indices(self, indices: np.ndarray) -> 'EvaluationFrame':
        """Create a new EvaluationFrame from a subset of indices."""
        return EvaluationFrame(
            y_true=self.y_true[indices],
            y_pred=self.y_pred[indices],
            identifiers={k: v[indices] for k, v in self.identifiers.items()},
            metadata=self.metadata.copy()
        )

    def __repr__(self):
        return (f"EvaluationFrame(n_rows={self.n_rows}, "
                f"n_samples={self.n_samples}, "
                f"ids={list(self.identifiers.keys())})")
