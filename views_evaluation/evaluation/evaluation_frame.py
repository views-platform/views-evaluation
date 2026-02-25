import numpy as np
from typing import Dict, Any, Optional, List

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
        
        for key, arr in identifiers.items():
            if len(arr) != n_rows:
                raise ValueError(f"Identifier '{key}' length ({len(arr)}) mismatch y_true ({n_rows})")

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
        """
        arr = self.identifiers[identifier_key]
        unique_vals = np.unique(arr)
        # Use np.where for simplicity, though searchsorted or argpartition 
        # might be faster for sorted data.
        return {val: np.where(arr == val)[0] for val in unique_vals}

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
