import numpy as np
import pytest

from views_evaluation.evaluation.evaluation_frame import EvaluationFrame
from views_evaluation.evaluation.evaluation_report import EvaluationReport
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


class TestOptionalExtraAbsentRed:
    """Register C-44. A missing optional extra must name itself; this module has no
    pandas import-skip, so the case runs wherever the suite runs."""

    def test_to_dataframe_without_pandas_names_the_extra(self, monkeypatch, caplog):
        """Contract, not mechanism: pandas genuinely unimportable (a `None` entry in
        sys.modules makes both `import pandas` and `find_spec("pandas")` report absence
        on CPython) → the deprecation warning is still emitted first, the raise names
        `views-evaluation[dataframe]`, keeps the type the bare import raised
        (`ModuleNotFoundError`, `.name == "pandas"`), the extra it names is one
        pyproject.toml declares, and nothing is logged (Level 0 does not log). Seen red
        on the bare import first: that raised `ModuleNotFoundError: import of pandas
        halted` with no extra named."""
        import logging
        import re
        import sys
        import tomllib
        import warnings
        from pathlib import Path

        monkeypatch.setitem(sys.modules, "pandas", None)
        report = EvaluationReport('t', 'regression', 'point', {
            'month': {'month100': {'MSE': 42.0}}, 'time_series': {}, 'step': {}})
        with caplog.at_level(logging.DEBUG):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                with pytest.raises(ModuleNotFoundError, match=r"views-evaluation\[dataframe\]") as exc:
                    report.to_dataframe('month')
        assert [x.category for x in w] == [DeprecationWarning], "warn first, then raise"
        assert exc.value.name == "pandas"
        assert not caplog.records, "Level 0 does not log (logging standard §5.1)"
        # The extra named in the message must exist, or the advice is unfollowable.
        extra = re.search(r"views-evaluation\[(\w+)\]", str(exc.value)).group(1)
        pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
        declared = pyproject.get("project", {}).get("optional-dependencies") or pyproject["tool"]["poetry"]["extras"]
        assert extra in declared, f"message names extra {extra!r}; declared: {sorted(declared)}"
