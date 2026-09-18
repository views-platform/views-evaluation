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


class TestOptionalExtraAbsentRed:
    """A missing optional extra must name itself, with the extra pyproject actually
    declares, in the exception type the CIC documents, logged on the emit path. Until
    2.0.0 this class pinned `to_dataframe()`'s pandas branch (C-44); that method is gone,
    and this is the same shape transplanted to the surviving bridge. Seen red on three
    injections before it was trusted: extra renamed to `[framez]`, `ImportError` swapped
    for `RuntimeError`, and the `.error()` call deleted — and a fourth, the logger renamed, found by the guard audit."""

    def test_to_metric_frame_without_views_frames_names_the_extra(self, monkeypatch, caplog):
        import logging
        import re
        import sys
        import tomllib
        from pathlib import Path
        from views_evaluation.evaluation.evaluation_report import EvaluationReport

        # A `None` entry makes both `import views_frames` and `find_spec` report absence.
        monkeypatch.setitem(sys.modules, "views_frames", None)
        report = EvaluationReport('t', 'regression', 'point', {
            'month': {'month100': {'MSE': 42.0}}, 'time_series': {}, 'step': {}})
        with caplog.at_level(logging.ERROR, logger="views_evaluation.evaluation.metric_frame"):
            with pytest.raises(ImportError, match=r"views-evaluation\[frames\]") as exc:
                report.to_metric_frame()
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert errors and errors[0].message == str(exc.value), "emit path logs the message it raises (§5.1)"
        assert errors[0].name == "views_evaluation.evaluation.metric_frame", (
            "the emit path logs under the metric_frame logger, not the file's (§5.1)"
        )
        # The extra named in the message must exist, or the advice is unfollowable.
        extra = re.search(r"views-evaluation\[(\w+)\]", str(exc.value)).group(1)
        pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
        declared = pyproject.get("project", {}).get("optional-dependencies") or pyproject["tool"]["poetry"]["extras"]
        assert extra in declared, f"message names extra {extra!r}; declared: {sorted(declared)}"
