"""FAILING TESTS — falsification audit, 2026-08-02.

Claim under audit: "development is ready to merge to main and be released as 0.5.0."

**FALSIFIED.** ADR-015 ruling 7 (#37) makes `legacy_compatibility=True` raise when
`config['steps']` requests a step beyond the shortest origin sequence. That is precisely
the situation the flag exists to handle, and `views-pipeline-core` calls it that way
unconditionally:

    views_pipeline_core/managers/evaluation/stage.py:149-151

        # legacy_compatibility=True preserves step-wise truncation to shortest
        # sequence, matching the deleted EvaluationManager wrapper (C-29).
        report = evaluator.evaluate(ef=ef, legacy_compatibility=True)

Every views-models config sets ``steps = list(range(1, 37))``, so whenever sequences are
unequal the config asks for 36 steps while the shortest supplies fewer — and the call
raised.

**How often that actually happens** (derived from the real partition arithmetic, not
assumed — an earlier version of this docstring claimed "every model in the platform,
deterministically", which was wrong):

``core_config_sniffer`` enforces ``test_len == time_steps + MAX_SHIFT_COUNT`` = 48 months,
and ``MAX_SHIFT_COUNT = 12`` gives 13 sequences. Sequence 12 ends exactly on the last
actual, so all 13 are full length.

    eval_type=standard (default) / live -> 13 sequences, equal   -> 0 of 256 truncate
    eval_type=long                      -> 37 sequences, unequal -> 256 of 256 truncate

``eval_type=long`` is used nowhere outside pipeline-core's own arg-parser tests, so the
raise was **latent**. These tests still matter: they pin pipeline-core's real invocation,
which nothing else in this repo does, and the ruling was reversed for a reason that does
not depend on frequency — ``legacy_compatibility=True`` is itself a request to truncate,
so raising blames the caller for doing exactly what they asked for.

These tests are expected to FAIL until ruling 7 is revised. They encode the behaviour
pipeline-core needs, not the behaviour currently shipped. Do not "fix" them by relaxing
the assertions.

See also `tests/test_native_evaluator.py::TestNativeEvaluatorGreen::
test_legacy_compatibility_true_truncates_to_shortest_sequence`, which currently asserts
the raise — it will need to change in step with whatever ruling 7 becomes.
"""
import numpy as np
import pytest

from views_evaluation.evaluation.evaluation_frame import EvaluationFrame
from views_evaluation.evaluation.native_evaluator import NativeEvaluator


def _rolling_origin_ef(n_origins=12, full_steps=36, truncate_from=6):
    """A realistic rolling-origin parallelogram with unequal sequence lengths.

    Mirrors what pipeline-core's EvaluationAdapter produces at the data edge: early
    origins have their full forecast window matched against actuals, later origins are
    progressively short because the actuals run out.
    """
    rows, y_true, y_pred = [], [], []
    for origin in range(n_origins):
        n_steps = full_steps if origin < truncate_from else full_steps - (origin - truncate_from + 1) * 2
        for step in range(1, n_steps + 1):
            for unit in (1, 2):
                rows.append((500 + origin + step - 1, unit, origin, step))
                y_true.append(float(step))
                y_pred.append([float(step) + 0.1])
    return EvaluationFrame(
        y_true=np.array(y_true),
        y_pred=np.array(y_pred),
        identifiers={
            'time':   np.array([r[0] for r in rows]),
            'unit':   np.array([r[1] for r in rows]),
            'origin': np.array([r[2] for r in rows]),
            'step':   np.array([r[3] for r in rows]),
        },
        metadata={'target': 'lr_ged_os'},
    )


def _real_model_config():
    """Keys and values taken verbatim from views-models/models/ravaging_thief."""
    return {
        'steps': list(range(1, 37)),
        'regression_targets': ['lr_ged_os'],
        'regression_point_metrics': ['MSLE', 'MSE', 'MCR_point', 'y_hat_bar'],
    }


class TestLegacyCompatibilityMustNotBreakPipelineCore:

    def test_pipeline_core_call_path_evaluates(self):
        """THE falsifying probe: pipeline-core's exact call must not raise.

        `evaluator.evaluate(ef=ef, legacy_compatibility=True)` with a full 1..36 step
        config and unequal origin sequences is the normal case in production, not an
        edge case.
        """
        ef = _rolling_origin_ef()
        report = NativeEvaluator(_real_model_config()).evaluate(
            ef=ef, legacy_compatibility=True
        )
        assert report is not None

    def test_truncation_still_truncates(self):
        """The flag's stated purpose must survive: cap step-wise at the shortest sequence.

        pipeline-core's own comment: "preserves step-wise truncation to shortest
        sequence, matching the deleted EvaluationManager wrapper".
        """
        ef = _rolling_origin_ef()
        shortest = min(
            len(np.unique(ef.identifiers['step'][idx]))
            for idx in ef.get_group_indices('origin').values()
        )
        report = NativeEvaluator(_real_model_config()).evaluate(
            ef=ef, legacy_compatibility=True
        )
        steps = report.to_dict()['schemas']['step']
        scored = {k for k, v in steps.items() if v}
        assert scored, "no step groups were scored at all"
        assert all(int(k.replace('step', '')) <= shortest for k in scored), (
            f"steps beyond the shortest sequence ({shortest}) were scored"
        )

    def test_truncated_steps_are_not_empty_placeholders(self):
        """The C-20 defect must stay fixed however ruling 7 is revised.

        A step beyond the shortest sequence must NOT come back as an empty dict that
        looks evaluated and emits no MetricFrame rows. Omit the key, or carry the
        truncation explicitly — but do not present nothing as something.
        """
        ef = _rolling_origin_ef()
        report = NativeEvaluator(_real_model_config()).evaluate(
            ef=ef, legacy_compatibility=True
        )
        steps = report.to_dict()['schemas']['step']
        empty = [k for k, v in steps.items() if not v]
        assert not empty, (
            f"steps returned as empty placeholder dicts: {sorted(empty)} — "
            "the original C-20 defect"
        )

    def test_emitted_metric_frame_is_not_vacuous(self):
        """End to end: the evaluation-of-record must still be emittable from this path."""
        pytest.importorskip("views_frames")
        ef = _rolling_origin_ef()
        report = NativeEvaluator(_real_model_config()).evaluate(
            ef=ef, legacy_compatibility=True
        )
        mf = report.to_metric_frame(model_id='ravaging_thief', level='cm')
        assert mf.n_rows > 0


class TestExplicitOverRequestStillFailsLoud:
    """Whatever ruling 7 becomes, an over-request the caller CAN act on should stay loud.

    The distinction the current implementation misses: `legacy_compatibility=True` is an
    explicit request FOR truncation, so requesting more steps than the data supplies is
    not a contract violation — it is the flag doing its job. With the flag OFF, there is
    no truncation contract, so this test documents the case that should still be caught.
    """

    def test_steps_beyond_available_data_without_the_flag(self):
        """With legacy_compatibility=False, steps with no data yield nothing to score.

        This currently passes (no raise). It is here to pin the boundary: if ruling 7 is
        revised to be silent under the flag, it must not become silent without it.
        """
        ef = _rolling_origin_ef()
        report = NativeEvaluator(_real_model_config()).evaluate(
            ef=ef, legacy_compatibility=False
        )
        steps = report.to_dict()['schemas']['step']
        empty = [k for k, v in steps.items() if not v]
        assert not empty, (
            f"steps with no data returned as empty placeholder dicts: {sorted(empty)}"
        )
