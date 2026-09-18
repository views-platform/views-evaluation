import re
import warnings

import numpy as np
import pytest
from views_evaluation.evaluation.native_metric_calculators import (
    calculate_mse_native,
    calculate_msle_native,
    calculate_rmsle_native,
    calculate_crps_native,
    calculate_twcrps_native,
    calculate_quantile_interval_score_native,
    calculate_ap_native,
    calculate_emd_native,
    calculate_pearson_native,
    calculate_coverage_native,
    calculate_ignorance_score_native,
    calculate_mean_interval_score_native,
    calculate_mtd_native,
    calculate_mcr_native,
    calculate_brier_cls_point_native,
    calculate_brier_cls_sample_native,
    calculate_brier_rgs_sample_native,
    calculate_qs_sample_native,
    calculate_qs_point_native,
)
from views_evaluation.evaluation.metric_catalog import METRIC_MEMBERSHIP
from views_evaluation.evaluation.native_metric_calculators import (
    _average_precision_numpy,
    _tweedie_deviance_numpy,
)
# The parity oracle. Imported at module level on purpose: the parity tests record
# warnings around the oracle call, and a lazy import inside that recording would capture
# any import-time warning from the scikit-learn stack as a spurious parity failure.
from sklearn.metrics import average_precision_score, mean_tweedie_deviance


# Point-prediction test data (N=4, S=1)
_POINT_Y_TRUE = np.array([1.0, 2.0, 3.0, 4.0])
_POINT_Y_PRED = np.array([[1.1], [1.9], [3.1], [3.9]])

# Sample-prediction test data (N=4, S=3)
_SAMPLE_Y_TRUE = np.array([1.0, 2.0, 3.0, 4.0])
_SAMPLE_Y_PRED = np.array([[1.0, 1.1, 1.2], [1.8, 2.0, 2.2], [2.9, 3.0, 3.1], [3.8, 4.0, 4.2]])


def test_calculate_mse_native():
    """Test MSE calculation with pure NumPy arrays."""
    result = calculate_mse_native(_POINT_Y_TRUE, _POINT_Y_PRED)
    assert isinstance(result, float)
    assert result >= 0

def test_calculate_rmsle_native_point():
    """Test RMSLE calculation."""
    result = calculate_rmsle_native(_POINT_Y_TRUE, _POINT_Y_PRED)
    assert isinstance(result, float)
    assert result >= 0

def test_calculate_crps_native_point():
    """Test CRPS calculation with point predictions."""
    result = calculate_crps_native(_POINT_Y_TRUE, _POINT_Y_PRED)
    assert isinstance(result, float)
    assert result >= 0


def test_calculate_crps_native_sample():
    """Test CRPS calculation with sample predictions."""
    result = calculate_crps_native(_SAMPLE_Y_TRUE, _SAMPLE_Y_PRED)
    assert isinstance(result, float)
    assert result >= 0


def test_calculate_ap_native():
    """Test Average Precision with binary actuals and probability scores."""
    y_true = np.array([1.0, 0.0, 1.0, 0.0])
    y_pred = np.array([[0.9], [0.4], [0.3], [0.1]])
    result = calculate_ap_native(y_true, y_pred)
    assert isinstance(result, float)
    assert 0 <= result <= 1


def test_calculate_emd_native():
    """Test Earth Mover's Distance calculation."""
    result = calculate_emd_native(_POINT_Y_TRUE, _POINT_Y_PRED)
    assert isinstance(result, float)
    assert result >= 0


def test_calculate_pearson_native():
    """Test Pearson correlation calculation."""
    result = calculate_pearson_native(_POINT_Y_TRUE, _POINT_Y_PRED)
    assert isinstance(result, float)
    assert -1 <= result <= 1


def test_calculate_mtd_native():
    """Test Mean Tweedie Deviance calculation."""
    result = calculate_mtd_native(_POINT_Y_TRUE, _POINT_Y_PRED, power=1.5)
    assert isinstance(result, float)
    assert result >= 0


def test_calculate_mtd_native_with_power():
    """Test Mean Tweedie Deviance with different power values."""
    result_15 = calculate_mtd_native(_POINT_Y_TRUE, _POINT_Y_PRED, power=1.5)
    assert isinstance(result_15, float)
    assert result_15 >= 0

    result_2 = calculate_mtd_native(_POINT_Y_TRUE, _POINT_Y_PRED, power=2.0)
    assert isinstance(result_2, float)
    assert result_2 >= 0


def test_calculate_coverage_native_sample():
    """Test Coverage calculation with sample predictions."""
    result = calculate_coverage_native(_SAMPLE_Y_TRUE, _SAMPLE_Y_PRED, alpha=0.1)
    assert isinstance(result, float)
    assert 0 <= result <= 1


def test_calculate_ignorance_score_native_sample():
    """Test Ignorance Score calculation."""
    result = calculate_ignorance_score_native(
        _SAMPLE_Y_TRUE, _SAMPLE_Y_PRED,
        bins=[0, 0.5, 2.5, 5.5, 10.5, 25.5, 50.5, 100.5, 250.5, 500.5, 1000.5],
        low_bin=0, high_bin=10000,
    )
    assert isinstance(result, float)
    assert result >= 0


def test_calculate_mis_sample():
    """Test Mean Interval Score calculation."""
    result = calculate_mean_interval_score_native(_SAMPLE_Y_TRUE, _SAMPLE_Y_PRED, alpha=0.05)
    assert isinstance(result, float)
    assert result >= 0


def test_metric_membership_regression_point():
    """METRIC_MEMBERSHIP contains expected regression point metrics."""
    members = METRIC_MEMBERSHIP[("regression", "point")]
    for m in ["MSE", "MSLE", "RMSLE", "EMD", "Pearson", "MTD", "y_hat_bar", "MCR_point", "QS_point"]:
        assert m in members
    assert "AP" not in members
    assert "CRPS" not in members


def test_metric_membership_regression_sample():
    """METRIC_MEMBERSHIP contains expected regression sample metrics."""
    members = METRIC_MEMBERSHIP[("regression", "sample")]
    for m in ["CRPS", "twCRPS", "MIS", "QIS", "Coverage", "Ignorance", "y_hat_bar", "QS_sample", "MCR_sample"]:
        assert m in members
    assert "AP" not in members


def test_metric_membership_classification_point():
    """METRIC_MEMBERSHIP contains expected classification point metrics."""
    members = METRIC_MEMBERSHIP[("classification", "point")]
    assert "AP" in members
    assert "Brier_cls_point" in members
    assert "RMSLE" not in members


def test_metric_membership_classification_sample():
    """METRIC_MEMBERSHIP contains expected classification sample metrics."""
    members = METRIC_MEMBERSHIP[("classification", "sample")]
    for m in ["CRPS", "twCRPS", "Brier_cls_sample", "Jeffreys"]:
        assert m in members
    assert "RMSLE" not in members


def test_not_implemented_metrics():
    """Test that unimplemented metrics raise ValueError with clear message."""
    from views_evaluation.evaluation.native_metric_calculators import (
        calculate_jeffreys_native,
        calculate_sd_native,
        calculate_pEMDiv_native,
        calculate_variogram_native,
    )

    for func in [calculate_jeffreys_native, calculate_sd_native,
                 calculate_pEMDiv_native, calculate_variogram_native]:
        with pytest.raises(ValueError, match="not yet implemented"):
            func(np.array([1.0]), np.array([[1.0]]))


# ---------------------------------------------------------------------------
# CRPS parity tests: pure-numpy vs properscoring oracle
# ---------------------------------------------------------------------------

class TestCRPSParityWithProperscoring:
    """Prove that the pure-numpy CRPS matches properscoring.crps_ensemble."""

    @staticmethod
    def _crps_oracle(y_true, y_pred):
        """Reference CRPS using properscoring (dev-only dependency)."""
        import properscoring as ps
        return float(np.mean(ps.crps_ensemble(y_true, y_pred, axis=1)))

    def test_parity_single_member_ensemble(self):
        """Point forecast (1-sample ensemble)."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([[1.5], [2.5], [3.5]])
        ours = calculate_crps_native(y_true, y_pred)
        oracle = self._crps_oracle(y_true, y_pred)
        assert ours == pytest.approx(oracle, abs=1e-10)

    def test_parity_small_ensemble(self):
        """5-member ensemble."""
        rng = np.random.RandomState(42)
        y_true = rng.rand(10)
        y_pred = rng.rand(10, 5)
        ours = calculate_crps_native(y_true, y_pred)
        oracle = self._crps_oracle(y_true, y_pred)
        assert ours == pytest.approx(oracle, abs=1e-10)

    def test_parity_large_ensemble(self):
        """100-member ensemble."""
        rng = np.random.RandomState(123)
        y_true = rng.rand(50)
        y_pred = rng.rand(50, 100)
        ours = calculate_crps_native(y_true, y_pred)
        oracle = self._crps_oracle(y_true, y_pred)
        assert ours == pytest.approx(oracle, abs=1e-10)

    def test_parity_constant_ensemble(self):
        """All ensemble members identical (degenerate case)."""
        y_true = np.array([3.0, 5.0])
        y_pred = np.full((2, 10), 4.0)
        ours = calculate_crps_native(y_true, y_pred)
        oracle = self._crps_oracle(y_true, y_pred)
        assert ours == pytest.approx(oracle, abs=1e-10)

    def test_parity_perfect_forecast(self):
        """Ensemble centered on truth should give CRPS near 0 for large S."""
        y_true = np.array([0.0])
        y_pred = np.zeros((1, 50))
        ours = calculate_crps_native(y_true, y_pred)
        oracle = self._crps_oracle(y_true, y_pred)
        assert ours == pytest.approx(oracle, abs=1e-10)
        assert ours == pytest.approx(0.0, abs=1e-10)

    def test_parity_single_observation(self):
        """N=1 edge case."""
        y_true = np.array([5.0])
        y_pred = np.array([[3.0, 4.0, 5.0, 6.0, 7.0]])
        ours = calculate_crps_native(y_true, y_pred)
        oracle = self._crps_oracle(y_true, y_pred)
        assert ours == pytest.approx(oracle, abs=1e-10)

    def test_parity_wide_spread(self):
        """Large spread ensemble to test numerical stability."""
        rng = np.random.RandomState(999)
        y_true = rng.rand(20) * 1000
        y_pred = rng.rand(20, 30) * 1000
        ours = calculate_crps_native(y_true, y_pred)
        oracle = self._crps_oracle(y_true, y_pred)
        assert ours == pytest.approx(oracle, abs=1e-7)


# ---------------------------------------------------------------------------
# AP / MTD parity: pure-numpy candidates vs scikit-learn oracle (C-05; epic #66, S4)
# ---------------------------------------------------------------------------
#
# The candidates are `_average_precision_numpy` and `_tweedie_deviance_numpy`. Nothing
# dispatches to them yet; these tests are the gate #64 must pass — and must have been
# seen failing on a deliberately broken kernel — before the public kernels switch and
# the sklearn import is dropped. The oracle is whatever scikit-learn the environment
# resolves (C-41): a numeric parity failure after a scikit-learn upgrade is a finding
# about the oracle, never a reason to loosen the tolerance.


def _as_kernel_input(y_true, y_pred):
    """The exact form the public kernels hand to the oracle: truth repeated per sample."""
    return np.repeat(y_true, y_pred.shape[1]), y_pred.flatten()


def test_as_kernel_input_matches_what_the_public_kernels_build():
    """A parity suite cannot see a defect shared by both paths, and this helper IS the
    shared path. Pin it to the construction in calculate_ap_native / calculate_mtd_native
    by observing what they hand to the oracle."""
    from unittest import mock
    y_true = np.array([1., 0., 1.])
    y_pred = np.array([[.1, .2], [.3, .4], [.5, .6]])
    seen = {}

    def spy(yt, ys, **kw):
        seen["ap"] = (yt.copy(), ys.copy())
        return 0.0

    def spy_mtd(yt, ys, power):
        seen["mtd"] = (yt.copy(), ys.copy())
        return 0.0

    with mock.patch("views_evaluation.evaluation.native_metric_calculators.average_precision_score", spy), \
         mock.patch("views_evaluation.evaluation.native_metric_calculators.mean_tweedie_deviance", spy_mtd):
        calculate_ap_native(y_true, y_pred)
        calculate_mtd_native(y_true, y_pred, power=1.5)
    expected = _as_kernel_input(y_true, y_pred)
    for key in ("ap", "mtd"):
        np.testing.assert_array_equal(seen[key][0], expected[0])
        np.testing.assert_array_equal(seen[key][1], expected[1])


def _warnings_of(fn, *args, **kwargs):
    """(result, [(category, message)]) — the two paths are compared on both."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    return result, [(x.category, str(x.message)) for x in w]


class TestAPParityWithSklearn:
    """The numpy AP must equal `sklearn.metrics.average_precision_score` to 1e-10 on
    every shape of input the public kernel can see, and warn where and as it warns."""

    def _assert_parity(self, y_true, y_pred):
        yt, ys = _as_kernel_input(y_true, y_pred)
        ours, ours_w = _warnings_of(_average_precision_numpy, yt, ys)
        ref, ref_w = _warnings_of(average_precision_score, yt, ys)
        assert ours == pytest.approx(ref, abs=1e-10), (ours, ref)
        assert type(ours) is float  # a Python float, not np.float64
        assert ours_w == ref_w, f"warnings differ: ours {ours_w} vs sklearn {ref_w}"

    # ── deterministic shapes ──
    def test_perfect_ranking(self):
        self._assert_parity(np.array([1., 1., 0., 0.]), np.array([[.9], [.8], [.2], [.1]]))

    def test_inverted_ranking(self):
        self._assert_parity(np.array([0., 0., 1., 1.]), np.array([[.9], [.8], [.2], [.1]]))

    def test_all_scores_tied(self):
        self._assert_parity(np.array([1., 0., 1., 0.]), np.full((4, 1), .5))

    def test_all_positive_truth(self):
        self._assert_parity(np.ones(5), np.array([[.1], [.5], [.9], [.3], [.7]]))

    def test_all_negative_truth_warns_and_is_zero(self):
        """sklearn returns 0.0 and warns (plain UserWarning, this text); so must we,
        identically — transcribed, not endorsed: #64 owes ADR-015 a ruling here."""
        yt, ys = _as_kernel_input(np.zeros(4), np.array([[.1], [.2], [.3], [.4]]))
        ours, ours_w = _warnings_of(_average_precision_numpy, yt, ys)
        ref, ref_w = _warnings_of(average_precision_score, yt, ys)
        assert ours == ref == 0.0
        assert ours_w == ref_w == [
            (UserWarning, "No positive class found in y_true, recall is set to one for all thresholds.")
        ]

    def test_single_positive(self):
        self._assert_parity(np.array([0., 0., 1., 0., 0.]), np.array([[.3], [.9], [.5], [.1], [.5]]))

    def test_n_equals_one(self):
        self._assert_parity(np.array([1.]), np.array([[.42]]))

    def test_n_equals_two(self):
        self._assert_parity(np.array([1., 0.]), np.array([[.3], [.7]]))

    def test_duplicated_scores_with_mixed_labels(self):
        """The tie-grouping case: step-wise and trapezoidal AP differ here."""
        self._assert_parity(np.array([1., 0., 1., 0., 1., 0.]),
                            np.array([[.9], [.9], [.5], [.5], [.5], [.1]]))

    def test_label_set_minus_one_and_one(self):
        self._assert_parity(np.array([-1., 1., 1., -1.]), np.array([[.2], [.8], [.6], [.4]]))

    def test_label_set_one_and_two_treats_one_as_positive(self):
        """sklearn silently computes with 1 as the positive class; parity, not judgement."""
        self._assert_parity(np.array([1., 2., 1., 2.]), np.array([[.9], [.8], [.2], [.1]]))

    def test_integer_truth_and_scores(self):
        self._assert_parity(np.array([1, 0, 1, 0]), np.array([[3], [1], [2], [2]]))

    def test_bool_truth_computes(self):
        """A bool label array is a valid binary target on both paths."""
        self._assert_parity(np.array([True, False, True, False]), np.array([[.9], [.1], [.8], [.2]]))

    def test_python_lists_are_accepted(self):
        yt, ys = [1., 0., 1., 0.], [.9, .1, .8, .2]
        assert _average_precision_numpy(yt, ys) == pytest.approx(average_precision_score(yt, ys), abs=1e-10)

    def test_scores_one_ulp_apart_are_distinct(self):
        """Ties are exact equality, not a tolerance: scores one ulp apart are separate
        operating points, on both paths. Chosen so that merging them changes AP: the
        positive sits one ulp ABOVE the negative, so exact ordering ranks it alone first
        (precision 1 there) while any tolerance ties it with the negative and lowers it."""
        s = 0.5
        self._assert_parity(np.array([0., 1., 0., 1.]),
                            np.array([[s], [np.nextafter(s, 1.0)], [.1], [.9]]))

    def test_int64_scores_above_2_pow_53_stay_distinct(self):
        """A float64 cast would merge these two into a tie; sklearn sorts natively."""
        self._assert_parity(np.array([1., 0.]), np.array([[2**53 + 1], [2**53]], dtype=np.int64))

    @pytest.mark.parametrize("S", [1, 10, 100])
    def test_ensemble_widths(self, S):
        rng = np.random.RandomState(7)
        y_true = (rng.rand(20) < .4).astype(float)
        self._assert_parity(y_true, rng.rand(20, S))

    def test_large_n_and_s(self):
        """One N=5000, S=100 case — the sweep stays at N ≤ 500 to keep the suite fast."""
        rng = np.random.RandomState(99)
        self._assert_parity((rng.rand(5000) < .3).astype(float), np.round(rng.rand(5000, 100), 2))

    # ── seeded sweep: N × S × prevalence, heavy ties from rounding ──
    @pytest.mark.parametrize("seed", range(200))
    def test_random_sweep(self, seed):
        rng = np.random.RandomState(seed)
        N = int(rng.choice([5, 50, 500]))
        S = int(rng.choice([1, 10, 100]))
        prevalence = rng.uniform(.01, .99)
        y_true = (rng.rand(N) < prevalence).astype(float)
        y_pred = np.round(rng.rand(N, S), int(rng.choice([1, 2, 6])))
        self._assert_parity(y_true, y_pred)

    @pytest.mark.parametrize("scale", [1e6, 1e-6])
    def test_extreme_score_magnitudes(self, scale):
        rng = np.random.RandomState(3)
        self._assert_parity((rng.rand(50) < .5).astype(float), rng.rand(50, 3) * scale)

    # ── domain: both raise ValueError; the TEXT is a documented deviation ──
    @pytest.mark.parametrize("y_true", [
        np.array([0., 1., 2., 0.]),      # three labels
        np.array([0., 2., 0., 2.]),      # two labels, neither is 1
        np.array([0.5, 1., 0.5, 1.]),    # two labels, one non-integral: sklearn "continuous"
        np.array([0.2, 0.7, 0.4, 0.9]),  # continuous
    ], ids=["three-labels", "no-positive-label", "non-integral", "continuous"])
    def test_non_binary_truth_raises_on_both_paths(self, y_true):
        yt, ys = _as_kernel_input(y_true, np.array([[.1], [.4], [.35], [.8]]))
        with pytest.raises(ValueError):
            average_precision_score(yt, ys)
        with pytest.raises(ValueError, match="binary y_true"):
            _average_precision_numpy(yt, ys)

    # ── input validation the oracle did via check_array; ours must keep raising ──
    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _average_precision_numpy(np.array([]), np.array([]))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="different lengths"):
            _average_precision_numpy(np.array([1., 0., 1.]), np.array([.9, .1]))

    def test_column_shaped_input_raises(self):
        with pytest.raises(ValueError, match="must be 1-D"):
            _average_precision_numpy(np.array([1., 0.]), np.array([[.9], [.1]]))

    def test_zero_dimensional_input_raises(self):
        with pytest.raises(ValueError, match="must be 1-D"):
            _average_precision_numpy(np.array(1.), np.array(.5))

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_score_raises_naming_value_and_index(self, bad):
        """The message names which array, which value, and where — ADR-015."""
        with pytest.raises(ValueError, match=rf"^y_score contains {re.escape(repr(bad))} at index 2; values must be finite$"):
            _average_precision_numpy(np.array([1., 0., 1., 0.]), np.array([.9, .8, bad, .1]))

    def test_nan_in_truth_raises(self):
        with pytest.raises(ValueError, match="must be finite"):
            _average_precision_numpy(np.array([1., np.nan, 1., 0.]), np.array([.9, .8, .2, .1]))

    @pytest.mark.parametrize("arr", [
        np.array([.1 + 1j, .4, .35, .8]),
        np.array(["0.1", "0.4", "0.35", "0.8"]),
    ], ids=["complex", "string"])
    def test_non_real_dtype_raises(self, arr):
        with pytest.raises(ValueError, match="real-numeric"):
            _average_precision_numpy(np.array([1., 0., 1., 0.]), arr)


class TestMTDParityWithSklearn:
    """The numpy Tweedie deviance must equal `sklearn.metrics.mean_tweedie_deviance` to
    1e-10 (relative) on every power branch and every domain rule, for float64 input."""

    POWERS = [-2, -1, -0.5, 0, 1, 1.0001, 1.5, 1.9999, 2, 2.0001, 2.5, 3, 5]

    def _assert_parity(self, y_true, y_pred, power, rel=1e-10):
        yt, ys = _as_kernel_input(y_true, y_pred)
        ours, ours_w = _warnings_of(_tweedie_deviance_numpy, yt, ys, power)
        ref, ref_w = _warnings_of(mean_tweedie_deviance, yt, ys, power=power)
        assert ours == pytest.approx(ref, rel=rel, abs=1e-12), (power, ours, ref)
        assert type(ours) is float
        assert ours_w == ref_w, f"warnings differ at p={power}: ours {ours_w} vs sklearn {ref_w}"

    @staticmethod
    def _positive_data(seed, N=50, S=1):
        rng = np.random.RandomState(seed)
        return rng.gamma(2., 2., N) + 1e-3, rng.gamma(2., 2., (N, S)) + 1e-3

    @pytest.mark.parametrize("power", POWERS)
    def test_every_power_branch(self, power):
        y, mu = self._positive_data(11)
        self._assert_parity(y, mu, power)

    @pytest.mark.parametrize("power", [1, 1.0001, 1.5, 1.9999])
    def test_exact_zeros_in_truth_where_permitted(self, power):
        y, mu = self._positive_data(5)
        y[::7] = 0.0
        self._assert_parity(y, mu, power)

    @pytest.mark.parametrize("power", POWERS)
    def test_truth_equals_prediction_is_near_zero(self, power):
        """Deviance of a perfect prediction is 0 up to round-off; both paths agree."""
        y, _ = self._positive_data(9)
        mu = y[:, np.newaxis].copy()
        self._assert_parity(y, mu, power)
        assert _tweedie_deviance_numpy(*_as_kernel_input(y, mu), power) == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("power", POWERS)
    @pytest.mark.parametrize("ratio", [1e-3, 1e3], ids=["y<<mu", "y>>mu"])
    def test_extreme_ratios(self, power, ratio):
        y, mu = self._positive_data(13)
        self._assert_parity(y * ratio, mu, power)

    @pytest.mark.parametrize("S", [1, 10, 100])
    def test_ensemble_widths(self, S):
        y, mu = self._positive_data(17, S=S)
        self._assert_parity(y, mu, 1.5)

    def test_large_n_and_s(self):
        """One N=5000, S=100 case per branch family — the sweep stays at N ≤ 500."""
        y, mu = self._positive_data(99, N=5000, S=100)
        for power in (-1, 0, 1, 1.5, 2, 3):
            self._assert_parity(y, mu, power)

    def test_negative_truth_allowed_below_zero_power(self):
        """p < 0 admits any real y (the first term clamps at 0); only y_pred must be > 0."""
        y, mu = self._positive_data(19)
        self._assert_parity(y - 5.0, mu, -1)

    def test_power_zero_admits_any_values(self):
        y, mu = self._positive_data(23)
        self._assert_parity(y - 5.0, mu - 5.0, 0)

    @pytest.mark.parametrize("seed", range(200))
    def test_random_sweep(self, seed):
        """One dataset per seed, every power on it; negative y where the branch admits
        it, a zero in y where it is permitted. 200 datasets × 13 powers."""
        rng = np.random.RandomState(seed)
        N = int(rng.choice([5, 50, 500]))
        S = int(rng.choice([1, 10, 100]))
        y = rng.gamma(2., 2., N) + 1e-3
        mu = rng.gamma(2., 2., (N, S)) + 1e-3
        for power in self.POWERS:
            yy = y.copy()
            if power <= 0 and rng.rand() < .5:
                yy = yy - rng.rand() * 5.0          # negative y admitted for p <= 0
            if 1 <= power < 2 and rng.rand() < .3:
                yy[rng.randint(N)] = 0.0             # zero admitted for 1 <= p < 2
            self._assert_parity(yy, mu, power)

    @pytest.mark.parametrize("scale", [1e6, 1e-6])
    def test_extreme_magnitudes(self, scale):
        y, mu = self._positive_data(29)
        self._assert_parity(y * scale, mu * scale, 1.5)

    @pytest.mark.parametrize("power", POWERS)
    def test_float32_input_is_computed_in_float64(self, power):
        """Documented deviation: sklearn computes float32 input in float32; we always
        upcast. Parity to sklearn's *float64* answer on the same numbers is tight. The
        gap to its *float32* answer is the deviation — ~1e-8 at p=1.5, up to ~2e-3 near
        p=1 and p=2 where the closed form cancels — and is recorded, not hidden."""
        y, mu = self._positive_data(31)
        y32, mu32 = y.astype(np.float32), mu.astype(np.float32)
        ours = _tweedie_deviance_numpy(*_as_kernel_input(y32, mu32), power)
        ref64 = mean_tweedie_deviance(*_as_kernel_input(y32.astype(np.float64), mu32.astype(np.float64)), power=power)
        ref32 = mean_tweedie_deviance(*_as_kernel_input(y32, mu32), power=power)
        assert ours == pytest.approx(ref64, rel=1e-10)
        assert ours == pytest.approx(ref32, rel=5e-3), f"float32 gap larger than documented at p={power}"

    # ── domain rules: raise where sklearn raises, with its sentence verbatim ──
    @pytest.mark.parametrize("power, y_edit, mu_edit, sentence", [
        (-1,    None,  0.0,  "strictly positive y_pred."),
        (1,     -1.0,  None, "non-negative y and strictly positive y_pred."),
        (1,     None,  0.0,  "non-negative y and strictly positive y_pred."),
        (1.5,   -1.0,  None, "non-negative y and strictly positive y_pred."),
        (1.5,   None,  0.0,  "non-negative y and strictly positive y_pred."),
        (2,     0.0,   None, "strictly positive y and y_pred."),
        (2,     None,  0.0,  "strictly positive y and y_pred."),
        (3,     0.0,   None, "strictly positive y and y_pred."),
    ])
    def test_domain_errors_match_sklearn(self, power, y_edit, mu_edit, sentence):
        y, mu = self._positive_data(37)
        if y_edit is not None:
            y[0] = y_edit
        if mu_edit is not None:
            mu[0, 0] = mu_edit
        yt, ys = _as_kernel_input(y, mu)
        # The full sentence, anchored, so the p<0 text is not satisfied by the [1,2) one.
        full = re.escape(f"Mean Tweedie deviance error with power={power} can only be used on {sentence}")
        with pytest.raises(ValueError, match="^" + full + "$"):
            mean_tweedie_deviance(yt, ys, power=power)
        with pytest.raises(ValueError, match="^" + full + r" Offending (y|y_pred): .* at index 0\.$"):
            _tweedie_deviance_numpy(yt, ys, power)

    def test_domain_error_names_the_first_offender_by_value_and_index(self):
        """Not index 0, not the first array: the third y_pred is the offender, and the
        message prints that value at that index."""
        y, mu = self._positive_data(47)
        mu[2, 0] = -0.25
        mu[7, 0] = -0.75   # a second offender: the FIRST must be reported
        yt, ys = _as_kernel_input(y, mu)
        with pytest.raises(ValueError, match=r" Offending y_pred: -0\.25 at index 2\.$"):
            _tweedie_deviance_numpy(yt, ys, 1.5)

    def test_domain_error_reports_y_before_y_pred_when_both_offend(self):
        """Precedence is y then y_pred, as scikit-learn checks them; with both offending
        the message names y — and y's value, not y_pred's."""
        y, mu = self._positive_data(53)
        y[4] = -3.0
        mu[1, 0] = 0.0
        yt, ys = _as_kernel_input(y, mu)
        with pytest.raises(ValueError, match=r" Offending y: -3\.0 at index 4\.$"):
            _tweedie_deviance_numpy(yt, ys, 1.5)

    def test_fraction_power_is_a_real_number(self):
        """A numbers.Real that numpy cannot coerce directly must compute as the equal
        float does, not raise TypeError. (scikit-learn itself cannot take a Fraction, so
        this is a property test, not a parity one.)"""
        from fractions import Fraction
        y, mu = self._positive_data(59)
        yt, ys = _as_kernel_input(y, mu)
        assert _tweedie_deviance_numpy(yt, ys, Fraction(3, 2)) == _tweedie_deviance_numpy(yt, ys, 1.5)

    def test_python_lists_are_accepted(self):
        yt, ys = [1., 2., 3.], [1.5, 2.5, 2.5]
        assert _tweedie_deviance_numpy(yt, ys, 1.5) == pytest.approx(mean_tweedie_deviance(yt, ys, power=1.5), rel=1e-10)

    @pytest.mark.parametrize("power", [0.5, 0.0001, 0.9999])
    def test_power_in_open_unit_interval_raises_on_both_paths(self, power):
        y, mu = self._positive_data(41)
        yt, ys = _as_kernel_input(y, mu)
        with pytest.raises(ValueError):
            mean_tweedie_deviance(yt, ys, power=power)
        with pytest.raises(ValueError, match=r"not defined for power in \(0, 1\)"):
            _tweedie_deviance_numpy(yt, ys, power)

    @pytest.mark.parametrize("power", [np.nan, np.inf, -np.inf, "1.5", None, True],
                             ids=["nan", "inf", "-inf", "str", "None", "bool"])
    def test_invalid_power_raises(self, power):
        """sklearn's validate_params rejects these; the numpy kernel must too, and must
        never fall through to a branch (inf used to reach `else: p >= 2` and score 0.0)."""
        y, mu = self._positive_data(43)
        with pytest.raises(ValueError, match="finite real power"):
            _tweedie_deviance_numpy(*_as_kernel_input(y, mu), power)

    # ── input validation the oracle did via check_array; ours must keep raising ──
    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _tweedie_deviance_numpy(np.array([]), np.array([]), 1.5)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="different lengths"):
            _tweedie_deviance_numpy(np.array([1., 2., 3.]), np.array([1., 2.]), 1.5)

    def test_column_shaped_input_raises(self):
        with pytest.raises(ValueError, match="must be 1-D"):
            _tweedie_deviance_numpy(np.array([1., 2.]), np.array([[1.], [2.]]), 1.5)

    @pytest.mark.parametrize("power", [-1, 0, 1, 1.5, 2, 3])
    def test_nan_input_raises(self, power):
        with pytest.raises(ValueError, match="must be finite"):
            _tweedie_deviance_numpy(np.array([1., np.nan, 3.]), np.array([1., 2., 3.]), power)

    def test_inf_prediction_raises_at_power_three(self):
        """inf**(1-p) and inf**(2-p) are both 0 for p=3, so an unchecked inf produced a
        finite, plausible, wrong value."""
        with pytest.raises(ValueError, match="must be finite"):
            _tweedie_deviance_numpy(np.array([1., 2., 3., .5]), np.array([1.5, np.inf, 2.5, .7]), 3)

    @pytest.mark.parametrize("arr", [
        np.array([1.5 + 1j, 2.5, 2.5, 3.5]),
        np.array(["1.5", "2.5", "2.5", "3.5"]),
    ], ids=["complex", "string"])
    def test_non_real_dtype_raises(self, arr):
        with pytest.raises(ValueError, match="real-numeric"):
            _tweedie_deviance_numpy(np.array([1., 2., 3., 4.]), arr, 1.5)


# ---------------------------------------------------------------------------
# twCRPS tests
# ---------------------------------------------------------------------------

class TestTwCRPS:

    def test_twcrps_basic_smoke(self):
        """twCRPS produces a non-negative float."""
        result = calculate_twcrps_native(_SAMPLE_Y_TRUE, _SAMPLE_Y_PRED, threshold=0.0)
        assert isinstance(result, float)
        assert result >= 0

    def test_twcrps_threshold_zero_positive_data_equals_crps(self):
        """For all-positive data, twCRPS(threshold=0) == CRPS (max(x,0) == x)."""
        rng = np.random.RandomState(42)
        y_true = rng.rand(20) + 1.0  # all > 0
        y_pred = rng.rand(20, 10) + 0.5  # all > 0
        crps = calculate_crps_native(y_true, y_pred)
        twcrps = calculate_twcrps_native(y_true, y_pred, threshold=0.0)
        assert twcrps == pytest.approx(crps, abs=1e-10)

    def test_twcrps_high_threshold_reduces_score(self):
        """With threshold above all data, everything clamps to the threshold → CRPS=0."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])
        twcrps = calculate_twcrps_native(y_true, y_pred, threshold=100.0)
        # max(y, 100) == 100 for all, so all values are identical → CRPS = 0
        assert twcrps == pytest.approx(0.0, abs=1e-10)

    def test_twcrps_threshold_changes_result(self):
        """twCRPS with a meaningful threshold differs from standard CRPS."""
        rng = np.random.RandomState(7)
        y_true = rng.rand(20) * 10
        y_pred = rng.rand(20, 15) * 10
        crps = calculate_crps_native(y_true, y_pred)
        twcrps = calculate_twcrps_native(y_true, y_pred, threshold=5.0)
        # They should differ for data straddling the threshold
        assert twcrps != pytest.approx(crps, abs=1e-5)

    def test_twcrps_in_metric_membership(self):
        """twCRPS must be in both regression and classification sample membership."""
        assert "twCRPS" in METRIC_MEMBERSHIP[("regression", "sample")]
        assert "twCRPS" in METRIC_MEMBERSHIP[("classification", "sample")]


# ---------------------------------------------------------------------------
# Quantile Interval Score tests
# ---------------------------------------------------------------------------

class TestQuantileIntervalScore:

    def test_qis_basic_smoke(self):
        """QIS produces a non-negative float."""
        result = calculate_quantile_interval_score_native(
            _SAMPLE_Y_TRUE, _SAMPLE_Y_PRED, lower_quantile=0.025, upper_quantile=0.975,
        )
        assert isinstance(result, float)
        assert result >= 0

    def test_qis_symmetric_agrees_with_mis_on_interval_width(self):
        """
        QIS and MIS use different penalty conventions but identical quantile intervals.

        MIS (Gneiting & Raftery 2007) uses penalty 2/alpha for a (1-alpha) interval.
        QIS (quantile decomposition) uses 2/q_lower and 2/(1-q_upper).
        When q_lower=alpha/2, the QIS penalty is 4/alpha — twice the MIS penalty.

        Verify they share the same interval bounds but differ in penalty scaling.
        """
        rng = np.random.RandomState(42)
        y_true = rng.rand(30)
        y_pred = rng.rand(30, 20)
        alpha = 0.05
        mis = calculate_mean_interval_score_native(y_true, y_pred, alpha=alpha)
        qis = calculate_quantile_interval_score_native(
            y_true, y_pred,
            lower_quantile=alpha / 2,
            upper_quantile=1 - alpha / 2,
        )
        # Both produce finite non-negative values
        assert mis >= 0
        assert qis >= 0
        # QIS penalty is 2× MIS penalty, so QIS >= MIS when there are violations
        assert qis >= mis or qis == pytest.approx(mis, abs=1e-10)

    def test_qis_no_violations_equals_mis_no_violations(self):
        """When all obs fall within the interval, both QIS and MIS equal the interval width."""
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([
            [3.0, 4.0, 5.0, 6.0, 7.0],
            [3.0, 4.0, 5.0, 6.0, 7.0],
            [3.0, 4.0, 5.0, 6.0, 7.0],
        ])
        alpha = 0.10
        mis = calculate_mean_interval_score_native(y_true, y_pred, alpha=alpha)
        qis = calculate_quantile_interval_score_native(
            y_true, y_pred,
            lower_quantile=alpha / 2,
            upper_quantile=1 - alpha / 2,
        )
        # With no violations, penalty terms are zero → both equal interval width
        assert qis == pytest.approx(mis, abs=1e-12)

    def test_qis_asymmetric_differs_from_symmetric(self):
        """Asymmetric quantile levels should produce a different score."""
        rng = np.random.RandomState(42)
        y_true = rng.rand(30)
        y_pred = rng.rand(30, 20)
        symmetric = calculate_quantile_interval_score_native(
            y_true, y_pred, lower_quantile=0.05, upper_quantile=0.95,
        )
        asymmetric = calculate_quantile_interval_score_native(
            y_true, y_pred, lower_quantile=0.10, upper_quantile=0.95,
        )
        assert symmetric != pytest.approx(asymmetric, abs=1e-5)

    def test_qis_golden_value_hand_computed(self):
        """
        Hand-computed QIS for a trivial case.

        y_true = [10.0], y_pred = [[5, 10, 15]] (3 members)
        lower_quantile=0.1, upper_quantile=0.9
        lower = quantile(0.1) of [5,10,15] = 5 + 0.1*2*(10-5) = 6.0
        upper = quantile(0.9) of [5,10,15] = 10 + 0.9*2*(15-10)... actually np.quantile uses linear interpolation

        Let's compute directly with numpy to establish the golden value.
        """
        y_true = np.array([10.0])
        y_pred = np.array([[5.0, 10.0, 15.0]])
        q_lo, q_hi = 0.1, 0.9

        lower = float(np.quantile(y_pred, q_lo, axis=1).item())
        upper = float(np.quantile(y_pred, q_hi, axis=1).item())
        # y_true=10 should be within [lower, upper], so penalties = 0
        width = upper - lower
        expected = width  # no violation penalty

        result = calculate_quantile_interval_score_native(
            y_true, y_pred, lower_quantile=q_lo, upper_quantile=q_hi,
        )
        assert result == pytest.approx(expected, abs=1e-10)

    def test_qis_golden_value_with_violation(self):
        """Hand-computed QIS with a clear upper violation."""
        y_true = np.array([20.0])  # way above the ensemble
        y_pred = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        q_lo, q_hi = 0.1, 0.9

        lower = np.quantile(y_pred, q_lo, axis=1).item()
        upper = np.quantile(y_pred, q_hi, axis=1).item()

        width = upper - lower
        upper_penalty = (2 / (1 - q_hi)) * (20.0 - upper)
        expected = width + upper_penalty

        result = calculate_quantile_interval_score_native(
            y_true, y_pred, lower_quantile=q_lo, upper_quantile=q_hi,
        )
        assert result == pytest.approx(expected, abs=1e-10)

    def test_qis_perfect_coverage_minimal_score(self):
        """If all obs fall within the interval, score equals interval width (no penalty)."""
        y_true = np.array([5.0, 5.0, 5.0])
        # Ensemble tightly around 5.0
        y_pred = np.array([
            [4.0, 4.5, 5.0, 5.5, 6.0],
            [4.0, 4.5, 5.0, 5.5, 6.0],
            [4.0, 4.5, 5.0, 5.5, 6.0],
        ])
        result = calculate_quantile_interval_score_native(
            y_true, y_pred, lower_quantile=0.1, upper_quantile=0.9,
        )
        # All obs are within the 10th-90th percentile interval, so
        # score = mean(interval_width) with zero penalty terms
        assert result >= 0
        lower = np.quantile(y_pred, 0.1, axis=1)
        upper = np.quantile(y_pred, 0.9, axis=1)
        expected_width = float(np.mean(upper - lower))
        assert result == pytest.approx(expected_width, abs=1e-10)

    def test_qis_in_metric_membership(self):
        """QIS must be in regression sample membership."""
        assert "QIS" in METRIC_MEMBERSHIP[("regression", "sample")]


# ---------------------------------------------------------------------------
# Green: Golden-value correctness tests — hand-computed expected values (ADR-020)
# ---------------------------------------------------------------------------

class TestGoldenValues:
    """Verify numerical correctness of all implemented metrics against hand-computed or oracle values."""

    def test_mse_known_errors(self):
        """y_true=[1,2,3], y_pred=[[2],[3],[4]] → errors=[1,1,1], MSE=1.0."""
        result = calculate_mse_native(np.array([1.0, 2.0, 3.0]), np.array([[2.0], [3.0], [4.0]]))
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_msle_known_values(self):
        """y_true=[e-1], y_pred=[[0]] → log1p(e-1)=1, log1p(0)=0, MSLE=1.0."""
        result = calculate_msle_native(np.array([np.e - 1]), np.array([[0.0]]))
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_rmsle_is_sqrt_msle(self):
        """RMSLE = sqrt(MSLE) for the same input."""
        y_true = np.array([np.e - 1])
        y_pred = np.array([[0.0]])
        msle = calculate_msle_native(y_true, y_pred)
        rmsle = calculate_rmsle_native(y_true, y_pred)
        assert rmsle == pytest.approx(np.sqrt(msle), abs=1e-10)

    def test_emd_point_prediction(self):
        """y_true=[0], y_pred=[[5]] → wasserstein_distance([5],[0]) = 5.0."""
        result = calculate_emd_native(np.array([0.0]), np.array([[5.0]]))
        assert result == pytest.approx(5.0, abs=1e-10)

    def test_pearson_perfect_correlation(self):
        """y_true=[1,2,3], y_pred=[[1],[2],[3]] → r = 1.0."""
        result = calculate_pearson_native(np.array([1.0, 2.0, 3.0]), np.array([[1.0], [2.0], [3.0]]))
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_pearson_perfect_negative(self):
        """y_true=[1,2,3], y_pred=[[3],[2],[1]] → r = -1.0."""
        result = calculate_pearson_native(np.array([1.0, 2.0, 3.0]), np.array([[3.0], [2.0], [1.0]]))
        assert result == pytest.approx(-1.0, abs=1e-10)

    def test_mtd_known_tweedie(self):
        """Tweedie deviance with power=2 reduces to (y/mu - ln(y/mu) - 1) * 2."""
        from sklearn.metrics import mean_tweedie_deviance
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([[2.0], [2.0], [2.0]])
        expected = mean_tweedie_deviance(
            np.repeat(y_true, 1), y_pred.flatten(), power=2
        )
        result = calculate_mtd_native(y_true, y_pred, power=2)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_mcr_perfect_calibration(self):
        """mean(y_pred) == mean(y_true) → MCR = 1.0."""
        y_true = np.array([2.0, 4.0, 6.0])
        y_pred = np.array([[2.0], [4.0], [6.0]])
        result = calculate_mcr_native(y_true, y_pred)
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_mcr_double_overprediction(self):
        """mean(y_pred) = 2 * mean(y_true) → MCR = 2.0."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([[2.0], [4.0], [6.0]])
        result = calculate_mcr_native(y_true, y_pred)
        assert result == pytest.approx(2.0, abs=1e-10)

    def test_ignorance_known_bin_distribution(self):
        """Hand-computed Ignorance: 5 ensemble members, 3 bins, known distribution.

        bins=[0,4,8,12], preds=[1,3,5,7,9] → bin counts [2,2,1]
        smoothed=[3,3,2], total=8. Truth 5.0 → bin 1, prob=3/8.
        Score = -log2(3/8) = log2(8/3).
        """
        y_true = np.array([5.0])
        y_pred = np.array([[1.0, 3.0, 5.0, 7.0, 9.0]])
        result = calculate_ignorance_score_native(
            y_true, y_pred, bins=[0, 4, 8, 12], low_bin=0, high_bin=12,
        )
        expected = np.log2(8.0 / 3.0)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_ap_oracle_sklearn(self):
        """AP matches sklearn.metrics.average_precision_score."""
        from sklearn.metrics import average_precision_score
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_pred = np.array([[0.9], [0.1], [0.8], [0.2]])
        result = calculate_ap_native(y_true, y_pred)
        # AP native repeats y_true for S columns, flattens y_pred
        expected = average_precision_score(
            np.repeat(y_true, 1), y_pred.flatten()
        )
        assert result == pytest.approx(expected, abs=1e-10)

    def test_coverage_all_inside(self):
        """All obs inside the central interval → coverage = 1.0."""
        y_true = np.array([5.0])
        y_pred = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])
        result = calculate_coverage_native(y_true, y_pred, alpha=0.1)
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_coverage_all_outside(self):
        """Obs far outside the interval → coverage = 0.0."""
        y_true = np.array([100.0])
        y_pred = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        result = calculate_coverage_native(y_true, y_pred, alpha=0.1)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_mis_obs_inside_interval(self):
        """Obs inside interval → MIS = interval width only (no penalty)."""
        y_true = np.array([5.0])
        y_pred = np.array([[0.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0]])
        alpha = 0.1
        lower = np.quantile([0, 2, 4, 5, 6, 8, 10], alpha / 2)
        upper = np.quantile([0, 2, 4, 5, 6, 8, 10], 1 - alpha / 2)
        expected = upper - lower  # no penalty since obs is inside
        result = calculate_mean_interval_score_native(y_true, y_pred, alpha=alpha)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_crps_point_prediction_equals_absolute_error(self):
        """CRPS of 1-member ensemble = |y - x|."""
        result = calculate_crps_native(np.array([5.0]), np.array([[8.0]]))
        assert result == pytest.approx(3.0, abs=1e-10)

    def test_twcrps_zero_threshold_equals_crps(self):
        """twCRPS with threshold=0 on non-negative data = CRPS."""
        y_true = np.array([5.0, 10.0])
        y_pred = np.array([[3.0, 7.0], [8.0, 12.0]])
        crps = calculate_crps_native(y_true, y_pred)
        twcrps = calculate_twcrps_native(y_true, y_pred, threshold=0.0)
        assert twcrps == pytest.approx(crps, abs=1e-10)

    def test_qis_symmetric_equals_mis(self):
        """QIS with symmetric quantiles (alpha/2, 1-alpha/2) equals MIS."""
        y_true = np.array([5.0, 15.0])
        y_pred = np.array([[1.0, 3.0, 5.0, 7.0, 9.0], [10.0, 12.0, 14.0, 16.0, 18.0]])
        alpha = 0.1
        mis = calculate_mean_interval_score_native(y_true, y_pred, alpha=alpha)
        qis = calculate_quantile_interval_score_native(
            y_true, y_pred, lower_quantile=alpha / 2, upper_quantile=1 - alpha / 2
        )
        assert qis == pytest.approx(mis, abs=1e-10)


# ---------------------------------------------------------------------------
# Green: Brier Score golden-value tests (ADR-020)
# ---------------------------------------------------------------------------

class TestBrierScore:

    def test_brier_rgs_sample_golden_value(self):
        """Hand-computed Brier rgs_sample: threshold=1, mixed binary outcomes."""
        y_true = np.array([0.0, 2.0, 5.0])
        y_pred = np.array([[0.5, 1.5], [0.5, 1.5], [4.0, 6.0]])
        # y_binary = [0, 1, 1] (0 < 1, 2 > 1, 5 > 1)
        # p_hat = [0.5, 0.5, 1.0] (fraction of ensemble > threshold)
        # Brier = mean([(0.5-0)^2, (0.5-1)^2, (1.0-1)^2]) = mean([0.25, 0.25, 0]) = 1/6
        result = calculate_brier_rgs_sample_native(y_true, y_pred, threshold=1.0)
        assert result == pytest.approx(1.0 / 6.0, abs=1e-10)

    def test_brier_cls_point_golden_value(self):
        """Hand-computed Brier cls_point: threshold=1, probabilities vs binary outcomes."""
        y_true = np.array([0.0, 2.0, 5.0])
        y_pred = np.array([[0.1], [0.7], [0.9]])
        # y_binary = [0, 1, 1]
        # p_hat = [0.1, 0.7, 0.9] (point prediction as probability)
        # Brier = mean([(0.1-0)^2, (0.7-1)^2, (0.9-1)^2]) = mean([0.01, 0.09, 0.01]) = 11/300
        result = calculate_brier_cls_point_native(y_true, y_pred, threshold=1.0)
        assert result == pytest.approx(11.0 / 300.0, abs=1e-10)

    def test_brier_cls_sample_golden_value(self):
        """Hand-computed Brier cls_sample: average probability samples, threshold=0."""
        y_true = np.array([1.0, 0.0])
        y_pred = np.array([[0.9, 0.8], [0.1, 0.2]])
        # y_binary = [1, 0] (1 > 0, 0 not > 0)
        # p_hat = [mean(0.9, 0.8), mean(0.1, 0.2)] = [0.85, 0.15]
        # Brier = mean([(0.85-1)^2, (0.15-0)^2]) = mean([0.0225, 0.0225]) = 0.0225
        result = calculate_brier_cls_sample_native(y_true, y_pred, threshold=0.0)
        assert result == pytest.approx(0.0225, abs=1e-10)

    def test_brier_rgs_sample_perfect(self):
        """All above threshold, all ensemble members above → p_hat=1, y_binary=1, Brier=0."""
        y_true = np.array([5.0, 10.0])
        y_pred = np.array([[2.0, 3.0], [2.0, 3.0]])
        result = calculate_brier_rgs_sample_native(y_true, y_pred, threshold=1.0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_brier_cls_point_perfect(self):
        """p_hat matches y_binary exactly → Brier=0."""
        y_true = np.array([0.0, 2.0])  # binary=[0, 1] at threshold=1
        y_pred = np.array([[0.0], [1.0]])  # perfect probability predictions
        result = calculate_brier_cls_point_native(y_true, y_pred, threshold=1.0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_brier_cls_sample_perfect(self):
        """Perfect probability samples → Brier=0."""
        y_true = np.array([1.0, 0.0])
        y_pred = np.array([[1.0, 1.0], [0.0, 0.0]])
        result = calculate_brier_cls_sample_native(y_true, y_pred, threshold=0.0)
        assert result == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Green: Quantile Score (pinball loss) golden-value tests (ADR-020)
# ---------------------------------------------------------------------------

class TestQuantileScore:

    def test_qs_sample_golden_value_at_median(self):
        """Median matches observation → QS = 0."""
        y_true = np.array([3.0])
        y_pred = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        # median of [1,2,3,4,5] = 3.0, diff = 3-3 = 0, QS = 0
        result = calculate_qs_sample_native(y_true, y_pred, quantile=0.5)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_qs_point_golden_value_overprediction(self):
        """Point overpredicts: y=3, q=5, quantile=0.9 → (1-0.9)*(5-3) = 0.2."""
        y_true = np.array([3.0])
        y_pred = np.array([[5.0]])
        # diff = 3 - 5 = -2 < 0 → branch: -diff * (1-quantile) = 2 * 0.1 = 0.2
        result = calculate_qs_point_native(y_true, y_pred, quantile=0.9)
        assert result == pytest.approx(0.2, abs=1e-10)

    def test_qs_sample_underprediction(self):
        """Sample underpredicts: y=10, q=2.0 at quantile=0.9 → 0.9*(10-2) = 7.2."""
        y_true = np.array([10.0])
        y_pred = np.array([[1.0, 2.0, 3.0]])
        # quantile(0.9) of [1,2,3] = 2.8 via linear interpolation
        q = np.quantile([1.0, 2.0, 3.0], 0.9)  # = 2.8
        expected = 0.9 * (10.0 - q)
        result = calculate_qs_sample_native(y_true, y_pred, quantile=0.9)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_qs_point_underprediction(self):
        """Point underpredicts: y=10, y_hat=2, quantile=0.9 → 0.9*(10-2) = 7.2."""
        y_true = np.array([10.0])
        y_pred = np.array([[2.0]])
        # diff = 10 - 2 = 8 ≥ 0 → branch: diff * quantile = 8 * 0.9 = 7.2
        result = calculate_qs_point_native(y_true, y_pred, quantile=0.9)
        assert result == pytest.approx(7.2, abs=1e-10)


# ---------------------------------------------------------------------------
# Beige: realistic edge cases (ADR-020)
# ---------------------------------------------------------------------------

class TestTwCRPSBeige:

    def test_single_observation(self):
        """twCRPS handles N=1, S=1 without error."""
        result = calculate_twcrps_native(np.array([1.0]), np.array([[1.0]]), threshold=0.0)
        assert np.isfinite(result)

    def test_large_ensemble_stable(self):
        """twCRPS is stable with S=1000 samples."""
        rng = np.random.default_rng(42)
        y_true = np.array([5.0, 10.0, 0.0])
        y_pred = rng.normal(loc=y_true[:, None], scale=1.0, size=(3, 1000))
        result = calculate_twcrps_native(y_true, y_pred, threshold=0.1)
        assert np.isfinite(result)
        assert result >= 0

    def test_threshold_at_exact_data_value(self):
        """twCRPS with threshold equal to an observation value — no crash."""
        y_true = np.array([5.0, 5.0])
        y_pred = np.array([[4.0, 6.0], [4.0, 6.0]])
        result = calculate_twcrps_native(y_true, y_pred, threshold=5.0)
        assert np.isfinite(result)

    def test_all_zero_with_positive_threshold(self):
        """All-zero data with τ > 0: both sides clamp to τ, so twCRPS = 0."""
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([[0.0], [0.0], [0.0]])
        result = calculate_twcrps_native(y_true, y_pred, threshold=10.0)
        assert result == pytest.approx(0.0, abs=1e-12)


class TestQISBeige:

    def test_single_observation(self):
        """QIS handles N=1 without error."""
        y_true = np.array([5.0])
        y_pred = np.array([[3.0, 4.0, 5.0, 6.0, 7.0]])
        result = calculate_quantile_interval_score_native(
            y_true, y_pred, lower_quantile=0.1, upper_quantile=0.9,
        )
        assert np.isfinite(result)

    def test_narrow_quantile_levels(self):
        """Very close quantile levels (0.49, 0.51) — narrow interval, finite result."""
        y_true = np.array([5.0, 10.0])
        y_pred = np.array([[4.0, 5.0, 6.0, 7.0, 8.0], [8.0, 9.0, 10.0, 11.0, 12.0]])
        result = calculate_quantile_interval_score_native(
            y_true, y_pred, lower_quantile=0.49, upper_quantile=0.51,
        )
        assert np.isfinite(result)
        assert result >= 0

    def test_identical_samples(self):
        """All ensemble members identical — interval width = 0, penalties if obs differs."""
        y_true = np.array([10.0])
        y_pred = np.array([[5.0, 5.0, 5.0, 5.0, 5.0]])
        result = calculate_quantile_interval_score_native(
            y_true, y_pred, lower_quantile=0.025, upper_quantile=0.975,
        )
        assert np.isfinite(result)
        assert result > 0  # obs outside collapsed interval → penalty


class TestMISBeige:

    def test_single_observation(self):
        """MIS handles N=1 without error."""
        y_true = np.array([5.0])
        y_pred = np.array([[3.0, 4.0, 5.0, 6.0, 7.0]])
        result = calculate_mean_interval_score_native(y_true, y_pred, alpha=0.05)
        assert np.isfinite(result)

    def test_identical_predictions(self):
        """All predictions identical — interval width = 0, penalty if obs differs."""
        y_true = np.array([10.0])
        y_pred = np.array([[5.0, 5.0, 5.0, 5.0, 5.0]])
        result = calculate_mean_interval_score_native(y_true, y_pred, alpha=0.05)
        assert np.isfinite(result)
        assert result > 0

    def test_small_alpha(self):
        """Alpha very close to 0 (wide interval) — large penalty factor but finite."""
        y_true = np.array([100.0])
        y_pred = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        result = calculate_mean_interval_score_native(y_true, y_pred, alpha=0.001)
        assert np.isfinite(result)
        assert result > 0

    def test_large_alpha(self):
        """Alpha close to 1 (nearly empty interval) — finite result."""
        y_true = np.array([3.0])
        y_pred = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        result = calculate_mean_interval_score_native(y_true, y_pred, alpha=0.99)
        assert np.isfinite(result)


class TestBrierScoreBeige:

    def test_rgs_single_observation(self):
        """Brier rgs_sample handles N=1, S=1 without error."""
        result = calculate_brier_rgs_sample_native(np.array([2.0]), np.array([[3.0]]), threshold=1.0)
        assert np.isfinite(result)

    def test_rgs_large_ensemble_stable(self):
        """Brier rgs_sample is stable with S=1000 samples."""
        rng = np.random.default_rng(42)
        y_true = np.array([0.0, 5.0, 10.0])
        y_pred = rng.normal(loc=y_true[:, None], scale=2.0, size=(3, 1000))
        result = calculate_brier_rgs_sample_native(y_true, y_pred, threshold=1.0)
        assert np.isfinite(result)
        assert 0 <= result <= 1  # Brier is bounded [0, 1]

    def test_rgs_threshold_at_exact_data_value(self):
        """Threshold equals an observation — no crash."""
        y_true = np.array([5.0, 5.0])
        y_pred = np.array([[4.0, 6.0], [4.0, 6.0]])
        result = calculate_brier_rgs_sample_native(y_true, y_pred, threshold=5.0)
        assert np.isfinite(result)

    def test_rgs_all_above_threshold(self):
        """All y_true above threshold — y_binary all 1, finite result."""
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([[0.5, 1.5], [0.5, 1.5]])
        result = calculate_brier_rgs_sample_native(y_true, y_pred, threshold=1.0)
        assert np.isfinite(result)

    def test_rgs_all_below_threshold(self):
        """All y_true below threshold — y_binary all 0, finite result."""
        y_true = np.array([0.0, 0.5])
        y_pred = np.array([[0.5, 1.5], [0.5, 1.5]])
        result = calculate_brier_rgs_sample_native(y_true, y_pred, threshold=1.0)
        assert np.isfinite(result)

    def test_cls_sample_large_ensemble_stable(self):
        """Brier cls_sample is stable with S=1000 probability samples."""
        rng = np.random.default_rng(42)
        y_true = np.array([1.0, 0.0, 1.0])
        y_pred = rng.beta(a=2, b=2, size=(3, 1000))  # probabilities in [0, 1]
        result = calculate_brier_cls_sample_native(y_true, y_pred, threshold=0.0)
        assert np.isfinite(result)
        assert 0 <= result <= 1


class TestQuantileScoreBeige:

    def test_single_observation(self):
        """QS handles N=1, S=1 without error."""
        result = calculate_qs_sample_native(np.array([1.0]), np.array([[1.0]]), quantile=0.5)
        assert np.isfinite(result)

    def test_large_ensemble_stable(self):
        """QS is stable with S=1000 samples."""
        rng = np.random.default_rng(42)
        y_true = np.array([5.0, 10.0, 0.0])
        y_pred = rng.normal(loc=y_true[:, None], scale=1.0, size=(3, 1000))
        result = calculate_qs_sample_native(y_true, y_pred, quantile=0.99)
        assert np.isfinite(result)
        assert result >= 0

    def test_extreme_quantile_near_one(self):
        """Quantile very close to 1 — finite result."""
        y_true = np.array([5.0])
        y_pred = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        result = calculate_qs_sample_native(y_true, y_pred, quantile=0.999)
        assert np.isfinite(result)

    def test_extreme_quantile_near_zero(self):
        """Quantile very close to 0 — finite result."""
        y_true = np.array([5.0])
        result = calculate_qs_point_native(y_true, np.array([[2.0]]), quantile=0.001)
        assert np.isfinite(result)


class TestMCRBeige:

    def test_single_observation(self):
        """MCR handles N=1, S=1 without error."""
        result = calculate_mcr_native(np.array([2.0]), np.array([[4.0]]))
        assert result == 2.0

    def test_near_zero_denominator(self):
        """Very small mean(y_true) — large but finite MCR."""
        y_true = np.array([1e-10, 0.0, 0.0])
        y_pred = np.array([[1.0], [1.0], [1.0]])
        result = calculate_mcr_native(y_true, y_pred)
        assert np.isfinite(result)
        assert result > 1e9  # massive overprediction ratio

    def test_negative_predictions(self):
        """Negative predictions produce valid (possibly negative) MCR."""
        y_true = np.array([1.0, 1.0])
        y_pred = np.array([[-1.0], [-1.0]])
        result = calculate_mcr_native(y_true, y_pred)
        assert result == -1.0


# ---------------------------------------------------------------------------
# Red: adversarial — must fail loud (ADR-020)
# ---------------------------------------------------------------------------

class TestSharedRed:
    """Adversarial tests that apply to all metrics via _guard_shapes."""

    def test_shape_3d_raises(self):
        """3D y_pred should raise ValueError from _guard_shapes."""
        y_true = np.array([1.0, 2.0])
        y_pred = np.ones((2, 3, 4))
        with pytest.raises(ValueError, match="y_pred must be 2D"):
            calculate_twcrps_native(y_true, y_pred, threshold=0.0)

    def test_shape_mismatch_raises(self):
        """Row mismatch between y_true and y_pred raises ValueError."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([[1.0], [2.0]])  # only 2 rows
        with pytest.raises(ValueError, match="Row mismatch"):
            calculate_twcrps_native(y_true, y_pred, threshold=0.0)
        with pytest.raises(ValueError, match="Row mismatch"):
            calculate_mcr_native(y_true, y_pred)
        with pytest.raises(ValueError, match="Row mismatch"):
            calculate_mean_interval_score_native(y_true, y_pred, alpha=0.05)
        with pytest.raises(ValueError, match="Row mismatch"):
            calculate_quantile_interval_score_native(
                y_true, y_pred, lower_quantile=0.025, upper_quantile=0.975,
            )


class TestTwCRPSRed:

    def test_nan_in_y_true_propagates(self):
        """NaN in y_true propagates to result (not silently ignored)."""
        y_true = np.array([np.nan, 1.0])
        y_pred = np.array([[1.0], [1.0]])
        result = calculate_twcrps_native(y_true, y_pred, threshold=0.0)
        assert np.isnan(result)

    def test_nan_in_y_pred_propagates(self):
        """NaN in y_pred propagates to result."""
        y_true = np.array([1.0, 1.0])
        y_pred = np.array([[np.nan], [1.0]])
        result = calculate_twcrps_native(y_true, y_pred, threshold=0.0)
        assert np.isnan(result)

    def test_negative_threshold_accepted(self):
        """Negative threshold is mathematically valid (clamps below τ)."""
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([[1.0, 2.0], [2.0, 3.0]])
        result = calculate_twcrps_native(y_true, y_pred, threshold=-5.0)
        assert np.isfinite(result)


class TestQISRed:

    def test_lower_ge_upper_swapped(self):
        """lower_quantile > upper_quantile swaps the interval — result is still a number."""
        y_true = np.array([5.0])
        y_pred = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])
        # np.quantile with lower > upper still returns values — documents behavior
        result = calculate_quantile_interval_score_native(
            y_true, y_pred, lower_quantile=0.9, upper_quantile=0.1,
        )
        assert np.isfinite(result)  # no crash — behavior is defined but meaningless

    def test_quantile_outside_range_raises(self):
        """Quantile > 1 raises from np.quantile."""
        y_true = np.array([5.0])
        y_pred = np.array([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError):
            calculate_quantile_interval_score_native(
                y_true, y_pred, lower_quantile=0.025, upper_quantile=1.5,
            )

    def test_nan_in_y_true_propagates(self):
        """NaN in y_true propagates to result."""
        y_true = np.array([np.nan])
        y_pred = np.array([[1.0, 2.0, 3.0]])
        result = calculate_quantile_interval_score_native(
            y_true, y_pred, lower_quantile=0.025, upper_quantile=0.975,
        )
        assert np.isnan(result)


class TestMISRed:

    def test_alpha_zero_raises(self):
        """Alpha = 0 causes ZeroDivisionError in penalty (2/alpha)."""
        y_true = np.array([100.0])
        y_pred = np.array([[1.0, 2.0, 3.0]])
        with pytest.raises(ZeroDivisionError):
            calculate_mean_interval_score_native(y_true, y_pred, alpha=0.0)

    def test_nan_in_y_true_propagates(self):
        """NaN in y_true propagates to result."""
        y_true = np.array([np.nan])
        y_pred = np.array([[1.0, 2.0, 3.0]])
        result = calculate_mean_interval_score_native(y_true, y_pred, alpha=0.05)
        assert np.isnan(result)


class TestMCRRed:

    def test_nan_in_y_true_propagates(self):
        """NaN in y_true propagates to result."""
        y_true = np.array([np.nan, 1.0])
        y_pred = np.array([[1.0], [1.0]])
        result = calculate_mcr_native(y_true, y_pred)
        assert np.isnan(result)

    def test_nan_in_y_pred_propagates(self):
        """NaN in y_pred propagates to result."""
        y_true = np.array([1.0, 1.0])
        y_pred = np.array([[np.nan], [1.0]])
        result = calculate_mcr_native(y_true, y_pred)
        assert np.isnan(result)

    def test_negative_y_true_valid(self):
        """Negative y_true is mathematically valid — MCR can be negative."""
        y_true = np.array([-2.0, -2.0])
        y_pred = np.array([[4.0], [4.0]])
        result = calculate_mcr_native(y_true, y_pred)
        assert result == -2.0


class TestBrierScoreRed:

    def test_rgs_nan_in_y_true_swallowed_by_comparison(self):
        """NaN in y_true is swallowed by '>' comparison (NaN > x → False).

        Unlike arithmetic metrics, Brier's binarization step converts NaN to
        False (0.0) rather than propagating. This is NumPy's standard comparison
        semantics. The EvaluationFrame boundary should reject NaN before it
        reaches here (defense-in-depth).
        """
        y_true = np.array([np.nan, 1.0])
        y_pred = np.array([[1.0], [1.0]])
        result = calculate_brier_rgs_sample_native(y_true, y_pred, threshold=1.0)
        assert np.isfinite(result)

    def test_rgs_nan_in_y_pred_swallowed_by_comparison(self):
        """NaN in y_pred is swallowed by '>' comparison in p_hat computation."""
        y_true = np.array([1.0, 1.0])
        y_pred = np.array([[np.nan], [1.0]])
        result = calculate_brier_rgs_sample_native(y_true, y_pred, threshold=1.0)
        assert np.isfinite(result)

    def test_rgs_negative_threshold_accepted(self):
        """Negative threshold is mathematically valid for regression Brier."""
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([[1.0, 2.0], [2.0, 3.0]])
        result = calculate_brier_rgs_sample_native(y_true, y_pred, threshold=-5.0)
        assert np.isfinite(result)

    def test_cls_sample_nan_in_y_pred_propagates(self):
        """NaN in probability samples propagates via mean() — not swallowed."""
        y_true = np.array([1.0, 1.0])
        y_pred = np.array([[np.nan], [0.5]])
        result = calculate_brier_cls_sample_native(y_true, y_pred, threshold=0.0)
        # mean([nan]) = nan, (nan - 1)^2 = nan → result is nan
        assert np.isnan(result)


class TestQuantileScoreRed:

    def test_nan_in_y_true_propagates(self):
        """NaN in y_true propagates to result."""
        y_true = np.array([np.nan, 1.0])
        y_pred = np.array([[1.0], [1.0]])
        result = calculate_qs_sample_native(y_true, y_pred, quantile=0.5)
        assert np.isnan(result)

    def test_nan_in_y_pred_propagates(self):
        """NaN in y_pred propagates to result."""
        y_true = np.array([1.0, 1.0])
        y_pred = np.array([[np.nan], [1.0]])
        result = calculate_qs_point_native(y_true, y_pred, quantile=0.5)
        assert np.isnan(result)


# ---------------------------------------------------------------------------
# Red: Extreme-value tests (ADR-020)
# ---------------------------------------------------------------------------

class TestExtremeValues:
    """Test metric behavior near float64 limits — no overflow, no silent corruption."""

    def test_mse_large_matching_values(self):
        """Large but equal values → MSE = 0, not overflow."""
        result = calculate_mse_native(np.array([1e150]), np.array([[1e150]]))
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_crps_large_ensemble_values(self):
        """CRPS with large ensemble values remains finite."""
        y_true = np.array([1e50])
        y_pred = np.array([[0.9e50, 1.0e50, 1.1e50]])
        result = calculate_crps_native(y_true, y_pred)
        assert np.isfinite(result)
        assert result >= 0

    def test_brier_rgs_extreme_threshold(self):
        """Threshold at 1e300: all values below → y_binary all 0, p_hat all 0, Brier = 0."""
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([[0.5, 1.5], [0.5, 1.5]])
        result = calculate_brier_rgs_sample_native(y_true, y_pred, threshold=1e300)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_coverage_tiny_ensemble_spread(self):
        """Extremely narrow ensemble → interval width ~ 0, coverage depends on obs position."""
        base = 1e-15
        y_true = np.array([base])
        y_pred = np.array([[base - 1e-30, base + 1e-30]])
        result = calculate_coverage_native(y_true, y_pred, alpha=0.1)
        assert np.isfinite(result)


# ---------------------------------------------------------------------------
# Ignorance bin-range contract (ADR-015 ruling 8; risk register C-27, C-28a)
#
# RED   — observations outside the configured bins must raise, at BOTH tails.
# BEIGE — boundary behaviour at the edges themselves, pinned rather than incidental.
# ---------------------------------------------------------------------------

class TestIgnoranceBinRangeRed:

    def _profile(self):
        from views_evaluation.profiles.base import BASE_PROFILE
        return BASE_PROFILE["Ignorance"]

    def test_observation_above_top_edge_raises(self):
        """C-27: previously an IndexError on ordinary country-month fatality counts.

        BASE_PROFILE's top bin edge is 1000.5; monthly country-level counts exceed
        that routinely, so this was on the production path, not a corner case.
        """
        y_true = np.array([2000.0])
        y_pred = np.array([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError, match="outside the configured bin range"):
            calculate_ignorance_score_native(y_true, y_pred, **self._profile())

    def test_observation_below_bottom_edge_raises(self):
        """C-28a: previously negative-indexed into the LAST bin and returned 3.7004."""
        y_true = np.array([-1.0])
        y_pred = np.array([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError, match="outside the configured bin range"):
            calculate_ignorance_score_native(y_true, y_pred, **self._profile())

    def test_above_range_raises_even_when_a_prediction_is_also_out_of_range(self):
        """The old crash was masked when a prediction also exceeded the range.

        bincount(minlength=n_bins) grew the array, making the out-of-bounds index
        accidentally valid. The observation is still outside the configured bins,
        so it must raise either way.
        """
        y_true = np.array([2000.0])
        y_pred = np.array([[1.0, 2.0, 5000.0]])
        with pytest.raises(ValueError, match="outside the configured bin range"):
            calculate_ignorance_score_native(y_true, y_pred, **self._profile())

    def test_error_message_names_value_and_range(self):
        y_true = np.array([2000.0])
        y_pred = np.array([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError) as excinfo:
            calculate_ignorance_score_native(y_true, y_pred, **self._profile())
        msg = str(excinfo.value)
        assert "2000.0" in msg, "message must name the offending observation"
        assert "1000.5" in msg, "message must name the configured range"
        assert "bins" in msg, "message must tell the caller what to change"


class TestIgnoranceBinRangeBeige:

    def _profile(self):
        from views_evaluation.profiles.base import BASE_PROFILE
        return BASE_PROFILE["Ignorance"]

    def test_observation_on_bottom_edge_is_in_range(self):
        """Bins are half-open [lo, hi): the bottom edge belongs to the first bin."""
        y_true = np.array([0.0])
        y_pred = np.array([[0.0, 1.0, 2.0]])
        result = calculate_ignorance_score_native(y_true, y_pred, **self._profile())
        assert np.isfinite(result)

    def test_observation_on_top_edge_raises(self):
        """Half-open range: the top edge is in no bin, so it is out of range."""
        y_true = np.array([1000.5])
        y_pred = np.array([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError, match="outside the configured bin range"):
            calculate_ignorance_score_native(y_true, y_pred, **self._profile())

    def test_in_range_value_is_unchanged_by_the_guard(self):
        """The guard must not alter any previously-correct result."""
        y_true = np.array([3.0])
        y_pred = np.array([[1.0, 2.0, 3.0]])
        result = calculate_ignorance_score_native(y_true, y_pred, **self._profile())
        assert np.isfinite(result) and result > 0


# ---------------------------------------------------------------------------
# Pearson constant-input contract (ADR-015 ruling 2; risk register C-22)
#
# Contrast with MCR (TestMCR*), whose inf/nan IS retained as a documented sentinel:
# MCR's inf is a real answer ("predicted conflict where none occurred"); Pearson's
# nan is the absence of one. That asymmetry is the whole of ADR-015's exception test.
# ---------------------------------------------------------------------------

class TestPearsonConstantInputBeige:
    """`Pearson` returns NaN on constant input — a documented sentinel (ADR-015 R2).

    Same category as MCR's inf/nan: a constant series is a property of the data, not a
    broken invariant. These asserted a raise for one day; the ruling was reversed when
    it emerged that it aborted any evaluation of a constant baseline (ADR-041 workflow).
    """

    def test_constant_y_true_returns_nan(self):
        y_true = np.array([1.0, 1.0, 1.0])
        y_pred = np.array([[1.0], [2.0], [3.0]])
        assert np.isnan(calculate_pearson_native(y_true, y_pred))

    def test_constant_y_pred_returns_nan(self):
        """The baseline-model case: constant predictions, varied truth."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([[5.0], [5.0], [5.0]])
        assert np.isnan(calculate_pearson_native(y_true, y_pred))

    def test_no_constant_input_warning_escapes(self):
        """The case is contracted and handled, so SciPy's warning is noise.

        A ConstantInputWarning reaching the runner would mean the degenerate case is
        being encountered rather than handled (ADR-015, Validation & Monitoring).
        """
        import warnings as _w
        y_true = np.array([1.0, 1.0, 1.0])
        y_pred = np.array([[1.0], [2.0], [3.0]])
        with _w.catch_warnings():
            _w.simplefilter("error")      # any escaping warning becomes an exception
            assert np.isnan(calculate_pearson_native(y_true, y_pred))


class TestPearsonGreen:

    def test_non_constant_input_is_unchanged(self):
        """Perfect positive correlation still returns 1.0 — the guard changes nothing."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([[1.0], [2.0], [3.0]])
        assert calculate_pearson_native(y_true, y_pred) == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([[3.0], [2.0], [1.0]])
        assert calculate_pearson_native(y_true, y_pred) == pytest.approx(-1.0)


class TestMeanPredictionShapeGuardRed:
    """C-32: y_hat_bar must guard like every sibling kernel, not succeed silently."""

    def test_row_mismatch_raises(self):
        from views_evaluation.evaluation.native_metric_calculators import (
            calculate_mean_prediction_native,
        )
        with pytest.raises(ValueError, match="Row mismatch"):
            calculate_mean_prediction_native(np.zeros(4), np.zeros((7, 3)))

    def test_two_dimensional_y_true_raises(self):
        from views_evaluation.evaluation.native_metric_calculators import (
            calculate_mean_prediction_native,
        )
        with pytest.raises(ValueError, match="y_true must be 1D"):
            calculate_mean_prediction_native(np.zeros((4, 2)), np.zeros((4, 3)))

    def test_valid_input_value_is_unchanged(self):
        from views_evaluation.evaluation.native_metric_calculators import (
            calculate_mean_prediction_native,
        )
        y_pred = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert calculate_mean_prediction_native(np.zeros(2), y_pred) == pytest.approx(2.5)
