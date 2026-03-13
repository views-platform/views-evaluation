import pytest
import numpy as np
import pandas as pd
from views_evaluation.evaluation.native_metric_calculators import (
    calculate_mse_native,
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
    REGRESSION_POINT_NATIVE,
    REGRESSION_SAMPLE_NATIVE,
    CLASSIFICATION_POINT_NATIVE,
    CLASSIFICATION_SAMPLE_NATIVE,
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    actual = pd.DataFrame({
        'target': [[1.0], [2.0], [3.0], [4.0]]
    })
    pred = pd.DataFrame({
        'pred_target': [[1.1], [1.9], [3.1], [3.9]]
    })
    return actual, pred


@pytest.fixture
def sample_sample_data():
    """Create sample sample data for testing."""
    actual = pd.DataFrame({
        'target': [[1.0], [2.0], [3.0], [4.0]]
    })
    pred = pd.DataFrame({
        'pred_target': [[1.0, 1.1, 1.2], [1.8, 2.0, 2.2], [2.9, 3.0, 3.1], [3.8, 4.0, 4.2]]
    })
    return actual, pred


def test_calculate_mse_native(sample_data):
    """Test MSE calculation."""
    actual, pred = sample_data
    result = calculate_mse_native(actual, pred, 'target')
    assert isinstance(result, float)
    assert result >= 0

def test_calculate_rmsle_native_point(sample_data):
    """Test RMSLE calculation."""
    actual, pred = sample_data
    result = calculate_rmsle_native(actual, pred, 'target')
    assert isinstance(result, float)
    assert result >= 0

def test_calculate_crps_native_point(sample_data):
    """Test CRPS calculation."""
    actual, pred = sample_data
    result = calculate_crps_native(actual, pred, 'target')
    assert isinstance(result, float)
    assert result >= 0


def test_calculate_crps_native_sample(sample_sample_data):
    """Test CRPS calculation."""
    actual, pred = sample_sample_data
    result = calculate_crps_native(actual, pred, 'target')
    assert isinstance(result, float)
    assert result >= 0


def test_calculate_ap_native():
    """Test Average Precision calculation with pre-binarised actuals and probability scores."""
    # Binary actuals (0/1) and probability scores as predictions
    actual = pd.DataFrame({'target': [[1], [0], [1], [0]]})
    pred = pd.DataFrame({'pred_target': [[0.9], [0.4], [0.3], [0.1]]})
    result = calculate_ap_native(actual, pred, 'target')
    assert isinstance(result, float)
    assert 0 <= result <= 1


def test_calculate_emd_native(sample_data):
    """Test Earth Mover's Distance calculation."""
    actual, pred = sample_data
    result = calculate_emd_native(actual, pred, 'target')
    assert isinstance(result, float)
    assert result >= 0


def test_calculate_pearson_native(sample_data):
    """Test Pearson correlation calculation."""
    actual, pred = sample_data
    result = calculate_pearson_native(actual, pred, 'target')
    assert isinstance(result, float)
    assert -1 <= result <= 1


def test_calculate_mtd_native(sample_data):
    """Test Mean Tweedie Deviance calculation."""
    actual, pred = sample_data
    result = calculate_mtd_native(actual, pred, 'target', power=1.5)
    assert isinstance(result, float)
    assert result >= 0


def test_calculate_mtd_native_with_power(sample_data):
    """Test Mean Tweedie Deviance calculation with different power values."""
    actual, pred = sample_data
    # Test with power=1.5 (compound Poisson-Gamma)
    result_15 = calculate_mtd_native(actual, pred, 'target', power=1.5)
    assert isinstance(result_15, float)
    assert result_15 >= 0

    # Test with power=2 (Gamma)
    result_2 = calculate_mtd_native(actual, pred, 'target', power=2.0)
    assert isinstance(result_2, float)
    assert result_2 >= 0


def test_calculate_coverage_native_sample(sample_sample_data):
    """Test Coverage calculation."""
    actual, pred = sample_sample_data
    result = calculate_coverage_native(actual, pred, 'target', alpha=0.1)
    assert isinstance(result, float)
    assert 0 <= result <= 1


def test_calculate_ignorance_score_native_sample(sample_sample_data):
    """Test Ignorance Score calculation."""
    actual, pred = sample_sample_data
    result = calculate_ignorance_score_native(
        actual, pred, 'target',
        bins=[0, 0.5, 2.5, 5.5, 10.5, 25.5, 50.5, 100.5, 250.5, 500.5, 1000.5],
        low_bin=0, high_bin=10000,
    )
    assert isinstance(result, float)
    assert result >= 0


def test_calculate_mis_sample(sample_sample_data):
    """Test Mean Interval Score calculation."""
    actual, pred = sample_sample_data
    result = calculate_mean_interval_score_native(actual, pred, 'target', alpha=0.05)
    assert isinstance(result, float)
    assert result >= 0


def test_point_metric_functions():
    """Test that all point metric functions are available in the deprecated REGRESSION_POINT_NATIVE."""
    expected_metrics = [
        "MSE", "MSLE", "RMSLE", "EMD", "SD", "pEMDiv", "Pearson", "Variogram", "MTD", "y_hat_bar"
    ]


    for metric in expected_metrics:
        assert metric in REGRESSION_POINT_NATIVE
        assert callable(REGRESSION_POINT_NATIVE[metric])


def test_sample_metric_functions():
    """Test that all sample metric functions are available in the deprecated REGRESSION_SAMPLE_NATIVE."""
    expected_metrics = ["CRPS", "twCRPS", "MIS", "QIS", "Ignorance", "Brier", "Jeffreys", "Coverage"]

    for metric in expected_metrics:
        assert metric in REGRESSION_SAMPLE_NATIVE
        assert callable(REGRESSION_SAMPLE_NATIVE[metric])


def test_regression_point_metric_functions():
    """Test that all regression point metric functions are available in REGRESSION_POINT_NATIVE."""
    expected_metrics = ["MSE", "MSLE", "RMSLE", "EMD", "SD", "pEMDiv", "Pearson", "Variogram", "MTD", "y_hat_bar"]

    for metric in expected_metrics:
        assert metric in REGRESSION_POINT_NATIVE
        assert callable(REGRESSION_POINT_NATIVE[metric])

    # AP must NOT be in regression point functions
    assert "AP" not in REGRESSION_POINT_NATIVE
    # CRPS must NOT be in regression point functions
    assert "CRPS" not in REGRESSION_POINT_NATIVE


def test_regression_sample_metric_functions():
    """Test that all regression sample metric functions are available."""
    expected_metrics = ["CRPS", "twCRPS", "MIS", "QIS", "Coverage", "Ignorance", "y_hat_bar"]

    for metric in expected_metrics:
        assert metric in REGRESSION_SAMPLE_NATIVE
        assert callable(REGRESSION_SAMPLE_NATIVE[metric])

    # AP must NOT be in regression sample functions
    assert "AP" not in REGRESSION_SAMPLE_NATIVE


def test_classification_point_metric_functions():
    """Test that AP is in CLASSIFICATION_POINT_NATIVE."""
    assert "AP" in CLASSIFICATION_POINT_NATIVE
    assert callable(CLASSIFICATION_POINT_NATIVE["AP"])

    # RMSLE must NOT be in classification point functions
    assert "RMSLE" not in CLASSIFICATION_POINT_NATIVE


def test_classification_sample_metric_functions():
    """Test that classification sample metric functions are available."""
    expected_metrics = ["CRPS", "twCRPS", "Brier", "Jeffreys"]

    for metric in expected_metrics:
        assert metric in CLASSIFICATION_SAMPLE_NATIVE
        assert callable(CLASSIFICATION_SAMPLE_NATIVE[metric])

    # RMSLE must NOT be in classification sample functions
    assert "RMSLE" not in CLASSIFICATION_SAMPLE_NATIVE


def test_not_implemented_metrics():
    """Test that unimplemented metrics raise NotImplementedError."""
    actual = pd.DataFrame({'target': [[1.0]]})
    pred = pd.DataFrame({'pred_target': [[1.0]]})

    from views_evaluation.evaluation.native_metric_calculators import (
        calculate_brier_native,
        calculate_jeffreys_native,
        calculate_sd_native,
        calculate_pEMDiv_native,
        calculate_variogram_native,
    )

    unimplemented_functions = [
        calculate_brier_native,
        calculate_jeffreys_native,
        calculate_sd_native,
        calculate_pEMDiv_native,
        calculate_variogram_native,
    ]

    for func in unimplemented_functions:
        with pytest.raises(ValueError, match="not yet implemented"):
            func(actual, pred, 'target')


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
# twCRPS tests
# ---------------------------------------------------------------------------

class TestTwCRPS:

    def test_twcrps_basic_smoke(self, sample_sample_data):
        """twCRPS produces a non-negative float."""
        actual, pred = sample_sample_data
        result = calculate_twcrps_native(actual, pred, 'target', threshold=0.0)
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

    def test_twcrps_in_dispatch_dicts(self):
        """twCRPS must be in both regression and classification sample dispatch dicts."""
        assert "twCRPS" in REGRESSION_SAMPLE_NATIVE
        assert "twCRPS" in CLASSIFICATION_SAMPLE_NATIVE
        assert callable(REGRESSION_SAMPLE_NATIVE["twCRPS"])


# ---------------------------------------------------------------------------
# Quantile Interval Score tests
# ---------------------------------------------------------------------------

class TestQuantileIntervalScore:

    def test_qis_basic_smoke(self, sample_sample_data):
        """QIS produces a non-negative float."""
        actual, pred = sample_sample_data
        result = calculate_quantile_interval_score_native(
            actual, pred, 'target', lower_quantile=0.025, upper_quantile=0.975,
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

        lower = float(np.quantile(y_pred, q_lo, axis=1))
        upper = float(np.quantile(y_pred, q_hi, axis=1))

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

    def test_qis_in_dispatch_dict(self):
        """QIS must be in regression sample dispatch dict."""
        assert "QIS" in REGRESSION_SAMPLE_NATIVE
        assert callable(REGRESSION_SAMPLE_NATIVE["QIS"])


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
