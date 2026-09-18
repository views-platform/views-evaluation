from typing import Optional
from dataclasses import dataclass


@dataclass
class BaseEvaluationMetrics:
    """Common base of the four typed metric containers below.

    Until 2.0.0 it also carried three ``make_*_evaluation_dict`` factories (no callers)
    and ``evaluation_dict_to_dataframe`` (the DataFrame bridge, removed with
    ``EvaluationReport.to_dataframe()``, both gone in 2.0.0; register C-40). The
    containers are what ``EvaluationReport.get_schema_results()`` returns.
    """


# ---------------------------------------------------------------------------
# 2×2 dataclasses: {regression, classification} × {point, sample}
# ---------------------------------------------------------------------------

@dataclass
class RegressionPointEvaluationMetrics(BaseEvaluationMetrics):
    """Metrics for regression targets evaluated with point predictions."""
    MSE:       Optional[float] = None
    MSLE:      Optional[float] = None
    RMSLE:     Optional[float] = None
    EMD:       Optional[float] = None
    SD:        Optional[float] = None
    pEMDiv:    Optional[float] = None
    Pearson:   Optional[float] = None
    Variogram: Optional[float] = None
    MTD:       Optional[float] = None
    y_hat_bar: Optional[float] = None
    MCR_point: Optional[float] = None
    QS_point:  Optional[float] = None


@dataclass
class RegressionSampleEvaluationMetrics(BaseEvaluationMetrics):
    """Metrics for regression targets evaluated with sample-based predictions."""
    CRPS:       Optional[float] = None
    twCRPS:     Optional[float] = None
    MIS:        Optional[float] = None
    QIS:        Optional[float] = None
    QS_sample:  Optional[float] = None
    Coverage:   Optional[float] = None
    Ignorance:  Optional[float] = None
    y_hat_bar:  Optional[float] = None
    MCR_sample:      Optional[float] = None
    Brier_rgs_sample: Optional[float] = None


@dataclass
class ClassificationPointEvaluationMetrics(BaseEvaluationMetrics):
    """Metrics for classification targets evaluated with point (probability) predictions."""
    AP:              Optional[float] = None
    Brier_cls_point: Optional[float] = None


@dataclass
class ClassificationSampleEvaluationMetrics(BaseEvaluationMetrics):
    """Metrics for classification targets evaluated with sample-based predictions."""
    CRPS:              Optional[float] = None
    twCRPS:            Optional[float] = None
    Brier_cls_sample:  Optional[float] = None
    Jeffreys:          Optional[float] = None
