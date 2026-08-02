# ADR-031: Evaluation Metrics

**Status:** Accepted  
**Date:** 2024-09-12  
**Deciders:** Xiaolong  
**Consulted:** —  
**Informed:** All contributors  

## Context
In the context of the VIEWS pipeline, it is necessary to evaluate the models using a robust set of metrics that account for the characteristics of conflict data, such as right-skewness and zero-inflation in the outcome variable.

## Decision
> **Implementation status (verified 2026-08-02).** `METRIC_CATALOG` in
> `views_evaluation/evaluation/metric_catalog.py` is the authoritative source — each
> `MetricSpec` carries an `implemented` flag, and an unimplemented metric raises if
> requested.
> - **Not implemented:** `SD` (Sinkhorn Distance), `Variogram`, `pEMDiv`, `Jeffreys`.
> - **Brier Score IS implemented** as of 2026-04-09, split into three task-explicit
>   variants: `Brier_cls_point`, `Brier_cls_sample`, `Brier_rgs_sample`. The `_cls_`/
>   `_rgs_` infix denotes classification vs. regression context. `Brier_rgs_point` is
>   deliberately absent — a regression point estimate is not a probability.
>
> _(This note previously listed Brier Score as unimplemented and omitted `pEMDiv`
> entirely — wrong in both directions, and stale for four months. Corrected 2026-08-02;
> the detection gap that allowed it is register **C-34**.)_

Below are the evaluation metrics used to assess the performance of models in the VIEWS pipeline:

| Metric                              | Abbreviation          | Task             | Notes                                                                            |
|-------------------------------------|-----------------------|------------------|------------------------------------------------------------------------------------------------------------|
| Continuous Ranked Probability Score | CRPS                  | Probabilistic    | Measures the difference between predicted and observed cumulative distributions                             |
| Brier Score                         | Brier                 | Probabilistic    | Evaluates the accuracy of predicted probabilities by comparing them to actual outcomes                    |
| Jeffreys Divergence                 | JD                    | Probabilistic    | Measures the divergence between two probability distributions                                               |
| Coverage (Histograms)               | -                     | Probabilistic    | Histogram-based measure of prediction coverage                                                             |
| Sinkhorn/Earth-mover Distance       | Sinkhorn/EMD          | Probabilistic    | Measures the difference between distributions via transformation cost.                                     |
| Variogram                           | -                     | Probabilistic    | Measures spatial dependence.                                                                               |
| Average Precision                   | AP                    | Classification   | Measures the precision-recall trade-off in classification tasks.                                           |
| Root Mean Squared Logarithmic Error | RMSLE                 | Regression       | Evaluates error for skewed data.                                                                           |
| Pearson                             | -                     | Regression       | Evaluates linear correlation.                                                                              |

## Rationale
The selected metrics are designed to address the unique characteristics of conflict prediction data (zero-inflated, right-skewed). Using a mix of probabilistic and point-based metrics allows us to better capture outcomes and focus on critical shifts (onsets).

### Considerations
- **Onset Sensitivity**: Ability to detect onsets of conflict is prioritized.
- **Skewed Data**: Traditional error metrics like MSE may favor models predicting zeros too often.
