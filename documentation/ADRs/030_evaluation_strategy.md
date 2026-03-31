# ADR-030: Evaluation Strategy

**Status:** Accepted  
**Date:** 2025-07-16  
**Deciders:** Xiaolong, Mihai  
**Consulted:** —  
**Informed:** All contributors  

## Context
To ensure reliable and realistic model performance assessment, our forecasting framework supports both **offline** and **online** evaluation strategies. These strategies serve complementary purposes: offline evaluation simulates the forecasting process retrospectively, while online evaluation assesses actual deployed forecasts against observed data.

## Decision
We adopt a dual evaluation approach consisting of:
1. **Offline Evaluation:** Evaluating a model's performance on historical data, before deployment.
2. **Online Evaluation:** The ongoing process of evaluating a deployed model's performance as new, real-world data becomes available.

### Points of Definition: 
- **Rolling-Origin Holdout:** A robust backtesting strategy that simulates a real-world forecasting scenario by generating forecasts from multiple, rolling time origins.
- **Forecast Steps:** The time increment between predictions within a sequence of forecasts (further referred to as steps).
- **Sequence:** An ordered set of data points indexed by time. 

### Offline Evaluation
We adopt a **rolling-origin holdout evaluation strategy** for all offline (backtesting) evaluations.
The offline evaluation strategy involves:
1. **A single model** is trained on historical data up to training cutoff $H_0$.
2. Using this trained model object, a forecast is generated for the next **36 months**.
3. The origin is then rolled forward by one month, and another forecast is generated.
4. This process continues until a fixed number of sequences **k** is reached (standardized to k=12).

### Online Evaluation
Online evaluation reflects **true forecasting** and is based on the **Forecasting Partition**. It evaluates all forecasts made for a specific month from all available historical sequences.

## Consequences

### Positive Effects
- Reflects realistic deployment and monitoring conditions.
- Improves robustness by capturing temporal variation in model performance.

### Negative Effects
- Requires careful alignment of sequences and forecast windows.
- Models must be capable of generalizing across slightly shifted input windows.

## Rationale
The dual evaluation setup strikes a balance between experimentation and real-world monitoring. For further technical details see ADR-040 and ADR-032.
