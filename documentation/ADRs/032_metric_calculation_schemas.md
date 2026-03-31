# ADR-032: Metric Calculation Schemas

**Status:** Accepted  
**Date:** 2024-10-31  
**Deciders:** Mihai, Xiaolong  
**Consulted:** —  
**Informed:** All contributors  

## Context
Traditional machine learning metrics do not directly translate to time-series forecasting across multiple horizons. A standardized approach to regrouping data is necessary.

## Decision
Evaluation metrics are computed in three different ways (schemas) to capture various aspects of performance:

### 1. Time-series-wise Evaluation
Evaluating *along the sequence*. Metrics are computed per 36-month sequence. This shows the average predictive power of a specific model run. This is the standard approach in packages like `darts` or `prophet`.

### 2. Step-wise Evaluation
Grouping predictions by their *lead time* (step). Predictions from all sequences for Step 1 are grouped, all for Step 2, etc. This verifies which models predict best at short-term vs. long-term horizons.

### 3. Month-wise Evaluation
Grouping predictions by their *calendar month*. All predictions made for "January 2020" (from various origins) are compared against the actuals for that month. This is useful for accounting for rare events (e.g., 9/11).

## Rationale
A multi-faceted approach ensures that both short-term and long-term predictive performance is assessed, reflecting real-world application where model accuracy varies across the forecast window.
