# Current Alignment & Regrouping Semantics

**Date:** 2026-02-25
**Status:** Investigation Artifact

## 1. Overview

The Views Evaluation repository currently operates on a **List-of-DataFrames** abstraction. This structure implicitly encodes the rolling-origin nature of forecasting. This document details the exact mechanisms used for alignment, duplication, and regrouping, which must be preserved or formally replaced in the new `EvaluationFrame` architecture.

## 2. Input Data Structure

*   **Actuals**: A single `pd.DataFrame` indexed by `MultiIndex` (typically `['month_id', 'country_id']`).
*   **Predictions**: A `List[pd.DataFrame]`.
    *   Each DataFrame in the list represents a **Forecast Sequence** (predictions generated from a single model run/origin).
    *   Each DataFrame is indexed by `MultiIndex` (matching Actuals).
    *   **Implicit Invariant**: The rows in each Prediction DataFrame are sorted by time (step 1, step 2, ...).

## 3. Alignment Logic (`_match_actual_pred`)

This method is the core "Join" operation. It is called repeatedly—once for every DataFrame in the `predictions` list.

1.  **Input**: Single `actual` DF, Single `pred` DF.
2.  **Intersection**: `common_indices = actual.index.intersection(pred.index)`
3.  **Filtering**: `pred` is filtered to `common_indices`.
4.  **Reindexing (Duplication)**: `actual` is `reindex`ed to `pred.index`.
    *   **Critical Behavior**: If the `predictions` list contains overlapping time periods (e.g., `pred[0]` covers Jan-Mar, `pred[1]` covers Feb-Apr), the actual values for Feb and Mar are **fetched twice**.
    *   This duplication is intentional and necessary for "Rolling Origin" evaluation (evaluating how well we predicted Feb from Jan, and how well we predicted Feb from Dec).

## 4. Regrouping Logic

The system reshapes the data into three "views" (schemas).

### 4.1. Time-Series-Wise (Sequence-Wise)
*   **Logic**: No regrouping. Iterates over the input list `[pred_0, pred_1, ...]`.
*   **Semantics**: "How well did *this specific model run* perform?"
*   **Metric Calculation**: Metrics are computed per DataFrame in the list.

### 4.2. Month-Wise
*   **Logic**: `pd.concat(predictions)`. Flattens all sequences into one massive DataFrame.
*   **Grouping**: `matched_pred.groupby(level=0)` (First level of MultiIndex, usually `month_id`).
*   **Semantics**: "Across all model runs (origins), how well did we predict for *January 2020*?"
    *   Includes predictions for Jan 2020 made 1 month ago, 2 months ago, etc.
*   **Metric Calculation**: Metrics are computed per unique month.

### 4.3. Step-Wise (Lead-Time-Wise)
*   **Logic**: `_split_dfs_by_step`.
*   **Mechanism**:
    *   Assumes all DataFrames in the list start at "Step 1".
    *   Slices the `i`-th month (unique time index) from every DataFrame.
    *   Combines them into a new DataFrame for "Step `i+1`".
*   **Implicit Contract**: The "Step" (Lead Time) is **strictly positional**.
    *   Row 0 of `pred[k]` is Step 1.
    *   Row 1 of `pred[k]` is Step 2.
    *   *Risk*: If a forecast is missing "Step 1" (e.g., starts at Step 2), this logic breaks or misaligns without warning.

## 5. Structural Requirements for `EvaluationFrame`

To replace pandas while preserving behavior, the `EvaluationFrame` must explicitly store the dimensions that are currently implicit.

### Required Explicit Columns (Identifiers)
1.  **`time_id`** (e.g., `month_id`): From Index Level 0.
2.  **`unit_id`** (e.g., `country_id`): From Index Level 1.
3.  **`origin_id`** (NEW): An ID distinguishing which "Forecast Sequence" a row belongs to.
    *   Needed to reconstruct "Time-Series-Wise" view.
    *   Replaces the `List[...]` structure.
4.  **`step_id`** (NEW): Integer lead time (1, 2, 3...).
    *   Needed to reconstruct "Step-Wise" view.
    *   Replaces the positional assumption in `_split_dfs_by_step`.

### Invariants to Enforce
*   `len(identifiers) == len(y_pred) == len(y_true)`
*   Alignment happens *once* during construction. `y_true` must be pre-duplicated to match the shape of `y_pred`.

## 6. The "Lists-in-Cells" Problem
*   Current State: `y_pred` is a Series where each cell is `List[float]` or `np.ndarray`.
*   Detection: `isinstance(x, list)` checks on every element.
*   Target State: `y_pred` is a dense `(N, S)` numpy array (S=samples). Point forecasts are `(N, 1)`.
