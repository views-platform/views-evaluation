
<div style="width: 100%; max-width: 1500px; height: 400px; overflow: hidden; position: relative;">
  <img src="https://pbs.twimg.com/profile_banners/1237000633896652800/1717069203/1500x500" alt="VIEWS Twitter Header" style="position: absolute; top: -50px; width: 100%; height: auto;">
</div>

# Welcome to views-evaluation repository! 


The **views-evaluation** contains all the necesary components for evaluation of the VIEWS forecasts. This part of the VIEWS pipeline serves for storing and managing evaluation metrics for time series forecasting models.


## Table of Contents

<!-- toc -->

- [Overview](#overview)
    - [Evaluation Metrics](#1-evaluationmetrics)
    - [Metrics Manager](#metricsmanager)

<!-- tocstop -->





## Overview
### **EvaluationMetrics**
This is a data class for storing and managing evaluation metrics for time series forecasting models. It includes
* a set of metrics that account for the characteristics of conflict data, such as right-skewness and zero-inflation in the outcome variable. This decision was made at the Evaluation Metrics Workshop and more details can be found in the [Evaluation Metrics Workshop notes](https://www.notion.so/Notes-37de5410f8b547de8e03dddeb70193a6).
* function to generate dictionaries of EvaluationMetrics instances for three evaluation schemas: time-series wise, step-wise, and month-wise. 
* function to transform a structured dictionary of EvaluationMetrics instances into a DataFrame, where each row corresponds to a forecasting step and columns represent different metrics.

### **MetricsManager**
This is a class for calculating metrics on time series predictions. It includes:
* initialization by providing a list of metrics based on which models are evaluated. If some of the metrics are not in the list of metrics in data class EvaluationMetrics, there will be a warning and these metrics will be ignored.
* calculation of all the provided metrics.
* implementation of three evaluation schemas: time-series wise, step-wise, and month-wise. More details can be found in [schema.MD](https://github.com/prio-data/views_pipeline/blob/eval_docs/documentation/evaluation/schema.MD)

### 3. To do:
* Enable the evaluation of multiple targets (initialize MetricsManager by providing metrics and targets).
* Finish functions to calculate all the defined metrics (now only RMSLE, CRPS and AP are implemented).



## Funding and Partners 

<div style="width: 100%; max-width: 1500px; height: 400px; overflow: hidden; position: relative; margin-top: 50px;">
  <img src="image.png" alt="Funder logos" style="position: absolute; top: -50px; width: 100%; height: auto;">
</div>

