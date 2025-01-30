![GitHub License](https://img.shields.io/github/license/views-platform/views-evaluation)
![GitHub branch check runs](https://img.shields.io/github/check-runs/views-platform/views-evaluation/main)
![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/views-platform/views-evaluation)
![GitHub Release](https://img.shields.io/github/v/release/views-platform/views-evaluation)

<div style="width: 100%; max-width: 1500px; height: 400px; overflow: hidden; position: relative;">
  <img src="https://pbs.twimg.com/profile_banners/1237000633896652800/1717069203/1500x500" alt="VIEWS Twitter Header" style="position: absolute; top: -50px; width: 100%; height: auto;">
</div>

# **VIEWS Evaluation** 📊  

> **Part of the [VIEWS Platform](https://github.com/views-platform) ecosystem for large-scale conflict forecasting.**  

## 📚 **Table of Contents**  

1. [Overview](#overview)  
2. [Role in the VIEWS Pipeline](#role-in-the-views-pipeline)  
3. [Features](#features)  
4. [Installation](#installation)  
5. [Architecture](#architecture)  
6. [Project Structure](#project-structure)  
7. [Contributing](#contributing)  
8. [License](#license)  
9. [Acknowledgements](#acknowledgements)  

---

## 🧠 **Overview**  

The **VIEWS Evaluation** repository provides a standardized framework for **assessing time-series forecasting models** used in the **VIEWS conflict prediction pipeline**. It ensures consistent, robust, and interpretable evaluations through **metrics tailored to conflict-related data**, which often exhibit **right-skewness and zero-inflation**.  

---

## 🌍 **Role in the VIEWS Pipeline**  

VIEWS Evaluation ensures **forecasting accuracy and model robustness** as the **official evaluation component** of the VIEWS ecosystem.  

### **Pipeline Integration:**  
1. **Model Predictions** →  
2. **Evaluation Metrics Processing** →  
3. **Metrics Computation (via MetricsManager)** →  
4. **Final Performance Reports**  

### **Integration with Other Repositories:**  
- **[views-pipeline-core](https://github.com/views-platform/views-pipeline-core):** Supplies preprocessed data for evaluation.  
- **[views-models](https://github.com/views-platform/views-models):** Provides trained models to be assessed.  
- **[views-stepshifter](https://github.com/views-platform/views-stepshifter):** Evaluates **time-shifted forecasting models**.  
- **[views-hydranet](https://github.com/views-platform/views-hydranet):** Supports **spatiotemporal deep learning model evaluations**.  

---

## ✨ **Features**  

### **1. EvaluationMetrics**  
A **data class** for managing and storing evaluation metrics for time-series forecasting models.  

🔹 **Key Capabilities:**  
- **Handles conflict-specific data distributions**, including **skewness and zero-inflation**.  
- **Three evaluation schemas**:  
  1. **Time-series-wise**: Evaluates long-term forecasting behavior.  
  2. **Step-wise**: Assesses performance at each forecasting step.  
  3. **Month-wise**: Measures forecast accuracy on a rolling monthly basis.  
- **Transforms evaluation metrics into structured DataFrames** for analysis.  

📖 More details in the **[Evaluation Metrics Workshop Notes](https://www.notion.so/Notes-37de5410f8b547de8e03dddeb70193a6)**.  

---

### **2. MetricsManager**  
A **centralized evaluation engine** for computing metrics on time-series forecasts.  

🔹 **Key Capabilities:**  
- **Customizable metric lists** allow for flexible evaluation.  
- **Ensures metric consistency** by warning about unrecognized metrics.  
- **Implements all three evaluation schemas** (time-series, step-wise, month-wise).  
- **Batch processing** for multiple models and forecasting targets.  

📖 More details in **[schema.MD](https://github.com/prio-data/views_pipeline/blob/eval_docs/documentation/evaluation/schema.MD)**.  

---

### **3. Roadmap & Upcoming Features** 🚧  
✅ **Planned Enhancements:**  
- **Multi-target evaluation** (e.g., assessing multiple dependent variables simultaneously).  
- **Expanding metric calculations** beyond RMSLE, CRPS, and AP.  
- **New visualization tools** for better interpretability of evaluation reports.  

---

## ⚙️ **Installation**  

### **Prerequisites**  
- Python **>= 3.11**  

---

## 🏗 **Architecture**  

### **1. Evaluation Metrics Framework**  
- **Handles forecasting evaluation** across **multiple models, levels of analysis, and forecasting windows**.  
- Converts model outputs into **standardized evaluation reports**.  

### **2. Metrics Computation Pipeline**  
1. **Input**: Predictions from models in standardized DataFrames.  
2. **Processing**: Calculation of relevant evaluation metrics.  
3. **Output**: Performance scores for comparison across models.  

### **3. Error Handling & Standardization**  
- **Ensures conformity to VIEWS evaluation standards**.  
- **Warns about unrecognized or incorrectly formatted metrics**.  

---

## 🗂 **Project Structure**  

```plaintext
views-evaluation/
├── README.md                   # Documentation
├── .github/workflows/           # CI/CD pipelines
├── tests/                       # Unit tests
├── views_evaluation/            # Main source code
│   ├── evaluation/
│   │   ├── metrics.py
│   ├── __init__.py              # Package initialization
├── .gitignore                   # Git ignore rules
├── pyproject.toml               # Poetry project file
├── poetry.lock                  # Dependency lock file
```

---

## 🤝 **Contributing**  

We welcome contributions! Please follow the **[VIEWS Contribution Guidelines](https://github.com/views-platform/docs)**.  

---

## 📜 **License**  

This project is licensed under the [LICENSE](/LICENSE) file. 

---

## 💬 **Acknowledgements**  

<p align="center">
  <img src="https://raw.githubusercontent.com/views-platform/docs/main/images/views_funders.png" alt="Views Funders" width="80%">
</p>

Special thanks to the **VIEWS MD&D Team** for their collaboration and support.  

