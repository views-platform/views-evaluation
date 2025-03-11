![GitHub License](https://img.shields.io/github/license/views-platform/views-evaluation)
![GitHub branch check runs](https://img.shields.io/github/check-runs/views-platform/views-evaluation/main)
![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/views-platform/views-evaluation)
![GitHub Release](https://img.shields.io/github/v/release/views-platform/views-evaluation)

<div style="width: 100%; max-width: 1500px; height: 400px; overflow: hidden; position: relative;">
  <img src="https://github.com/user-attachments/assets/1ec9e217-508d-4b10-a41a-08dface269c7" alt="VIEWS Twitter Header" style="position: absolute; top: -50px; width: 100%; height: auto;">
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
3. **Metrics Computation (via EvaluationManager)** →  
4. **Final Performance Reports**  

### **Integration with Other Repositories:**  
- **[views-pipeline-core](https://github.com/views-platform/views-pipeline-core):** Supplies preprocessed data for evaluation.  
- **[views-models](https://github.com/views-platform/views-models):** Provides trained models to be assessed.  
- **[views-stepshifter](https://github.com/views-platform/views-stepshifter):** Evaluates **time-shifted forecasting models**.  
- **[views-hydranet](https://github.com/views-platform/views-hydranet):** Supports **spatiotemporal deep learning model evaluations**.  

---

## ✨ **Features**  
* **Comprehensive Evaluation Framework**: The `EvaluationManager` class provides structured methods to evaluate time series predictions based on **point** and **uncertainty** metrics.
* **Multiple Evaluation Schemas**:
  * **Step-wise evaluation**: groups and evaluates predictions by the respective steps from all models.
  * **Time-series-wise evaluation**: evaluates predictions for each time-series.
  * **Month-wise evaluation**: groups and evaluates predictions at a monthly level.
* **Support for Mulyiple Metrics**
  * **Point Evaluation Metrics**: RMSLE, CRPS, Average Precision (Brier Score, Jeffreys Divergence, Pearson Correlation, Sinkhorn/Earth-mover Distance & pEMDiv and Variogram to be added).
  * **Uncertainty Evaluation Metrics**: CRPS (and more to be added in the future).
* **Data Integrity Checks**: Ensures that input DataFrames conform to expected structures before evaluation based on point and uncertainty evaluation.
* **Automatic Index Matching**: Aligns actual and predicted values based on MultiIndex structures.
* **Planned Enhancements**: 
  * **Expanding metric calculations** beyond RMSLE, CRPS, and AP.  
  * **New visualization tools** for better interpretability of evaluation reports.  

---

## ⚙️ **Installation**  

### **Prerequisites**  
- Python **>= 3.11**  

### **From PyPI**
```
pip install views_evaluation
```

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

