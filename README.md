# 🧠 JPMorgan Chase – ML Take-Home Project  
**Author:** Dhruvil Gorasiya  
**Objective:** Income Classification & Population Segmentation using Weighted Census Data  

---

## 📌 1. Project Overview
This project was developed as part of the **JPMorgan Chase Data Science Take-Home Challenge**.  
The task involves:

1. **Income Classification:**  
   Build a model that predicts whether a person earns more or less than $50,000 per year based on 40 demographic and employment variables.  

2. **Population Segmentation:**  
   Develop a segmentation model to cluster individuals into distinct, actionable groups for marketing purposes.

The dataset represents **weighted census data** from the **U.S. Census Bureau (1994–1995)**, with each record including a weight that reflects its population representation.

---

## 🧩 2. Repository Structure

```
project_root/
│
├── data/
│   ├── census-bureau.data              # Raw data
│   ├── census-bureau.columns           # Column headers
│
├── src/
│   ├── data_processor.py               # Data loading, cleaning, encoding
│   ├── classification.py               # Model training, calibration, evaluation
│   ├── segmentation.py                 # Weighted K-Means segmentation & profiling
│   ├── metrics.py                      # Evaluation metrics (ROC-AUC, PR-AUC, F1, etc.)
│   ├── scoring.py                      # Scoring new data using saved model
│   └── config.py                       # Configuration constants (random seed, thresholds)
│
├── notebooks/
│   ├── census_eda.ipynb                # EDA and feature insights
│   ├── shap_analysis.ipynb             # Explainability (SHAP feature importance)
│   ├── demo.ipynb                      # End-to-end demo and visualization
│
├── outputs/
│   ├── model_performance.json          # Final classification metrics
│   ├── segmentation_results.json       # Cluster summary stats
│   ├── cluster_profiles.csv            # Cluster-level attribute breakdown
│   ├── cluster_lift_table.csv          # Cluster lift analysis
│   └── final_model.pkl                 # Trained calibrated XGBoost model
│
├── README.md                           # (This file)
└── report.pdf                          # 9-page final project report
```

---

## ⚙️ 3. Environment Setup

### Step 1: Create Environment
```bash
conda create -n jpmc_project python=3.11
conda activate jpmc_project
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```


## 🚀 4. How to Run the Project

### A. Income Classification
```bash
python -m src.classification
```

This will:
- Load the census data
- Preprocess and encode features
- Train and calibrate multiple models (Logistic Regression, Random Forest, XGBoost)
- Evaluate using weighted ROC-AUC, PR-AUC, F1 metrics
- Save the best calibrated model and performance summary to `/outputs/`

### B. Segmentation Analysis
```bash
python -m src.segmentation
```

This:
- Uses the encoded feature space to perform **weighted K-Means clustering**
- Selects the optimal number of clusters (default: 8)
- Generates cluster profiles and lift analysis
- Saves results to `/outputs/cluster_profiles.csv` and `/outputs/cluster_lift_table.csv`

### C. Scoring New Data
To score unseen data using the trained model:
```python
from src.scoring import score_and_decide
import pandas as pd

df_new = pd.read_csv("new_data.csv")
preds = score_and_decide(df_new, "outputs/final_model.pkl")
print(preds.head())
```

### **D. Run Complete Pipeline via `main.py`**

You can also execute the **entire workflow — including both income classification and segmentation —** directly from a single Python script.

#### **To run the full pipeline in one command:**
```bash
python main.py
```

---


## 📊 5. Model Summary

| Metric | Value |
|--------:|------:|
| ROC-AUC | **0.9451** |
| PR-AUC  | **0.6200** |
| Best Threshold | **0.19** |
| Precision | 0.590 |
| Recall | 0.601 |
| F1 Score | 0.595 |
| Optimal Clusters | **8** |
| Overall Positive Rate | 6.39% |

---

## 💡 6. End-to-End Pipeline (Demo Notebook)
-The demo.ipynb notebook provides a complete end-to-end execution pipeline that reproduces every stage of the project in a single run.
-It integrates all major components — from data preprocessing to final segmentation and explainability — in one streamlined workflow.
```bash
jupyter notebook notebooks/demo.ipynb
```
---

## 💡 7. Insights
- **Education level, hours worked per week, occupation type, and marital status** were among the strongest predictors of high income.  
- **Cluster segmentation** revealed eight distinct socioeconomic groups with varying purchasing potential.  
- **Cluster lift analysis** helps prioritize marketing efforts toward high-income or high-spend segments.

---

## 🧾 8. References
- U.S. Census Bureau: Current Population Surveys (1994–1995)  
- scikit-learn Documentation (v1.5)  
- XGBoost Documentation  
- SHAP Explainability Library  
