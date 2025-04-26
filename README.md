# 📦 Customer Churn Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Model-green.svg)](https://lightgbm.readthedocs.io/)
[![SMOTE](https://img.shields.io/badge/SMOTE-Oversampling-orange.svg)](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
[![Jupyter Notebook](https://img.shields.io/badge/Notebook-Jupyter-orange.svg)](https://jupyter.org/)
 
This project uses machine learning to predict customer churn for an e-commerce company based on customer demographics, purchase history, and browsing behavior.  
The goal is to proactively identify high-risk customers and improve retention strategies using predictive analytics.

## 🔍 Project Overview

- **Problem:**  
  Predict which customers are likely to churn within the next 30 days.

- **Dataset Features:**  
  - Demographics (Age, Gender, Location)
  - Purchase history (Frequency, Recency, Spend)
  - Browsing behavior (Pages visited, Time on site)
  - Customer feedback (Ratings, Reviews)

- **Target Variable:**  
  - Churn: 1 (Customer will churn) / 0 (Customer will not churn)

## 🛠️ Approach

- **Data Preprocessing**  
  - Confirmed **no missing values**.
  - Categorical encoding and feature scaling.
  
- **Handling Class Imbalance**  
  - Applied **SMOTE** (Synthetic Minority Oversampling Technique) to balance the churn/no-churn classes.

- **Models Trained:**  
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - XGBoost
  - LightGBM
  - CatBoost

## 📈 Model Performance (After SMOTE)

| Model                | Accuracy | Precision (Churn) | Recall (Churn) | ROC AUC |
|----------------------|----------|-------------------|----------------|---------|
| Logistic Regression  | 71%      | 37%               | 69%            | 0.70    |
| Decision Tree        | 68%      | 35%               | 76%            | 0.71    |
| Random Forest        | 84%      | 60%               | 60%            | 0.75    |
| KNN                  | 74%      | 41%               | 65%            | 0.71    |
| SVM                  | 79%      | 47%               | 73%            | 0.77    |
| XGBoost              | 85%      | 65%               | 55%            | 0.74    |
| LightGBM             | **86%**  | **68%**           | 56%            | **0.75** |
| CatBoost             | 86%      | 67%               | 55%            | 0.74    |

✅ **LightGBM achieved the best overall performance** after SMOTE balancing.

## 🔥 Feature Importance Insights

From LightGBM’s feature importance:
- **Top Predictors of Churn:**
  - Age
  - Tenure
  - Balance
  - Estimated Salary
  - Credit Score

Customers with higher age, lower tenure, and specific financial patterns are more likely to churn.

## 🚀 Next Steps

- Perform **hyperparameter tuning** to further boost precision and recall.
- Adjust the **classification threshold** to optimize business-specific trade-offs.
- Explore **ensemble techniques** like stacking models.
- Provide **actionable recommendations** for the marketing team to target high-risk customers.

## 📎 Files

- `customer_churn.ipynb` — Jupyter Notebook with all the code, modeling steps, visualizations, and final evaluation.

# 🔗 Useful Resources
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [SMOTE Documentation](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
