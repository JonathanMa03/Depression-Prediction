# Predicting Depression: A Comparative Analysis of Logistic Regression and Random Forest Models

## Overview

This project investigates the application of machine learning techniques for the early detection of depression. Using a modified and balanced version of the National Survey on Drug Use and Health (NSDUH), we compare two popular classification algorithms—**logistic regression** and **random forest**—to evaluate their predictive performance, interpretability, and applicability in clinical settings.

> **Goal:** Improve the accuracy and reliability of depression diagnosis through machine learning, with a focus on reducing false negatives and enhancing model interpretability.

---

## Key Features

- **Balanced Dataset**: Modified NSDUH data with equal representation of depressed and non-depressed individuals  
- **Feature Engineering**: One-hot encoding, chi-square tests, and threshold optimization  
- **Modeling**: 
  - Logistic Regression (high interpretability, high recall)
  - Random Forest (handles non-linear interactions, ensemble learning)
- **Evaluation Metrics**: Accuracy, Recall, AUC, ROC Curves, Geometric Mean Threshold Optimization  
- **Clinical Relevance**: Emphasis on minimizing false negatives to better identify individuals with depression

---

## Dataset

- Source: National Survey on Drug Use and Health (NSDUH)
- Features: Age, Gender, Race, Income, and various psychosocial indicators
- Target Variable: Presence or absence of a **major depressive episode**

---

## Results

| Model            | Accuracy | Recall  | AUC    | Optimal Threshold |
|------------------|----------|---------|--------|-------------------|
| Logistic Regression | 78%      | 88.96%  | 0.9012 | 0.29              |
| Random Forest       | 73.87%   | 88.28%  | 0.8882 | 0.07              |

- **Logistic regression** showed better balance between interpretability and recall, ideal for medical use-cases.
- **Random forest** demonstrated strong performance but slightly lower AUC and interpretability.

---

## Future Work

- Hyperparameter tuning (e.g., grid search for random forest)
- Feature reduction using PCA
- Integration of real-time data sources (e.g., smartphone sensors)
- Deployment as a decision-support tool for clinicians
