# 🏦 Loan Eligibility Prediction

This project builds machine learning models to predict whether a loan applicant is eligible for approval based on financial and demographic attributes. The dataset was cleaned, encoded, and used to train multiple classification algorithms, with a focus on comparing performance with and without feature scaling and hyperparameter tuning.

---

## 📊 Dataset

The dataset contains the following attributes:

- Gender, Marital Status, Dependents  
- Education, Self-Employment  
- Applicant Income, Coapplicant Income  
- Loan Amount, Loan Term  
- Credit History, Property Area  
- **Target**: `Loan_Status` (1 = Approved, 0 = Not Approved)

---

## ⚙️ Process Overview

### 1. **Data Cleaning**
- Filled missing values using median/mode
- Converted '3+' in `Dependents` to integer
- Encoded categorical features using `LabelEncoder`

### 2. **Feature Engineering**
- Transformed categorical variables to numerical form
- Encoded `Loan_Status` to binary target (0/1)

### 3. **Modeling**
Applied and compared:
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Multi-layer Perceptron (ANN / MLPClassifier)  

Tested each model under:
- No scaling  
- With feature scaling (StandardScaler)  
- With hyperparameter tuning via `GridSearchCV`

### 4. **Evaluation**
- Metrics: Accuracy, Precision, Recall, F1-score
- Used train/validation/test split
- Confusion matrix plotted for model performance visualization

---

## ✅ Results

### Model Accuracy Comparison (Validation / Test)

| Model Variant | Validation / Test Accuracy |
|-----------------------|----------------------------|
| Logistic (Raw) | 0.7805 / 0.8495 |
| Logistic (Scaled) | 0.7805 / 0.8495 |
| Logistic (Grid) | 0.7805 / 0.8495 |
| KNN (Raw) | 0.6344 / 0.6667 |
| KNN (Scaled) | 0.7317 / 0.8280 |
| KNN (Grid) | 0.7506 / 0.8387 |
| ANN (Raw) | 0.6585 / 0.6882 |
| ANN (Scaled) | 0.6992 / 0.7142 |
| ANN (Grid) | 0.7560 / 0.7956 |

---

## 🧠 Summary

- **Feature scaling significantly improved KNN and ANN** performance, while it had **no effect on Logistic Regression**.
- **Logistic Regression** was the most stable and accurate model across all conditions, consistently achieving **84.95% accuracy** on the test set.
- **GridSearchCV** improved both **KNN** and **ANN**, but had **no measurable effect on Logistic Regression**, confirming its robustness to hyperparameter tuning.
- **Best overall model**:  
  ➤ **Logistic Regression with or without GridSearch**, with **test accuracy of 84.95%** and validation accuracy of **78.05%**.

---

## 📦 Requirements

- `pandas`  
- `scikit-learn`  
- `matplotlib`  
- `seaborn`
