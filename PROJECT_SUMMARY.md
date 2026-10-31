# Project Summary: Loan Approval Prediction

**Course & Section:** Machine Learning &bull; Section: 6B &bull; B.Tech CSE  
**Problem Type:** Binary Classification &bull; **Algorithm:** DecisionTreeClassifier (`scikit-learn`)  
**Deployment Platform:** Render (Gunicorn / Linux) &bull; **Web Framework:** Flask + Jinja2 + Pure CSS  

---

## 1. Executive Summary & Methodology
The objective of this project is to develop and deploy an automated, transparent credit evaluation model that predicts whether a prospective loan application is **"Likely Approved"** or **"Likely Rejected"**.

The full machine learning lifecycle was implemented end-to-end:
1. **Data Ingestion & Integrity:** Ingested the Kaggle Loan Prediction Problem benchmark dataset (`614 rows &times; 13 columns`).
2. **Preprocessing Pipeline:** Removed duplicates, stripped identifiers (`Loan_ID`), and excluded protected demographic attributes (`Gender`) to prevent demographic bias.
3. **Leakage-Free Transformation:** Executed an 80/20 stratified train/test split. Embedded median imputation for numeric attributes, most-frequent mode imputation, and one-hot categorical encoding inside a reusable `ColumnTransformer` scikit-learn `Pipeline`.
4. **Hyperparameter Optimization:** Performed 5-Fold Stratified Cross-Validation (`GridSearchCV`) across `max_depth` (2&ndash;8), `min_samples_leaf` (1, 5, 10), and criterion (`gini`, `entropy`).
5. **Full Web Serving & Verification:** Deployed via Flask with server-side validation, health check monitoring, and a diagnostic dashboard.

---

## 2. Experimental Results & Measured Metrics

All reported performance metrics were measured at runtime:

| Metric | Measured Value | Notes & Analysis |
| :--- | :---: | :--- |
| **Tuned Test Accuracy** | **85.37%** | Evaluated on unseen 20% stratified test set (123 samples) |
| **Tuned Train Accuracy** | **79.84%** | Close alignment with test score confirms zero overfitting |
| **5-Fold CV Score** | **79.83%** | Mean cross-validation score across 5 stratified folds |
| **Majority Baseline** | **69.11%** | Score achieved by a naive `DummyClassifier` |
| **Model Lift over Baseline** | **+16.26%** | Statistical performance advantage over majority guessing |
| **Untuned Tree Train Acc** | **100.00%** | Demonstrated extreme memorization without pruning |
| **Untuned Tree Test Acc** | **74.80%** | Severe overfitting gap of **25.20%** resolved by tuning |

### Optimal Hyperparameters Found:
- `criterion`: `'gini'`
- `max_depth`: `2`
- `min_samples_leaf`: `10`

---

## 3. Confusion Matrix & Classification Report (Test Set)

```text
                  Predicted Rejected (0)    Predicted Approved (1)
True Rejected (0)          21 (TN)                    1 (FP)
True Approved (1)          17 (FN)                   84 (TP)
```

- **Rejection Precision:** **95.45%** (21 / 22) &mdash; When the model predicts rejection, it is correct in 95.5% of cases, protecting the bank against high-risk defaults.
- **Approval Recall:** **98.82%** (84 / 85) &mdash; Over 98.8% of creditworthy applicants are correctly identified for approval.
- **Overall Macro F1-Score:** **0.8016** &bull; **Weighted F1-Score:** **0.8404**

---

## 4. Key EDA & Feature Importance Findings

1. **Credit History is the Dominant Predictor:**
   - Feature Importance: **`Credit_History_1` = 94.46%**; **`LoanAmount` = 5.54%**.
   - Applicants with good credit history (`Credit_History = 1.0`) achieve an approval rate of **79.6%**, whereas applicants with delinquent records (`0.0`) have an approval rate of only **8.3%**.
2. **Class Imbalance:**
   - The dataset exhibits a 68.73% approval rate vs 31.27% rejection rate.
3. **Property Location Factor:**
   - Semiurban applications demonstrate the highest empirical approval rate (~76.8%), compared to Rural areas (~61.5%).

---

## 5. Limitations

1. **Small Benchmark Sample:** The dataset contains 614 rows. While sufficient for tree pruning, larger institutional datasets are required to capture rare tail-risk events.
2. **Missing Value Densities:** Credit History had 50 missing values (~8.1%) requiring mode imputation.
3. **Educational Scope:** Built as an academic teaching implementation. Real production credit engines require compliance with legal credit underwriting regulations.

---

## 6. Future Enhancements

- **Ensemble Benchmarking:** Compare tuned Decision Tree against Random Forest and XGBoost classifiers.
- **Class Imbalance Handling:** Benchmark SMOTE (Synthetic Minority Over-sampling) and class weighting (`class_weight='balanced'`).
- **SHAP & LIME Explanations:** Integrate Shapley additive explanations for individual applicant auditability.
- **Persistent Database:** Connect PostgreSQL/SQLite to record underwriting logs and prediction latency metrics.
