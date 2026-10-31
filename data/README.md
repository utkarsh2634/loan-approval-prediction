# Dataset Documentation: Loan Prediction Problem Dataset

## Overview
This repository uses the widely recognized **Loan Prediction Problem Dataset** from Kaggle, originally provided by Analytics Vidhya. The dataset captures historical retail banking loan application records and default outcomes used to model retail credit risk.

- **Primary file path:** `data/train.csv`
- **Fallback file path:** `data/synthetic_loan_data.csv` (used when `--use-synthetic` is specified)

---

## Instructions to Download from Kaggle

If you need to re-download the official dataset:
1. Visit the Kaggle dataset page: **[Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdeloitte/loan-prediction-problem-dataset)**.
2. Download the training dataset archive or the file named `train_u6lujuX_CVtDz.csv`.
3. Rename the downloaded file to `train.csv`.
4. Place `train.csv` into the `data/` directory of this repository:
   ```text
   loan-approval-prediction/
   └── data/
       ├── train.csv
       └── README.md
   ```
5. Run the training script:
   ```bash
   python train_model.py
   ```

---

## Dataset Schema

| Column Name | Description | Type / Values | Usage in Modeling |
| :--- | :--- | :--- | :--- |
| **Loan_ID** | Unique application identifier | String (e.g. `LP001002`) | **Dropped** (Identifier, no predictive power) |
| **Gender** | Applicant gender | `Male` / `Female` | **Dropped** (Excluded to prevent demographic bias) |
| **Married** | Applicant marital status | `Yes` / `No` | **Used** (Categorical) |
| **Dependents** | Number of dependents | `0`, `1`, `2`, `3+` (Cleaned to `3`) | **Used** (Categorical) |
| **Education** | Applicant education level | `Graduate` / `Not Graduate` | **Used** (Categorical) |
| **Self_Employed** | Employment status | `Yes` / `No` | **Used** (Categorical) |
| **ApplicantIncome** | Applicant monthly income | Numeric ($ / month) | **Used** (Numeric) |
| **CoapplicantIncome** | Co-applicant monthly income | Numeric ($ / month) | **Used** (Numeric) |
| **LoanAmount** | Loan amount requested | Numeric (Thousands of $) | **Used** (Numeric) |
| **Loan_Amount_Term** | Term of loan in months | Numeric (e.g. 120 to 480) | **Used** (Numeric) |
| **Credit_History** | Credit history meets guidelines | `1.0` (Good) / `0.0` (Poor) | **Used** (Categorical 0/1) |
| **Property_Area** | Location of collateral property | `Urban` / `Semiurban` / `Rural` | **Used** (Categorical) |
| **Loan_Status** | Target outcome variable | `Y` (Approved) / `N` (Rejected) | **Target** (`Y` &rarr; 1, `N` &rarr; 0) |

---

## Synthetic Data Generation

If the real dataset is unavailable or demonstration data is desired, run:
```bash
python train_model.py --use-synthetic
```
This automatically synthesizes `data/synthetic_loan_data.csv` (~614 rows) with realistic statistical distributions and flags all outputs as **"SYNTHETIC DATA — for demonstration only"**.
