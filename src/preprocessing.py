"""
Data preprocessing, cleaning, synthetic data generation, and pipeline builder.
This module is shared across train_model.py, app.py, streamlit_app.py, and automated tests.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier

from src.config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_NUMERIC_FEATURES,
    ENGINEERED_NUMERIC_FEATURES,
    FEATURE_COLS,
    TARGET_COL,
    DROPPED_COLS,
    RANDOM_STATE,
    TRAIN_DATA_PATH,
    SYNTHETIC_DATA_PATH,
)

logger = logging.getLogger(__name__)


class LoanFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that calculates domain-specific underwriting ratios:
    1. TotalIncome: Combined household earnings = ApplicantIncome + CoapplicantIncome
    2. EMI: Monthly loan repayment estimate = (LoanAmount * 1000) / Loan_Amount_Term
    3. DebtToIncome: Monthly debt obligation burden = EMI / (TotalIncome + 1.0)
    4. LoanToIncome: Total loan burden relative to annual earnings = (LoanAmount * 1000) / (TotalIncome * 12.0 + 1.0)
    
    This ensures that changes in income, loan amount, and loan term directly alter
    predictive signals fed into the downstream estimators.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = X.copy()
        if not isinstance(X_df, pd.DataFrame):
            X_df = pd.DataFrame(X_df, columns=FEATURE_COLS if X.shape[1] == len(FEATURE_COLS) else None)

        # Ensure Gender exists for backward compatibility if omitted
        if "Gender" not in X_df.columns:
            X_df["Gender"] = "Male"

        app_inc = pd.to_numeric(X_df["ApplicantIncome"], errors="coerce").fillna(0.0)
        coapp_inc = pd.to_numeric(X_df["CoapplicantIncome"], errors="coerce").fillna(0.0)
        total_income = (app_inc + coapp_inc).clip(lower=0.0)

        # LoanAmount in thousands ($), convert to dollars: LoanAmount * 1000
        raw_amount = pd.to_numeric(X_df["LoanAmount"], errors="coerce").fillna(0.0)
        loan_amount_dollars = (raw_amount * 1000.0).clip(lower=0.0)

        # Loan term in months (defaults to 360 if missing or zero)
        term_months = pd.to_numeric(X_df["Loan_Amount_Term"], errors="coerce").replace(0, np.nan).fillna(360.0).clip(lower=1.0)

        # Monthly EMI estimate
        emi = loan_amount_dollars / term_months

        # Debt-to-Income: monthly payment relative to monthly household income
        debt_to_income = emi / (total_income + 1.0)

        # Loan-to-Income: total loan relative to annual household income
        loan_to_income = loan_amount_dollars / (total_income * 12.0 + 1.0)

        X_df["TotalIncome"] = total_income
        X_df["EMI"] = emi
        X_df["DebtToIncome"] = debt_to_income
        X_df["LoanToIncome"] = loan_to_income

        return X_df


def clean_dependents(val) -> str:
    """
    Normalizes the Dependents column:
    Converts '3+' to '3', handles numeric floats/ints, and preserves string categories.
    """
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    if val_str == "3+":
        return "3"
    if val_str.endswith(".0"):
        val_str = val_str[:-2]
    return val_str


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Cleans raw loan dataset:
    1. Records and removes duplicate rows.
    2. Drops non-predictive columns (Loan_ID) while strictly retaining all 11 features including Gender.
    3. Cleans Dependents ('3+' -> '3').
    4. Formats Credit_History as categorical string ('0' / '1' / NaN).
    5. Formats Gender ('Male' / 'Female' / NaN).
    6. Ensures numeric columns have valid float representations.

    Returns:
        Tuple of (cleaned_df, duplicate_count)
    """
    df = df.copy()
    initial_count = len(df)

    # Remove duplicates
    df = df.drop_duplicates()
    duplicates_removed = initial_count - len(df)
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate rows.")

    # Drop non-predictive identifiers (Loan_ID only)
    cols_to_drop = [c for c in DROPPED_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info(f"Dropped excluded columns: {cols_to_drop}")

    # Clean Dependents
    if "Dependents" in df.columns:
        df["Dependents"] = df["Dependents"].apply(clean_dependents)

    # Standardize Credit_History as string category ('0' or '1' or np.nan)
    if "Credit_History" in df.columns:
        df["Credit_History"] = df["Credit_History"].apply(
            lambda v: np.nan if pd.isna(v) else str(int(float(v)))
        )

    # Standardize Gender ('Male', 'Female')
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].apply(
            lambda v: np.nan if pd.isna(v) else str(v).strip().capitalize()
        )

    # Ensure numeric columns are properly typed
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, duplicates_removed


def prepare_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Extracts strictly the 11 required features and maps the binary target.
    Target mapping: 'Y' -> 1, 'N' -> 0.
    Asserts that the target is strictly not inside the feature set X.
    """
    # Verify required feature columns exist
    missing_feats = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_feats:
        raise ValueError(f"Input data is missing required feature columns: {missing_feats}")

    X = df[FEATURE_COLS].copy()

    # Safety assertion: Target must never leak into feature matrix X
    assert TARGET_COL not in X.columns, f"Target '{TARGET_COL}' must not be present in features X!"

    y = None
    if TARGET_COL in df.columns:
        y_raw = df[TARGET_COL].dropna()
        if set(y_raw.unique()).issubset({"Y", "N", 1, 0, "1", "0"}):
            y = df[TARGET_COL].map({"Y": 1, "N": 0, 1: 1, 0: 0, "1": 1, "0": 0})
        else:
            y = df[TARGET_COL].astype(int)

    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs an end-to-end ColumnTransformer:
    - Numeric features (both raw 4 + engineered 4 = 8):
      Median Imputation + RobustScaler (handles heavy skew and outliers)
    - Categorical features (7 categories):
      Most Frequent Imputation + OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, ALL_NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def build_pipeline(classifier=None) -> Pipeline:
    """
    Combines the LoanFeatureEngineer domain transformer, ColumnTransformer preprocessor,
    and a classifier into an end-to-end scikit-learn Pipeline.
    Default classifier is a balanced Random Forest.
    """
    if classifier is None:
        classifier = RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
        )

    preprocessor = build_preprocessor()

    pipeline = Pipeline(
        steps=[
            ("engineer", LoanFeatureEngineer()),
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return pipeline


def generate_synthetic_data(
    output_path: Path = SYNTHETIC_DATA_PATH,
    n_samples: int = 614,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Generates a realistic synthetic loan dataset (~614 rows matching Kaggle's shape)
    with realistic distributions, realistic missing values (~5%), and loan approval
    realistically correlated with credit history, total household income, loan amount, and property area.
    """
    np.random.seed(random_state)

    loan_ids = [f"LP00{1000 + i}" for i in range(n_samples)]
    genders = np.random.choice(["Male", "Female"], size=n_samples, p=[0.81, 0.19])
    married = np.random.choice(["Yes", "No"], size=n_samples, p=[0.65, 0.35])
    dependents = np.random.choice(["0", "1", "2", "3+"], size=n_samples, p=[0.57, 0.17, 0.17, 0.09])
    education = np.random.choice(["Graduate", "Not Graduate"], size=n_samples, p=[0.78, 0.22])
    self_employed = np.random.choice(["No", "Yes"], size=n_samples, p=[0.86, 0.14])

    # Incomes (log-normal distributions)
    applicant_income = np.random.lognormal(mean=8.4, sigma=0.6, size=n_samples).astype(int)
    coapplicant_income = np.where(
        np.random.rand(n_samples) > 0.45,
        np.random.lognormal(mean=7.5, sigma=0.8, size=n_samples).astype(int),
        0,
    )

    # Loan amounts in thousands (median around 130)
    loan_amount = np.clip(
        np.random.lognormal(mean=4.8, sigma=0.5, size=n_samples), 10, 700
    ).round().astype(float)

    # Terms (predominantly 360 months)
    terms = np.random.choice([120, 180, 240, 300, 360, 480], size=n_samples, p=[0.01, 0.07, 0.01, 0.02, 0.85, 0.04])

    # Credit History (predominantly 1)
    credit_history = np.random.choice([1.0, 0.0], size=n_samples, p=[0.84, 0.16])

    # Property area
    property_area = np.random.choice(["Semiurban", "Urban", "Rural"], size=n_samples, p=[0.38, 0.33, 0.29])

    # Realistic approval calculation
    total_income = applicant_income + coapplicant_income
    income_loan_ratio = total_income / (loan_amount * 1000.0 + 1.0)

    logits = (
        (credit_history * 3.0 - 1.5)
        + (income_loan_ratio * 35.0 - 1.0)
        + (education == "Graduate") * 0.3
        + (property_area == "Semiurban") * 0.4
        - (property_area == "Rural") * 0.2
        + np.random.normal(0, 0.7, size=n_samples)
    )
    probs = 1 / (1 + np.exp(-logits))
    loan_status = np.where(probs >= 0.5, "Y", "N")

    df = pd.DataFrame(
        {
            "Loan_ID": loan_ids,
            "Gender": genders,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": terms,
            "Credit_History": credit_history,
            "Property_Area": property_area,
            "Loan_Status": loan_status,
        }
    )

    # Inject realistic missing values
    for col, rate in [
        ("Gender", 0.02),
        ("Married", 0.01),
        ("Dependents", 0.025),
        ("Self_Employed", 0.05),
        ("LoanAmount", 0.035),
        ("Loan_Amount_Term", 0.02),
        ("Credit_History", 0.08),
    ]:
        mask = np.random.rand(n_samples) < rate
        df.loc[mask, col] = np.nan

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Generated synthetic dataset with {len(df)} rows at {output_path}")
    return df


def load_dataset(
    data_path: Path = TRAIN_DATA_PATH,
    use_synthetic: bool = False,
) -> Tuple[pd.DataFrame, bool]:
    """
    Loads dataset:
    - If use_synthetic=True, uses/generates synthetic dataset.
    - Otherwise loads data_path if it exists.
    - If data_path is missing, falls back to synthetic dataset.

    Returns:
        Tuple of (df, is_synthetic_flag)
    """
    if use_synthetic:
        logger.info("Using synthetic dataset as explicitly requested.")
        if not SYNTHETIC_DATA_PATH.exists():
            df = generate_synthetic_data(SYNTHETIC_DATA_PATH)
        else:
            df = pd.read_csv(SYNTHETIC_DATA_PATH)
        return df, True

    if data_path.exists():
        logger.info(f"Loading real dataset from {data_path}")
        df = pd.read_csv(data_path)
        return df, False

    logger.warning(
        f"Real dataset not found at {data_path}. Falling back to synthetic demonstration data."
    )
    if not SYNTHETIC_DATA_PATH.exists():
        df = generate_synthetic_data(SYNTHETIC_DATA_PATH)
    else:
        df = pd.read_csv(SYNTHETIC_DATA_PATH)
    return df, True
