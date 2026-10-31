"""
Configuration file for the Loan Approval Prediction project.
Defines project directory paths, feature groupings, model hyperparameters,
and runtime constants.
"""

from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
STATIC_DIR = PROJECT_ROOT / "static"
TEMPLATES_DIR = PROJECT_ROOT / "templates"

# File paths
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
SYNTHETIC_DATA_PATH = DATA_DIR / "synthetic_loan_data.csv"
MODEL_FILE_PATH = MODELS_DIR / "loan_model.joblib"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"
REPORT_PATH = OUTPUTS_DIR / "classification_report.txt"

# Feature definitions
# Exactly 11 input features are used across training and inference:
NUMERIC_FEATURES = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
]

CATEGORICAL_FEATURES = [
    "Gender",          # Demographic feature (Male / Female)
    "Married",         # Marital status (Yes / No)
    "Dependents",      # Dependents count ('0', '1', '2', '3')
    "Education",       # Education level (Graduate / Not Graduate)
    "Self_Employed",   # Employment status (Yes / No)
    "Credit_History",  # Treated as categorical ('0' / '1')
    "Property_Area",   # Urban / Semiurban / Rural
]

FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Engineered financial ratio features generated inside the Pipeline
ENGINEERED_NUMERIC_FEATURES = [
    "TotalIncome",      # ApplicantIncome + CoapplicantIncome
    "EMI",              # Estimated monthly payment: (LoanAmount * 1000) / Loan_Amount_Term
    "DebtToIncome",     # EMI / (TotalIncome + 1)
    "LoanToIncome",     # (LoanAmount * 1000) / (TotalIncome * 12 + 1)
]

ALL_NUMERIC_FEATURES = NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES

# Target column
TARGET_COL = "Loan_Status"

# Excluded columns from raw dataset (only Loan_ID is dropped; Gender is strictly retained!)
DROPPED_COLS = ["Loan_ID"]

# Modeling constants
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Hyperparameter grids for model tuning
PARAM_GRID_DT = {
    "classifier__max_depth": [3, 4, 5, 6, 7],
    "classifier__min_samples_leaf": [2, 5, 10, 15],
    "classifier__criterion": ["gini", "entropy"],
}

PARAM_GRID_RF = {
    "classifier__n_estimators": [100, 150, 200],
    "classifier__max_depth": [4, 5, 6, 8, None],
    "classifier__min_samples_leaf": [2, 4, 6],
    "classifier__max_features": ["sqrt", "log2"],
}

PARAM_GRID_LR = {
    "classifier__C": [0.01, 0.1, 1.0, 10.0],
    "classifier__solver": ["lbfgs", "liblinear"],
}

# Educational disclaimer text displayed across app and reports
DISCLAIMER_TEXT = (
    "This is an educational prediction, not an actual bank decision or a "
    "guarantee of loan approval."
)

# Loan Term constraints and extrapolation thresholds (in months)
LOAN_TERM_MIN_MONTHS = 1
LOAN_TERM_MAX_MONTHS = 480
LOAN_TERM_SHORT_THRESHOLD_MONTHS = 12

# Non-blocking warning messages for rare loan term boundaries
SHORT_TERM_WARNING_TEXT = (
    "Short loan terms are rare in the training data, so this prediction is less reliable."
)
LONG_TERM_WARNING_TEXT = (
    "Loan terms above 480 months (40 years) are rare or unobserved in the training data, "
    "so this prediction is less reliable."
)
