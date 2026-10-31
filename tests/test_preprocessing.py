"""
Automated unit tests for data cleaning and preprocessing pipeline.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.config import FEATURE_COLS, TARGET_COL
from src.preprocessing import (
    build_pipeline,
    clean_dataset,
    clean_dependents,
    prepare_features_and_target,
)


@pytest.fixture
def sample_raw_dataframe():
    """Provides a synthetic in-memory fixture DataFrame for testing."""
    return pd.DataFrame(
        {
            "Loan_ID": ["LP001", "LP002", "LP002", "LP003", "LP004"],
            "Gender": ["Male", "Female", "Female", "Male", np.nan],
            "Married": ["Yes", "No", "No", "Yes", "Yes"],
            "Dependents": ["0", "1", "1", "3+", np.nan],
            "Education": ["Graduate", "Not Graduate", "Not Graduate", "Graduate", "Graduate"],
            "Self_Employed": ["No", "Yes", "Yes", "No", np.nan],
            "ApplicantIncome": [5000, 3000, 3000, 7500, np.nan],
            "CoapplicantIncome": [1500, 0, 0, 2200, 500],
            "LoanAmount": [120, 100, 100, 250, np.nan],
            "Loan_Amount_Term": [360, 360, 360, 180, 360],
            "Credit_History": [1.0, 0.0, 0.0, 1.0, np.nan],
            "Property_Area": ["Urban", "Rural", "Rural", "Semiurban", "Urban"],
            "Loan_Status": ["Y", "N", "N", "Y", "Y"],
        }
    )


def test_clean_dependents():
    """Verifies that '3+' is normalized to '3' and non-string inputs are handled."""
    assert clean_dependents("3+") == "3"
    assert clean_dependents("0") == "0"
    assert clean_dependents("1") == "1"
    assert clean_dependents("2") == "2"
    assert clean_dependents(3) == "3"
    assert clean_dependents(1.0) == "1"
    assert pd.isna(clean_dependents(np.nan))


def test_clean_dataset_removes_duplicates_and_drops_excluded(sample_raw_dataframe):
    """Verifies duplicate removal and stripping of Loan_ID and Gender."""
    cleaned_df, dupes = clean_dataset(sample_raw_dataframe)
    # The fixture has 1 duplicate row (LP002)
    assert dupes == 1
    assert len(cleaned_df) == 4
    # Ensure Loan_ID is removed while Gender is retained as an active feature
    assert "Loan_ID" not in cleaned_df.columns
    assert "Gender" in cleaned_df.columns
    # Ensure Dependents 3+ was cleaned
    assert "3+" not in cleaned_df["Dependents"].values


def test_prepare_features_and_target(sample_raw_dataframe):
    """Ensures target column is strictly isolated from feature matrix X."""
    cleaned_df, _ = clean_dataset(sample_raw_dataframe)
    X, y = prepare_features_and_target(cleaned_df)

    assert TARGET_COL not in X.columns
    assert list(X.columns) == FEATURE_COLS
    assert len(X) == len(cleaned_df)
    assert len(y) == len(cleaned_df)
    # Verify binary encoding: Y -> 1, N -> 0
    assert set(y.unique()).issubset({0, 1})


def test_pipeline_handles_nans_and_unseen_categories(sample_raw_dataframe):
    """Verifies that the ColumnTransformer pipeline handles NaNs and new categories safely."""
    cleaned_df, _ = clean_dataset(sample_raw_dataframe)
    X, y = prepare_features_and_target(cleaned_df)

    pipeline = build_pipeline()
    pipeline.fit(X, y)

    # Test with unseen categories and NaNs in both numeric & categorical columns
    test_sample = pd.DataFrame(
        [
            {
                "ApplicantIncome": np.nan,
                "CoapplicantIncome": 0,
                "LoanAmount": np.nan,
                "Loan_Amount_Term": np.nan,
                "Gender": "UnknownGender",
                "Credit_History": "UnseenCreditCode",  # unseen category
                "Self_Employed": "UnknownType",       # unseen category
                "Married": "Divorced",                # unseen category
                "Education": "Doctorate",             # unseen category
                "Dependents": "5",                    # unseen category
                "Property_Area": "Suburban",          # unseen category
            }
        ]
    )

    preds = pipeline.predict(test_sample)
    probs = pipeline.predict_proba(test_sample)

    assert len(preds) == 1
    assert preds[0] in (0, 1)
    assert probs.shape == (1, 2)
    assert np.isclose(probs.sum(), 1.0)
