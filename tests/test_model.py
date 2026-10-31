"""
Automated unit tests for pipeline fitting, persistence, and probability consistency.
"""

import tempfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.config import FEATURE_COLS, MODEL_FILE_PATH
from src.predict import load_model
from src.preprocessing import build_pipeline


@pytest.fixture
def small_training_data():
    """Generates a small in-memory dataset fixture for pipeline fitting."""
    np.random.seed(42)
    n = 30
    df = pd.DataFrame(
        {
            "ApplicantIncome": np.random.randint(2000, 10000, size=n),
            "CoapplicantIncome": np.random.randint(0, 4000, size=n),
            "LoanAmount": np.random.randint(50, 400, size=n),
            "Loan_Amount_Term": np.random.choice([180, 360], size=n),
            "Credit_History": np.random.choice(["1", "0"], size=n),
            "Gender": np.random.choice(["Male", "Female"], size=n),
            "Self_Employed": np.random.choice(["Yes", "No"], size=n),
            "Married": np.random.choice(["Yes", "No"], size=n),
            "Education": np.random.choice(["Graduate", "Not Graduate"], size=n),
            "Dependents": np.random.choice(["0", "1", "2", "3"], size=n),
            "Property_Area": np.random.choice(["Urban", "Semiurban", "Rural"], size=n),
        }
    )
    y = np.random.choice([0, 1], size=n)
    return df, y


def test_model_fit_save_and_load(small_training_data):
    """Verifies that a pipeline fits on a fixture, serializes to joblib, and loads correctly."""
    X, y = small_training_data
    pipeline = build_pipeline(classifier=DecisionTreeClassifier(max_depth=3, random_state=42))
    pipeline.fit(X, y)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_model_file = Path(tmp_dir) / "test_model.joblib"
        metadata = {"sklearn_version": "1.0", "custom_test": True}
        joblib.dump({"pipeline": pipeline, "metadata": metadata}, tmp_model_file)

        loaded_pipe, loaded_meta = load_model(tmp_model_file)
        assert hasattr(loaded_pipe, "predict")
        assert hasattr(loaded_pipe, "predict_proba")
        assert loaded_meta.get("custom_test") is True

        # Test prediction consistency
        sample = X.iloc[[0]]
        orig_pred = pipeline.predict(sample)
        loaded_pred = loaded_pipe.predict(sample)
        assert orig_pred[0] == loaded_pred[0]

        # Verify probabilities sum to 1
        probs = loaded_pipe.predict_proba(sample)[0]
        assert np.isclose(probs.sum(), 1.0)


def test_production_model_loads_if_present():
    """If the production model artifact models/loan_model.joblib exists, verify it loads."""
    if MODEL_FILE_PATH.exists():
        pipe, meta = load_model(MODEL_FILE_PATH)
        assert hasattr(pipe, "predict")
        assert hasattr(pipe, "predict_proba")
        assert "sklearn_version" in meta
        assert "metrics" in meta
